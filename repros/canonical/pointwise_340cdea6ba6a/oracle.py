"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 GPT-Neo head layout clone directly as the final contiguous `[4096, 2048]` storage and returns the sibling `[2048, 4096]` transpose as a metadata view of that storage, whereas Inductor lowers the view, dtype round trip, two permutes, clone, view, and returned transpose through its generic layout-copy scheduler; Inductor cannot do this today because its pointwise/layout planner does not recognize this fixed head transpose as a direct tiled materialization with the aliasing returned view preserved; the fix is NEW_PATTERN: add a shape-specialized head-layout clone lowering that chooses the clone storage as the single materialization target and returns view aliases from it."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.autotune(
    configs=[
        triton.Config({"MODE": 0, "BLOCK_M": 1, "BLOCK_N": 1024}, num_warps=4, num_stages=4),
        triton.Config({"MODE": 0, "BLOCK_M": 1, "BLOCK_N": 2048}, num_warps=8, num_stages=4),
        triton.Config({"MODE": 0, "BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4, num_stages=4),
        triton.Config({"MODE": 0, "BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"MODE": 0, "BLOCK_M": 16, "BLOCK_N": 256}, num_warps=8, num_stages=4),
        triton.Config({"MODE": 0, "BLOCK_M": 32, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"MODE": 1, "BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=1),
        triton.Config({"MODE": 1, "BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=1),
        triton.Config({"MODE": 1, "BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"MODE": 1, "BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=1),
        triton.Config({"MODE": 1, "BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"MODE": 1, "BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=1),
    ],
    key=[],
)
@triton.jit
def _head_layout_clone_kernel(
    input_ptr,
    output_ptr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    if MODE == 0:
        y = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        x = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)

        d = y % 128
        a = y // 128
        values = tl.load(input_ptr + d[:, None] + x[None, :] * 128 + a[:, None] * 262144)
        tl.store(output_ptr + x[None, :] + y[:, None] * 2048, values)
    else:
        ab = tl.program_id(0)
        d_block = tl.program_id(1)
        c_block = tl.program_id(2)

        a = ab // 16
        b = ab - a * 16
        d = d_block * BLOCK_M + tl.arange(0, BLOCK_M)
        c = c_block * BLOCK_N + tl.arange(0, BLOCK_N)

        load_offsets = a * 262144 + b * 16384 + c[:, None] * 128 + d[None, :]
        values = tl.load(input_ptr + load_offsets)

        store_offsets = (a * 128 + d[:, None]) * 2048 + b * 128 + c[None, :]
        tl.store(output_ptr + store_offsets, tl.trans(values))


@oracle_impl(hardware="B200", point="e6f344ac")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1, shape_param_2 = inputs
    output = torch.empty_strided(
        (int(shape_param_2[0]), int(shape_param_2[1])),
        (int(shape_param_2[1]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    grid = lambda meta: (
        triton.cdiv(2048, meta["BLOCK_N"]) if meta["MODE"] == 0 else 512,
        triton.cdiv(4096, meta["BLOCK_M"]) if meta["MODE"] == 0 else triton.cdiv(128, meta["BLOCK_M"]),
        1 if meta["MODE"] == 0 else triton.cdiv(128, meta["BLOCK_N"]),
    )
    _head_layout_clone_kernel[grid](
        arg0_1,
        output,
    )
    return output, output.permute(1, 0)
