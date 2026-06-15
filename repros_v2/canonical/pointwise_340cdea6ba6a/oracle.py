"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 GPT-Neo head layout clone directly as the final contiguous `[4096, 2048]` storage and returns the sibling `[2048, 4096]` transpose as a metadata view of that storage, whereas Inductor lowers the view, dtype round trip, two permutes, clone, view, and returned transpose through its generic layout-copy scheduler; Inductor cannot do this today because its pointwise/layout planner does not recognize this fixed head transpose as a direct tiled materialization with the aliasing returned view preserved; the fix is NEW_PATTERN: add a shape-specialized head-layout clone lowering that chooses the clone storage as the single materialization target and returns view aliases from it."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_layout_clone_kernel(
    input_ptr,
    output_ptr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    ab = tl.program_id(0)
    d_block = tl.program_id(1)
    c_block = tl.program_id(2)

    a = ab // 16
    b = ab - a * 16
    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)

    load_offsets = a * 262144 + b * 16384 + c[:, None] * 128 + d[None, :]
    values = tl.load(input_ptr + load_offsets)

    store_offsets = (a * 128 + d[:, None]) * 2048 + b * 128 + c[None, :]
    tl.store(output_ptr + store_offsets, tl.trans(values))


@oracle_impl(hardware="B200", point="e6f344ac", BLOCK_D=128, BLOCK_C=64, num_warps=4, num_stages=1)
def oracle_forward(inputs, *, BLOCK_D: int, BLOCK_C: int, num_warps: int, num_stages: int):
    arg0_1, _shape_param_0, _shape_param_1, shape_param_2 = inputs
    output = torch.empty_strided(
        (int(shape_param_2[0]), int(shape_param_2[1])),
        (int(shape_param_2[1]), 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _head_layout_clone_kernel[(512, triton.cdiv(128, BLOCK_D), triton.cdiv(128, BLOCK_C))](
        arg0_1,
        output,
        BLOCK_D=BLOCK_D,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output, output.permute(1, 0)
