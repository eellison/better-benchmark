"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Longformer cropped head-layout scope by directly materializing the returned contiguous `[8192,768]` clone, returning the `[768,8192]` transpose as a view of the same storage, and accumulating the fp32 `[768]` column sum with the captured bf16 rounding boundary, whereas Inductor lowers the as_strided/negative-pad/view/permute/clone/layout-view graph and the sibling reduction through generic materialization and reduction schedules; Inductor cannot do this today because the scheduler treats the crop-plus-permute clone as a fusion barrier and does not keep that materialization live for a simultaneous column reduction plus alias-only transpose return; the fix is SCHEDULER_FUSION: teach layout materialization scheduling to sink fixed negative padding and permutes into the clone store while emitting compatible column partials and metadata-only view aliases from the same producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _copy_and_partial_sum_kernel(
    x_ptr,
    out_ptr,
    partial_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_block = tl.program_id(0)
    head = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)
    seq = rows // 8
    batch = rows - seq * 8

    src_offsets = (
        (batch[:, None] * 12 + head) * 98304
        + (seq[:, None] + 256) * 64
        + d[None, :]
    )
    out_offsets = rows[:, None] * 768 + head * 64 + d[None, :]
    values = tl.load(x_ptr + src_offsets).to(tl.float32)
    tl.store(out_ptr + out_offsets, values.to(tl.bfloat16))

    partial = tl.sum(values, axis=0)
    tl.store(partial_ptr + row_block * 768 + head * 64 + d, partial)


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    sum_ptr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    d_block = tl.program_id(0)
    d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    tiles = tl.arange(0, BLOCK_TILES)

    values = tl.load(partial_ptr + tiles[:, None] * 768 + d[None, :]).to(tl.float32)
    acc = tl.sum(values, axis=0)
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(sum_ptr + d, rounded)


# a120887f: (T([9437184], bf16), S([96,1536,64]), S([98304,64,1]), ...)
@oracle_impl(hardware="B200", point="a120887f", BLOCK_M=32, BLOCK_D=64, FINAL_BLOCK_D=16)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_D: int, FINAL_BLOCK_D: int):
    arg0_1, *_shape_params = inputs
    out = torch.empty_strided(
        (8192, 768),
        (768, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (triton.cdiv(8192, BLOCK_M), 768),
        (768, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        (768,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _copy_and_partial_sum_kernel[(triton.cdiv(8192, BLOCK_M), 12)](
        arg0_1,
        out,
        partial,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=4,
    )
    _finalize_sum_kernel[(triton.cdiv(768, FINAL_BLOCK_D),)](
        partial,
        sum_out,
        BLOCK_TILES=triton.cdiv(8192, BLOCK_M),
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=8,
        num_stages=4,
    )
    return out, out.permute(1, 0), sum_out
