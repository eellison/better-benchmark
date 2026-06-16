"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete OPT bf16 masked materialization plus sibling column-sum scope by applying `arg0 <= 0 ? 0 : arg1` once, writing the returned contiguous `[8192,3072]` bf16 producer, returning its metadata-only `[3072,8192]` permute alias, and accumulating the dim-0 fp32 sum from the same bf16-rounded producer values before the captured bf16-to-fp32 round trip, whereas Inductor lowers the le/full/where/permute/sum/cast chain through generic producer materialization and reduction schedules; Inductor cannot do this today because its scheduler does not build an alias-aware full-scope materialize-plus-column-reduction plan for a visible bf16 producer with a dependent reduction and layout-only sibling output; the fix is SCHEDULER_FUSION: teach reduction scheduling to keep the masked producer virtual, emit the visible tensor and alias, and finalize the compatible column reduction from the same traversal while preserving explicit bf16 rounding boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _masked_materialize_partial_sum_kernel(
    pred_ptr,
    source_ptr,
    where_ptr,
    partial_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_tile = tl.program_id(0)
    col_tile = tl.program_id(1)
    rows = row_tile * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets = rows[:, None] * N + cols[None, :]

    pred = tl.load(pred_ptr + offsets).to(tl.float32)
    source = tl.load(source_ptr + offsets)
    selected = tl.where(pred <= 0.0, 0.0, source).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(where_ptr + offsets, selected)
    partial = tl.sum(selected.to(tl.float32), axis=0)
    tl.store(partial_ptr + row_tile * N + cols, partial)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    sum_ptr,
    NUM_ROW_TILES: tl.constexpr,
    N: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    tiles = tl.arange(0, BLOCK_TILES)
    partials = tl.load(
        partial_ptr + tiles[:, None] * N + cols[None, :],
        mask=tiles[:, None] < NUM_ROW_TILES,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(partials, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_ptr + cols, rounded)


@oracle_impl(
    hardware="B200",
    point="58ff2bc5",
    ROW_BLOCK=128,
    BLOCK_N=256,
    FINAL_BLOCK_N=32,
    producer_warps=4,
    final_warps=4,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_N: int,
    FINAL_BLOCK_N: int,
    producer_warps: int,
    final_warps: int,
):
    arg0_1, arg1_1, _shape_param_0 = inputs
    m = int(arg1_1.shape[0])
    n = int(arg1_1.shape[1])
    num_row_tiles = triton.cdiv(m, ROW_BLOCK)

    where = torch.empty_strided((m, n), (n, 1), device=arg1_1.device, dtype=torch.bfloat16)
    partial = torch.empty_strided(
        (num_row_tiles, n),
        (n, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_0),
        (1,),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _masked_materialize_partial_sum_kernel[(num_row_tiles, triton.cdiv(n, BLOCK_N))](
        arg0_1,
        arg1_1,
        where,
        partial,
        M=m,
        N=n,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_N=BLOCK_N,
        num_warps=producer_warps,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(n, FINAL_BLOCK_N),)](
        partial,
        sum_out,
        NUM_ROW_TILES=num_row_tiles,
        N=n,
        BLOCK_TILES=triton.next_power_of_2(num_row_tiles),
        BLOCK_N=FINAL_BLOCK_N,
        num_warps=final_warps,
        num_stages=3,
    )

    return where, where.permute(1, 0), sum_out
