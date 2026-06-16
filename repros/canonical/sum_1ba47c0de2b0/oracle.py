"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GoogleFnet fp32 masked producer once, stores the returned contiguous `[16384, 768]` tensor, returns its metadata-only `[768, 16384]` transpose alias, and accumulates the sibling f32 `[768]` column sum through row-block partials from the same producer values, whereas Inductor schedules the masked pointwise materialization, alias return, and dependent sum as separate generic pointwise/reduction regions over the materialized intermediate; Inductor cannot do this today because its scheduler does not form one alias-aware multi-output producer-store plus partial-reduction plan for a value that is both returned and reduced; the fix is SCHEDULER_FUSION: teach multi-output reduction scheduling to emit a fused store-and-partial-reduce template that preserves the observable base tensor, view alias, f32 scalar-cast boundaries, and final column reduction."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 768
SCALE = 1.1111111640930176


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _store_and_partial_sum_kernel(
    real_pair_ptr,
    residual_ptr,
    mask_ptr,
    out_ptr,
    partials_ptr,
    ROWS_N: tl.constexpr,
    HIDDEN_N: tl.constexpr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    SCALE_: tl.constexpr,
):
    col_block = tl.program_id(0)
    group = tl.program_id(1)

    rows = group * ROWS_PER_GROUP + tl.arange(0, ROWS_PER_GROUP)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < ROWS_N) & (cols[None, :] < HIDDEN_N)
    offsets = rows[:, None] * HIDDEN_N + cols[None, :]

    real = tl.load(real_pair_ptr + offsets * 2, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.float32)

    add = _add_rn_f32(residual, real)
    scaled_keep = _mul_rn_f32(keep, SCALE_)
    value = _mul_rn_f32(add, scaled_keep)

    tl.store(out_ptr + offsets, value, mask=mask)
    partial = tl.sum(tl.where(mask, value, 0.0), axis=0)
    tl.store(partials_ptr + group * HIDDEN_N + cols, partial, mask=cols < HIDDEN_N)


@triton.jit
def _final_sum_kernel(
    partials_ptr,
    sum_out_ptr,
    GROUPS_N: tl.constexpr,
    HIDDEN_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[:, None]
    groups = tl.arange(0, GROUP_BLOCK)[None, :]
    mask = (cols < HIDDEN_N) & (groups < GROUPS_N)

    values = tl.load(
        partials_ptr + groups * HIDDEN_N + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    total = tl.sum(values, axis=1)[:, None]
    tl.store(sum_out_ptr + cols, total, mask=cols < HIDDEN_N)


@oracle_impl(
    hardware="B200",
    point="bdff4e84",
    ROWS_PER_GROUP=64,
    BLOCK_C=64,
    FINAL_BLOCK_C=64,
    store_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    store_warps: int,
    final_warps: int,
):
    real_pair, residual, keep_mask, _shape0, _shape1 = inputs
    del _shape0, _shape1

    groups = triton.cdiv(ROWS, ROWS_PER_GROUP)
    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=residual.device,
        dtype=torch.float32,
    )
    partials = torch.empty_strided(
        (groups, HIDDEN),
        (HIDDEN, 1),
        device=residual.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided((HIDDEN,), (1,), device=residual.device, dtype=torch.float32)

    _store_and_partial_sum_kernel[(triton.cdiv(HIDDEN, BLOCK_C), groups)](
        real_pair,
        residual,
        keep_mask,
        out,
        partials,
        ROWS_N=ROWS,
        HIDDEN_N=HIDDEN,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_C=BLOCK_C,
        SCALE_=SCALE,
        num_warps=store_warps,
        num_stages=3,
    )
    _final_sum_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_C),)](
        partials,
        sum_out,
        GROUPS_N=groups,
        HIDDEN_N=HIDDEN,
        BLOCK_C=FINAL_BLOCK_C,
        GROUP_BLOCK=256,
        num_warps=final_warps,
        num_stages=1,
    )

    return out, out.permute(1, 0), sum_out
