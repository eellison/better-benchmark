"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete bf16 DeiT class-token select-scatter layer-norm-backward return tuple by deriving the token-0 row reductions directly from the `[128,192]` bf16 source, emitting the required f32 full `[128,197,192]` gradient tensor, bf16 `[25216,192]` view and `[192,25216]` permute alias, and accumulating all three returned channel reductions from the same sparse producer, whereas Inductor materializes the dense zero/select_scatter tensor and schedules the row reductions, sibling channel reductions, bf16 cast/view/permute, and final bf16-rounded sum as generic dense work; Inductor cannot do this today because scheduler/codegen does not model zero-fill `select_scatter` as a structured sparse token producer feeding row-reduction epilogues, materialized side outputs, and sibling reductions with the explicit bf16 rounding boundary; the fix is SCATTER_REDUCE: lower single-token select-scatter into direct sparse row epilogues plus required inactive-row stores and compatible channel reductions without materializing the dense producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
HIDDEN = 192
ROWS = BATCH * TOKENS
ACTIVE_ROWS = BATCH


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _token0_rows_kernel(
    source_ptr,
    gamma_ptr,
    xhat_ptr,
    scale_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    partial_sum3_ptr,
    partial_sum4_ptr,
    partial_sum5_ptr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < 192

    source = tl.load(source_ptr + batch * 192 + cols, mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    row = batch * 197
    offsets = row * 192 + cols
    xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + row).to(tl.float32)

    weighted = _f32_mul(source, gamma)
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
    weighted_xhat = _f32_mul(weighted, xhat)
    row_dot = tl.sum(tl.where(mask, weighted_xhat, 0.0), axis=0)

    centered = _f32_sub(_f32_mul(weighted, 192.0), row_sum)
    centered = _f32_sub(centered, _f32_mul(xhat, row_dot))
    grad = _f32_mul(scale, centered)
    grad_bf16 = grad.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out_f32_ptr + offsets, grad, mask=mask)
    tl.store(out_bf16_ptr + offsets, grad_bf16, mask=mask)

    partial_offsets = batch * 192 + cols
    tl.store(partial_sum3_ptr + partial_offsets, _f32_mul(source, xhat), mask=mask)
    tl.store(partial_sum4_ptr + partial_offsets, source, mask=mask)
    tl.store(partial_sum5_ptr + partial_offsets, grad_bf16.to(tl.float32), mask=mask)


@triton.jit
def _zero_inactive_rows_kernel(
    out_f32_ptr,
    out_bf16_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    tail_row = offsets // 192
    col = offsets - tail_row * 192
    batch = tail_row // 196
    token_tail = tail_row - batch * 196
    row = batch * 197 + token_tail + 1
    out_offsets = row * 192 + col
    zero = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(out_f32_ptr + out_offsets, zero, mask=mask)
    tl.store(out_bf16_ptr + out_offsets, zero.to(tl.bfloat16), mask=mask)


@triton.jit
def _finalize_channel_sums_kernel(
    partial_sum3_ptr,
    partial_sum4_ptr,
    partial_sum5_ptr,
    out_sum3_ptr,
    out_sum4_ptr,
    out_sum5_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = tl.arange(0, BLOCK_ROWS)
    mask = (rows[:, None] < 128) & (cols[None, :] < 192)
    offsets = rows[:, None] * 192 + cols[None, :]

    vals3 = tl.load(partial_sum3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    vals4 = tl.load(partial_sum4_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    vals5 = tl.load(partial_sum5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sum3 = tl.sum(vals3, axis=0)
    sum4 = tl.sum(vals4, axis=0)
    sum5 = tl.sum(vals5, axis=0).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    col_mask = cols < 192
    tl.store(out_sum3_ptr + cols, sum3, mask=col_mask)
    tl.store(out_sum4_ptr + cols, sum4, mask=col_mask)
    tl.store(out_sum5_ptr + cols, sum5, mask=col_mask)


# (T([128,192], bf16), T([192], f32), T([128,197,192], f32), T([128,197,1], f32), S([128,197,192]), S([25216,192]), S([192]))
@oracle_impl(hardware="B200", point="7639f983", BLOCK_C=256, ZERO_BLOCK=1024, FINAL_BLOCK_C=16, num_warps=8, zero_warps=8, final_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    ZERO_BLOCK: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
    zero_warps: int,
    final_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    device = arg0_1.device
    out_f32 = torch.empty_strided(
        (BATCH, TOKENS, HIDDEN),
        (TOKENS * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    out_bf16 = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial_sum3 = torch.empty((ACTIVE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    partial_sum4 = torch.empty((ACTIVE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    partial_sum5 = torch.empty((ACTIVE_ROWS, HIDDEN), device=device, dtype=torch.float32)

    _token0_rows_kernel[(BATCH,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out_f32,
        out_bf16,
        partial_sum3,
        partial_sum4,
        partial_sum5,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )

    tail_elements = BATCH * (TOKENS - 1) * HIDDEN
    _zero_inactive_rows_kernel[(triton.cdiv(tail_elements, ZERO_BLOCK),)](
        out_f32,
        out_bf16,
        TOTAL=tail_elements,
        BLOCK=ZERO_BLOCK,
        num_warps=zero_warps,
        num_stages=3,
    )

    out_sum3 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum5 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    _finalize_channel_sums_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_C),)](
        partial_sum3,
        partial_sum4,
        partial_sum5,
        out_sum3,
        out_sum4,
        out_sum5,
        BLOCK_ROWS=triton.next_power_of_2(ACTIVE_ROWS),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )

    return out_f32, out_sum3, out_sum4, out_bf16, out_bf16.permute(1, 0), out_sum5
