"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DINOv2 one-token select_scatter LayerNorm-backward return tuple by deriving the token-0 row reductions directly from the `[128,768]` source, zero-filling the inactive 1369 token lanes, emitting the required f32 `[128,1370,768]` gradient tensor, bf16 `[175360,768]` projection view plus transpose alias, and accumulating all four returned channel reductions from the sparse producer, whereas Inductor materializes the dense zero/select_scatter tensor and schedules the row reductions, sibling channel reductions, bf16 cast/view/permute, and final bf16-rounded sum as separate generic work; Inductor cannot do this today because scheduler/codegen does not model zero-fill select_scatter as a structured scatter-reduce producer with sparse token lanes, full side-output stores, and sibling reductions; the fix is SCATTER_REDUCE: add a structured select_scatter lowering that maps sparse token sources directly into row-reduction epilogues, emits required materialized full outputs, and accumulates compatible channel reductions without materializing the dense producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 1370
HIDDEN = 768
ROWS = BATCH * TOKENS
ACTIVE_ROWS = BATCH
PADDED_TOKEN_STRIDE = 1376


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
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
def _token0_kernel(
    source_ptr,
    gamma_ptr,
    xhat_ptr,
    scale_ptr,
    projection_source_ptr,
    projection_weight_ptr,
    out_f32_ptr,
    out_bf16_ptr,
    partial_x_xhat_ptr,
    partial_x_ptr,
    partial_grad_proj_source_ptr,
    partial_projected_sum_ptr,
    BLOCK_C: tl.constexpr,
):
    batch = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < 768

    source = tl.load(source_ptr + batch * 768 + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    row_base = batch * 1370 * 768
    xhat = tl.load(xhat_ptr + row_base + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + batch * 1376).to(tl.float32)

    weighted = _f32_mul(source, gamma)
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
    row_dot = tl.sum(tl.where(mask, _f32_mul(weighted, xhat), 0.0), axis=0)

    centered = _f32_sub(_f32_mul(weighted, 768.0), row_sum)
    centered = _f32_sub(centered, _f32_mul(xhat, row_dot))
    grad = _f32_mul(scale, centered)

    projection_source = tl.load(
        projection_source_ptr + row_base + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    projection_weight = tl.load(projection_weight_ptr + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    projected_bf16 = _f32_mul(grad, projection_weight).to(
        tl.bfloat16,
        fp_downcast_rounding="rtne",
    )

    tl.store(out_f32_ptr + row_base + cols, grad, mask=mask)
    tl.store(out_bf16_ptr + row_base + cols, projected_bf16, mask=mask)

    partial_offsets = batch * 768 + cols
    tl.store(partial_x_xhat_ptr + partial_offsets, _f32_mul(source, xhat), mask=mask)
    tl.store(partial_x_ptr + partial_offsets, source, mask=mask)
    tl.store(
        partial_grad_proj_source_ptr + partial_offsets,
        _f32_mul(grad, projection_source),
        mask=mask,
    )
    tl.store(
        partial_projected_sum_ptr + partial_offsets,
        projected_bf16.to(tl.float32),
        mask=mask,
    )


@triton.jit
def _zero_inactive_rows_kernel(
    out_f32_ptr,
    out_bf16_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    tail_row = offsets // 768
    col = offsets - tail_row * 768
    batch = tail_row // 1369
    token_tail = tail_row - batch * 1369
    row = batch * 1370 + token_tail + 1
    out_offsets = row * 768 + col
    zero = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(out_f32_ptr + out_offsets, zero, mask=mask)
    tl.store(out_bf16_ptr + out_offsets, zero.to(tl.bfloat16), mask=mask)


@triton.jit
def _finalize_channel_sums_kernel(
    partial_x_xhat_ptr,
    partial_x_ptr,
    partial_grad_proj_source_ptr,
    partial_projected_sum_ptr,
    out_x_xhat_ptr,
    out_x_ptr,
    out_grad_proj_source_ptr,
    out_projected_sum_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = tl.arange(0, BLOCK_ROWS)
    mask = (rows[:, None] < 128) & (cols[None, :] < 768)
    offsets = rows[:, None] * 768 + cols[None, :]

    x_xhat = tl.sum(
        tl.load(partial_x_xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    x = tl.sum(
        tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    grad_proj_source = tl.sum(
        tl.load(partial_grad_proj_source_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    projected_sum = tl.sum(
        tl.load(partial_projected_sum_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )

    col_mask = cols < 768
    tl.store(out_x_xhat_ptr + cols, x_xhat, mask=col_mask)
    tl.store(out_x_ptr + cols, x, mask=col_mask)
    tl.store(out_grad_proj_source_ptr + cols, grad_proj_source, mask=col_mask)
    tl.store(
        out_projected_sum_ptr + cols,
        projected_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
        mask=col_mask,
    )


# ef8b4ed0: DINOv2 one-token select_scatter LN-backward tuple.
@oracle_impl(
    hardware="B200",
    point="ef8b4ed0",
    BLOCK_C=1024,
    ZERO_BLOCK=1024,
    FINAL_BLOCK_C=16,
    token_warps=8,
    zero_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    ZERO_BLOCK: int,
    FINAL_BLOCK_C: int,
    token_warps: int,
    zero_warps: int,
    final_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4

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
    partial_x_xhat = torch.empty((ACTIVE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    partial_x = torch.empty((ACTIVE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    partial_grad_proj_source = torch.empty(
        (ACTIVE_ROWS, HIDDEN),
        device=device,
        dtype=torch.float32,
    )
    partial_projected_sum = torch.empty(
        (ACTIVE_ROWS, HIDDEN),
        device=device,
        dtype=torch.float32,
    )

    _token0_kernel[(BATCH,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        out_f32,
        out_bf16,
        partial_x_xhat,
        partial_x,
        partial_grad_proj_source,
        partial_projected_sum,
        BLOCK_C=BLOCK_C,
        num_warps=token_warps,
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

    out_x_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_x = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_grad_proj_source = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_projected_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    _finalize_channel_sums_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_C),)](
        partial_x_xhat,
        partial_x,
        partial_grad_proj_source,
        partial_projected_sum,
        out_x_xhat,
        out_x,
        out_grad_proj_source,
        out_projected_sum,
        BLOCK_ROWS=triton.next_power_of_2(ACTIVE_ROWS),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
        num_stages=3,
    )

    return (
        out_f32,
        out_x_xhat,
        out_x,
        out_grad_proj_source,
        out_bf16,
        out_bf16.permute(1, 0),
        out_projected_sum,
    )
