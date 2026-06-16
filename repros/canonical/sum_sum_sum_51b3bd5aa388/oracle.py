"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DeiT bf16 LayerNorm-backward fanout, including the bf16 flat input view, row-local hidden-size reductions, global `x * norm` and `x` channel sums, token-wise f32 batch sums, class-token sum, returned bf16 patch image view, and the final bf16 patch-view channel sum, whereas Inductor lowers the row reductions, dependent pointwise layernorm-backward expression, token reductions, view/cast stores, and sibling channel reductions as separate generic producer/consumer regions. Inductor cannot do this today because its scheduler lacks a cooperative split-K multi-output reduction template that shares row-local layernorm scalars with reductions over batch, token, and patch-view axes while preserving the visible bf16 cast and layout boundary. The fix is COOPERATIVE_SPLIT_K: compute compact per-row summaries once, then use token-wise cooperative reductions to emit all visible side outputs and final channel reductions without materializing the f32 layernorm-backward tensor."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
CHANNELS = 192
ROWS = BATCH * TOKENS
PATCH_TOKENS = TOKENS - 1
INV_CHANNELS = 1.0 / CHANNELS


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
def _row_summary_kernel(
    x_ptr,
    weight_ptr,
    source_ptr,
    pos_ptr,
    mean_ptr,
    rsqrt_ptr,
    row_sum_ptr,
    row_dot_ptr,
    partial_x_norm_ptr,
    partial_x_ptr,
    ROWS_: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < CHANNELS_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_norm = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.static_range(0, ROW_SPLIT, XBLOCK):
        rows = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = rows < ROWS_
        token = rows % TOKENS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = rows[:, None] * CHANNELS_ + c[None, :]
        pos_offsets = token[:, None] * CHANNELS_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        pos = tl.load(pos_ptr + pos_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        rsqrt = tl.load(rsqrt_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        add = _f32_add(source, pos)
        centered = _f32_sub(add, mean[:, None])
        norm = _f32_mul(centered, rsqrt[:, None])
        weighted = _f32_mul(x, weight[None, :])
        weighted_norm = _f32_mul(weighted, norm)

        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted_norm, 0.0), axis=1)
        tl.store(row_sum_ptr + rows, row_sum, mask=row_mask)
        tl.store(row_dot_ptr + rows, row_dot, mask=row_mask)

        acc_x_norm += tl.sum(tl.where(mask, _f32_mul(x, norm), 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)

    partial_offsets = pid * CHANNELS_ + c
    tl.store(partial_x_norm_ptr + partial_offsets, acc_x_norm, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)


@triton.jit
def _token_reduce_store_kernel(
    x_ptr,
    weight_ptr,
    source_ptr,
    pos_ptr,
    mean_ptr,
    rsqrt_ptr,
    residual_ptr,
    row_sum_ptr,
    row_dot_ptr,
    token_sum_ptr,
    patch_view_ptr,
    patch_partial_ptr,
    TOKEN_SUM_S0: tl.constexpr,
    TOKEN_SUM_S1: tl.constexpr,
    TOKEN_SUM_S2: tl.constexpr,
    VIEW_S0: tl.constexpr,
    VIEW_S1: tl.constexpr,
    VIEW_S2: tl.constexpr,
    VIEW_S3: tl.constexpr,
    BATCH_: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_CHANNELS_: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    b = tl.arange(0, BLOCK_B)
    c_mask = c < CHANNELS_
    b_mask = b < BATCH_
    row = b * TOKENS_ + token
    mask = b_mask[:, None] & c_mask[None, :]
    offsets = row[:, None] * CHANNELS_ + c[None, :]
    pos_offsets = token * CHANNELS_ + c

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + pos_offsets, mask=c_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row, mask=b_mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + row, mask=b_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + row, mask=b_mask, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + row, mask=b_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    add = _f32_add(source, pos[None, :])
    centered = _f32_sub(add, mean[:, None])
    norm = _f32_mul(centered, rsqrt[:, None])
    weighted = _f32_mul(x, weight[None, :])
    scaled = _f32_mul(weighted, 192.0)
    sub0 = _f32_sub(scaled, row_sum[:, None])
    norm_dot = _f32_mul(norm, row_dot[:, None])
    sub1 = _f32_sub(sub0, norm_dot)
    div = _f32_mul(rsqrt[:, None], INV_CHANNELS_)
    ln_grad = _f32_mul(div, sub1)
    value = _f32_add(residual, ln_grad)

    reduced = tl.sum(tl.where(mask, value, 0.0), axis=0)
    out_offsets = token * TOKEN_SUM_S1 + c
    tl.store(token_sum_ptr + out_offsets, reduced, mask=c_mask)

    if token != 0:
        patch = token - 1
        ph = patch // 14
        pw = patch - ph * 14
        view_offsets = b[:, None] * VIEW_S0 + c[None, :] * VIEW_S1 + ph * VIEW_S2 + pw * VIEW_S3
        view_value = value.to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(patch_view_ptr + view_offsets, view_value, mask=mask)
        patch_sum = tl.sum(tl.where(mask, view_value.to(tl.float32), 0.0), axis=0)
        tl.store(patch_partial_ptr + (token - 1) * CHANNELS_ + c, patch_sum, mask=c_mask)


@triton.jit
def _finalize_vectors_kernel(
    partial_x_norm_ptr,
    partial_x_ptr,
    patch_partial_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    patch_sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    PATCH_TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_PATCH: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c < CHANNELS_

    row_block = tl.arange(0, BLOCK_ROW_BLOCKS)
    row_mask = (row_block[:, None] < NUM_ROW_BLOCKS) & c_mask[None, :]
    row_offsets = row_block[:, None] * CHANNELS_ + c[None, :]
    x_norm = tl.load(partial_x_norm_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    tl.store(out_x_norm_ptr + c, tl.sum(x_norm, axis=0), mask=c_mask)
    tl.store(out_x_ptr + c, tl.sum(x, axis=0), mask=c_mask)

    patch = tl.arange(0, BLOCK_PATCH)
    patch_mask = (patch[:, None] < PATCH_TOKENS_) & c_mask[None, :]
    patch_offsets = patch[:, None] * CHANNELS_ + c[None, :]
    patch_values = tl.load(patch_partial_ptr + patch_offsets, mask=patch_mask, other=0.0).to(tl.float32)
    patch_sum = tl.sum(patch_values, axis=0)
    tl.store(
        patch_sum_ptr + c,
        patch_sum.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32),
        mask=c_mask,
    )


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="c8d56283",
    ROW_SPLIT=8,
    XBLOCK=1,
    ROW_BLOCK_C=256,
    TOKEN_BLOCK_C=32,
    FINAL_BLOCK_C=8,
    row_warps=4,
    token_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROW_SPLIT: int,
    XBLOCK: int,
    ROW_BLOCK_C: int,
    TOKEN_BLOCK_C: int,
    FINAL_BLOCK_C: int,
    row_warps: int,
    token_warps: int,
):
    flat, weight, source, pos, mean, rsqrt, residual, _shape0, _shape1 = inputs
    device = flat.device
    row_blocks = triton.cdiv(ROWS, ROW_SPLIT)

    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    partials = torch.empty((2, row_blocks, CHANNELS), device=device, dtype=torch.float32)
    partial_x_norm = partials[0]
    partial_x = partials[1]

    token_sum = torch.empty_strided((1, TOKENS, CHANNELS), (TOKENS * CHANNELS, CHANNELS, 1), device=device, dtype=torch.float32)
    patch_view = torch.empty_strided((BATCH, CHANNELS, 14, 14), (CHANNELS * PATCH_TOKENS, 1, 14 * CHANNELS, CHANNELS), device=device, dtype=torch.bfloat16)
    patch_partial = torch.empty((PATCH_TOKENS, CHANNELS), device=device, dtype=torch.float32)
    vector_outputs = torch.empty((3, CHANNELS), device=device, dtype=torch.float32)
    out_x_norm = vector_outputs[0]
    out_x = vector_outputs[1]
    patch_sum = vector_outputs[2]

    _row_summary_kernel[(row_blocks,)](
        flat,
        weight,
        source,
        pos,
        mean,
        rsqrt,
        row_sum,
        row_dot,
        partial_x_norm,
        partial_x,
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        CHANNELS_=CHANNELS,
        ROW_SPLIT=ROW_SPLIT,
        XBLOCK=XBLOCK,
        BLOCK_C=ROW_BLOCK_C,
        num_warps=row_warps,
        num_stages=3,
    )
    _token_reduce_store_kernel[(TOKENS, triton.cdiv(CHANNELS, TOKEN_BLOCK_C))](
        flat,
        weight,
        source,
        pos,
        mean,
        rsqrt,
        residual,
        row_sum,
        row_dot,
        token_sum,
        patch_view,
        patch_partial,
        TOKEN_SUM_S0=token_sum.stride(0),
        TOKEN_SUM_S1=token_sum.stride(1),
        TOKEN_SUM_S2=token_sum.stride(2),
        VIEW_S0=patch_view.stride(0),
        VIEW_S1=patch_view.stride(1),
        VIEW_S2=patch_view.stride(2),
        VIEW_S3=patch_view.stride(3),
        BATCH_=BATCH,
        TOKENS_=TOKENS,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
        BLOCK_B=128,
        BLOCK_C=TOKEN_BLOCK_C,
        num_warps=token_warps,
        num_stages=3,
    )
    _finalize_vectors_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partial_x_norm,
        partial_x,
        patch_partial,
        out_x_norm,
        out_x,
        patch_sum,
        NUM_ROW_BLOCKS=row_blocks,
        PATCH_TOKENS_=PATCH_TOKENS,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS=_next_power_of_2(row_blocks),
        BLOCK_PATCH=_next_power_of_2(PATCH_TOKENS),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )

    cls_sum = torch.as_strided(token_sum, (1, 1, CHANNELS), (TOKENS * CHANNELS, CHANNELS, 1))
    return out_x_norm, out_x, token_sum, cls_sum, patch_view, patch_sum
