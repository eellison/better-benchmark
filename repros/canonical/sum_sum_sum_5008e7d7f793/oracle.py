"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle keeps the ViT layer-norm backward row reduction, dense bf16 patch materialization, token-wise batch sums, and global column reductions in one specialized schedule. Inductor lowers the row-local hidden reductions, broadcasted residual arithmetic, sliced bf16 patch conversion/view, token batch reductions, and final column reductions as separate generic regions, which leaves repeated reads of the 128x1370x768 activation-sized tensors and cannot reuse the row scalars that define the backward formula. The fix is COOPERATIVE_SPLIT_K: emit a row-group producer that computes the hidden reductions once, writes the observable bf16 patch payload once, stores compact partials for the batch and column sums, and finalize those partials with the required bf16 rounding boundary for the patch sum."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 1370
PATCH_TOKENS = 1369
CHANNELS = 768
PADDED_TOKEN_STRIDE = 1376
CHANNEL_FACTOR = 768.0
INV_CHANNEL_FACTOR = 1.0 / 768.0


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_group_kernel(
    x_ptr,
    weight_ptr,
    residual_ptr,
    token_bias_ptr,
    row_center_ptr,
    row_scale_ptr,
    grad_base_ptr,
    bf16_patch_ptr,
    token_partials_ptr,
    column_partials_ptr,
    BATCH_: tl.constexpr,
    TOKENS_: tl.constexpr,
    PATCH_TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    PADDED_TOKEN_STRIDE_: tl.constexpr,
    NUM_TOKEN_GROUPS: tl.constexpr,
    B_BLOCK: tl.constexpr,
    T_GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
    CHANNEL_FACTOR_: tl.constexpr,
    INV_CHANNEL_FACTOR_: tl.constexpr,
):
    b_group = tl.program_id(0)
    t_group = tl.program_id(1)
    b_offsets = b_group * B_BLOCK + tl.arange(0, B_BLOCK)
    cols = tl.arange(0, BLOCK_C)
    b_mask = b_offsets < BATCH_
    c_mask = cols < CHANNELS_
    matrix_base = b_offsets[:, None] * TOKENS_ * CHANNELS_ + cols[None, :]
    scalar_base = b_offsets * PADDED_TOKEN_STRIDE_

    weight = tl.load(weight_ptr + cols, mask=c_mask, other=0.0).to(tl.float32)
    channel_factor = tl.full((B_BLOCK, BLOCK_C), CHANNEL_FACTOR_, tl.float32)

    acc_prod = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_src = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_patch = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_t in tl.range(0, T_GROUP):
        token = t_group * T_GROUP + local_t
        element_offsets = matrix_base + token * CHANNELS_
        mask = b_mask[:, None] & c_mask[None, :]

        x = tl.load(x_ptr + element_offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = _mul_rn(x, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)

        residual = tl.load(residual_ptr + element_offsets, mask=mask, other=0.0).to(tl.float32)
        token_bias = tl.load(token_bias_ptr + token * CHANNELS_ + cols, mask=c_mask, other=0.0).to(tl.float32)
        center = tl.load(row_center_ptr + scalar_base + token, mask=b_mask, other=0.0).to(tl.float32)
        scale = tl.load(row_scale_ptr + scalar_base + token, mask=b_mask, other=0.0).to(tl.float32)
        shifted = _sub_rn(_add_rn(residual, token_bias[None, :]), center[:, None])
        normed = _mul_rn(shifted, scale[:, None])

        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, normed), 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, channel_factor), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(normed, row_dot[:, None]))
        grad = _mul_rn(_mul_rn(scale, INV_CHANNEL_FACTOR_)[:, None], centered)
        out = _add_rn(
            tl.load(grad_base_ptr + element_offsets, mask=mask, other=0.0).to(tl.float32),
            grad,
        )
        out_bf16 = out.to(tl.bfloat16, fp_downcast_rounding="rtne")

        patch_offsets = (b_offsets[:, None] * PATCH_TOKENS_ + (token - 1)) * CHANNELS_ + cols[None, :]
        patch_mask = mask & (token > 0)
        tl.store(bf16_patch_ptr + patch_offsets, out_bf16, mask=patch_mask)

        token_partial = tl.sum(tl.where(mask, out, 0.0), axis=0)
        tl.store(
            token_partials_ptr + (b_group * TOKENS_ + token) * CHANNELS_ + cols,
            token_partial,
            mask=c_mask,
        )

        acc_prod += tl.sum(tl.where(mask, _mul_rn(x, normed), 0.0), axis=0)
        acc_src += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_patch += tl.sum(tl.where(patch_mask, out_bf16.to(tl.float32), 0.0), axis=0)

    column_group = b_group * NUM_TOKEN_GROUPS + t_group
    column_base = column_group * 3 * CHANNELS_ + cols
    tl.store(column_partials_ptr + column_base, acc_prod, mask=c_mask)
    tl.store(column_partials_ptr + column_base + CHANNELS_, acc_src, mask=c_mask)
    tl.store(column_partials_ptr + column_base + 2 * CHANNELS_, acc_patch, mask=c_mask)


@triton.jit
def _finalize_token_kernel(
    token_partials_ptr,
    token_sum_ptr,
    token0_sum_ptr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    NUM_BATCH_GROUPS: tl.constexpr,
    BATCH_GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, BATCH_GROUP_BLOCK)[:, None]
    mask = (groups < NUM_BATCH_GROUPS) & (cols[None, :] < CHANNELS_)
    offsets = (groups * TOKENS_ + token) * CHANNELS_ + cols[None, :]
    total = tl.sum(tl.load(token_partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

    col_mask = cols < CHANNELS_
    tl.store(token_sum_ptr + token * CHANNELS_ + cols, total, mask=col_mask)
    tl.store(token0_sum_ptr + cols, total, mask=col_mask & (token == 0))


@triton.jit
def _global_stage1_kernel(
    column_partials_ptr,
    stage_ptr,
    NUM_COLUMN_GROUPS: tl.constexpr,
    CHANNELS_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    chunk = tl.program_id(1)
    groups = chunk * GROUP_BLOCK + tl.arange(0, GROUP_BLOCK)[:, None]
    mask = (groups < NUM_COLUMN_GROUPS) & (cols[None, :] < CHANNELS_)
    offsets = groups * 3 * CHANNELS_ + cols[None, :]

    prod = tl.sum(tl.load(column_partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    src = tl.sum(tl.load(column_partials_ptr + offsets + CHANNELS_, mask=mask, other=0.0).to(tl.float32), axis=0)
    patch = tl.sum(tl.load(column_partials_ptr + offsets + 2 * CHANNELS_, mask=mask, other=0.0).to(tl.float32), axis=0)

    col_mask = cols < CHANNELS_
    stage_base = chunk * 3 * CHANNELS_ + cols
    tl.store(stage_ptr + stage_base, prod, mask=col_mask)
    tl.store(stage_ptr + stage_base + CHANNELS_, src, mask=col_mask)
    tl.store(stage_ptr + stage_base + 2 * CHANNELS_, patch, mask=col_mask)


@triton.jit
def _global_stage2_kernel(
    stage_ptr,
    prod_out_ptr,
    src_out_ptr,
    patch_sum_out_ptr,
    NUM_STAGE_GROUPS: tl.constexpr,
    STAGE_GROUP_BLOCK: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, STAGE_GROUP_BLOCK)[:, None]
    mask = (groups < NUM_STAGE_GROUPS) & (cols[None, :] < CHANNELS_)
    offsets = groups * 3 * CHANNELS_ + cols[None, :]

    prod = tl.sum(tl.load(stage_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    src = tl.sum(tl.load(stage_ptr + offsets + CHANNELS_, mask=mask, other=0.0).to(tl.float32), axis=0)
    patch = tl.sum(tl.load(stage_ptr + offsets + 2 * CHANNELS_, mask=mask, other=0.0).to(tl.float32), axis=0)
    patch = patch.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    col_mask = cols < CHANNELS_
    tl.store(prod_out_ptr + cols, prod, mask=col_mask)
    tl.store(src_out_ptr + cols, src, mask=col_mask)
    tl.store(patch_sum_out_ptr + cols, patch, mask=col_mask)


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="9437dc93",
    B_BLOCK=16,
    T_GROUP=5,
    BLOCK_C=1024,
    TOKEN_BLOCK_C=64,
    GLOBAL_GROUP_BLOCK=256,
    GLOBAL_BLOCK_C=16,
    FINAL_BLOCK_C=32,
    row_warps=8,
    token_warps=2,
    global_warps=8,
    final_warps=2,
)
def oracle_forward(
    inputs,
    *,
    B_BLOCK: int,
    T_GROUP: int,
    BLOCK_C: int,
    TOKEN_BLOCK_C: int,
    GLOBAL_GROUP_BLOCK: int,
    GLOBAL_BLOCK_C: int,
    FINAL_BLOCK_C: int,
    row_warps: int,
    token_warps: int,
    global_warps: int,
    final_warps: int,
):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        *_shape_params,
    ) = inputs

    device = arg0.device
    num_batch_groups = triton.cdiv(BATCH, B_BLOCK)
    num_token_groups = triton.cdiv(TOKENS, T_GROUP)
    num_column_groups = num_batch_groups * num_token_groups
    num_stage_groups = triton.cdiv(num_column_groups, GLOBAL_GROUP_BLOCK)

    token_partials = torch.empty(
        (num_batch_groups, TOKENS, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    column_partials = torch.empty(
        (num_column_groups, 3, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    stage = torch.empty((num_stage_groups, 3, CHANNELS), device=device, dtype=torch.float32)

    token_sum = torch.empty_strided(
        (1, TOKENS, CHANNELS),
        (TOKENS * CHANNELS, CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    token0_sum = torch.empty_strided(
        (1, 1, CHANNELS),
        (CHANNELS, CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    bf16_patch = torch.empty_strided(
        (BATCH, PATCH_TOKENS, CHANNELS),
        (PATCH_TOKENS * CHANNELS, CHANNELS, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    prod_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    src_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    patch_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _row_group_kernel[(num_batch_groups, num_token_groups)](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        bf16_patch,
        token_partials,
        column_partials,
        BATCH_=BATCH,
        TOKENS_=TOKENS,
        PATCH_TOKENS_=PATCH_TOKENS,
        CHANNELS_=CHANNELS,
        PADDED_TOKEN_STRIDE_=PADDED_TOKEN_STRIDE,
        NUM_TOKEN_GROUPS=num_token_groups,
        B_BLOCK=B_BLOCK,
        T_GROUP=T_GROUP,
        BLOCK_C=BLOCK_C,
        CHANNEL_FACTOR_=CHANNEL_FACTOR,
        INV_CHANNEL_FACTOR_=INV_CHANNEL_FACTOR,
        num_warps=row_warps,
    )

    _finalize_token_kernel[(TOKENS, triton.cdiv(CHANNELS, TOKEN_BLOCK_C))](
        token_partials,
        token_sum,
        token0_sum,
        TOKENS_=TOKENS,
        CHANNELS_=CHANNELS,
        NUM_BATCH_GROUPS=num_batch_groups,
        BATCH_GROUP_BLOCK=_ceil_pow2(num_batch_groups),
        BLOCK_C=TOKEN_BLOCK_C,
        num_warps=token_warps,
    )

    _global_stage1_kernel[(triton.cdiv(CHANNELS, GLOBAL_BLOCK_C), num_stage_groups)](
        column_partials,
        stage,
        NUM_COLUMN_GROUPS=num_column_groups,
        CHANNELS_=CHANNELS,
        GROUP_BLOCK=GLOBAL_GROUP_BLOCK,
        BLOCK_C=GLOBAL_BLOCK_C,
        num_warps=global_warps,
    )
    _global_stage2_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        stage,
        prod_out,
        src_out,
        patch_sum,
        NUM_STAGE_GROUPS=num_stage_groups,
        STAGE_GROUP_BLOCK=_ceil_pow2(num_stage_groups),
        CHANNELS_=CHANNELS,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )

    patch_view = bf16_patch.permute(0, 2, 1).view(BATCH, CHANNELS, 37, 37)
    return prod_out, src_out, token_sum, token0_sum, patch_view, patch_sum
