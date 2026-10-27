"""cuTile port of sum_sum_sum_5008e7d7f793: ViT LayerNorm-backward scope.

Four cuTile kernels mirroring the Triton reference:
  1) _row_group_kernel: LN-backward per row-block, emits bf16 patch payload
     + f32 token partials + column partials (prod/src/patch).
  2) _finalize_token_kernel: reduces token partials across batch groups.
  3) _global_stage1_kernel: reduces column partials across groups.
  4) _global_stage2_kernel: reduces stage into prod_out, src_out, patch_sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 1370
PATCH_TOKENS = 1369
CHANNELS = 768
PADDED_TOKEN_STRIDE = 1376
CHANNEL_FACTOR = 768.0
INV_CHANNEL_FACTOR = 1.0 / 768.0


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _row_group_kernel(
    x_ptr,                # bf16 [BATCH, TOKENS, CHANNELS]
    weight_ptr,           # f32 [CHANNELS]
    residual_ptr,         # f32 [BATCH, TOKENS, CHANNELS] = arg2
    token_bias_ptr,       # f32 [1, TOKENS, CHANNELS] = arg3
    row_center_ptr,       # f32 padded [BATCH, PADDED_TOKEN_STRIDE] = arg4 flat
    row_scale_ptr,        # f32 padded [BATCH, PADDED_TOKEN_STRIDE] = arg5 flat
    grad_base_ptr,        # f32 [BATCH, TOKENS, CHANNELS] = arg6
    bf16_patch_ptr,       # bf16 [BATCH, PATCH_TOKENS, CHANNELS]
    token_partials_ptr,   # f32 [NUM_BATCH_GROUPS, TOKENS, CHANNELS]
    column_partials_ptr,  # f32 [NUM_COLUMN_GROUPS*3, CHANNELS]
    BATCH_: ct.Constant[int],
    TOKENS_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    NUM_TOKEN_GROUPS: ct.Constant[int],
    B_BLOCK: ct.Constant[int],
    T_GROUP: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    CHANNEL_FACTOR_: ct.Constant[float],
    INV_CHANNEL_FACTOR_: ct.Constant[float],
):
    b_group = ct.bid(0)
    t_group = ct.bid(1)
    # b_group_offset in units of B_BLOCK -> rows: [b_group*B_BLOCK, +B_BLOCK).

    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    weight = ct.reshape(weight_1d, (1, BLOCK_C))

    c_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask = c_idx < CHANNELS_
    c_mask_2d = ct.reshape(c_mask, (1, BLOCK_C))

    acc_prod = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_src = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_patch = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)

    for local_t in range(T_GROUP):
        token = t_group * T_GROUP + local_t
        if token < TOKENS_:
            # Load B_BLOCK rows of (channel width) at (b_group*B_BLOCK + i, token, :).
            # Treat x as (BATCH*TOKENS, CHANNELS) 2D and load contiguous.
            # But easier: iterate b within group, load (1, BLOCK_C) tile.
            # For efficient tile ops, load one batch at a time.
            row_sum_arr = ct.full(shape=(B_BLOCK, 1), fill_value=0.0, dtype=ct.float32)
            row_dot_arr = ct.full(shape=(B_BLOCK, 1), fill_value=0.0, dtype=ct.float32)
            # Accumulate token_partial across batches in the group.
            tok_acc = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
            # For patch, sum over b (with mask token>0).
            for i in range(B_BLOCK):
                b = b_group * B_BLOCK + i
                if b < BATCH_:
                    # Load x at (b, token, :) → 2D tile with rows=1
                    x_bf = ct.load(x_ptr, index=(b, token, 0),
                                   shape=(1, 1, BLOCK_C),
                                   padding_mode=ct.PaddingMode.ZERO)
                    x = ct.astype(ct.reshape(x_bf, (1, BLOCK_C)), ct.float32)
                    residual = ct.reshape(
                        ct.load(residual_ptr, index=(b, token, 0),
                                shape=(1, 1, BLOCK_C),
                                padding_mode=ct.PaddingMode.ZERO),
                        (1, BLOCK_C),
                    )
                    token_bias = ct.reshape(
                        ct.load(token_bias_ptr, index=(0, token, 0),
                                shape=(1, 1, BLOCK_C),
                                padding_mode=ct.PaddingMode.ZERO),
                        (1, BLOCK_C),
                    )
                    center = ct.load(
                        row_center_ptr, index=(b, token),
                        shape=(1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    scale = ct.load(
                        row_scale_ptr, index=(b, token),
                        shape=(1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    grad_base = ct.reshape(
                        ct.load(grad_base_ptr, index=(b, token, 0),
                                shape=(1, 1, BLOCK_C),
                                padding_mode=ct.PaddingMode.ZERO),
                        (1, BLOCK_C),
                    )
                    weighted = x * weight
                    weighted_m = ct.where(c_mask_2d, weighted, 0.0)
                    row_sum = ct.sum(weighted_m, axis=1, keepdims=True)
                    shifted = residual + token_bias - center
                    normed = shifted * scale
                    row_dot = ct.sum(ct.where(c_mask_2d, weighted * normed, 0.0),
                                     axis=1, keepdims=True)
                    centered = weighted * CHANNEL_FACTOR_ - row_sum
                    centered = centered - normed * row_dot
                    grad = (scale * INV_CHANNEL_FACTOR_) * centered
                    out = grad_base + grad
                    out_bf16 = ct.astype(out, ct.bfloat16)

                    if token != 0:
                        # store patch at (b, token-1, :) 3D destination.
                        # Simpler: view as (BATCH*PATCH_TOKENS, CHANNELS) 2D.
                        patch_row = b * (TOKENS_ - 1) + (token - 1)
                        ct.store(bf16_patch_ptr,
                                 index=(patch_row, 0),
                                 tile=out_bf16)
                        acc_patch = acc_patch + ct.where(
                            c_mask_2d, ct.astype(out_bf16, ct.float32), 0.0,
                        )

                    tok_acc = tok_acc + ct.where(c_mask_2d, out, 0.0)
                    acc_prod = acc_prod + ct.where(c_mask_2d, x * normed, 0.0)
                    acc_src = acc_src + ct.where(c_mask_2d, x, 0.0)

            # Store token partial for (b_group, token, :).
            # Passed as (NUM_BATCH_GROUPS, TOKENS, CHANNELS) but 2D view -> use 2D:
            # store to token_partials as (NUM_BATCH_GROUPS*TOKENS, CHANNELS)
            tp_row = b_group * TOKENS_ + token
            ct.store(token_partials_ptr, index=(tp_row, 0), tile=tok_acc)

    # Store column partials for this (b_group, t_group). Passed as
    # (NUM_COLUMN_GROUPS * 3, CHANNELS) 2D view.
    column_group = b_group * NUM_TOKEN_GROUPS + t_group
    base = column_group * 3
    ct.store(column_partials_ptr, index=(base + 0, 0), tile=acc_prod)
    ct.store(column_partials_ptr, index=(base + 1, 0), tile=acc_src)
    ct.store(column_partials_ptr, index=(base + 2, 0), tile=acc_patch)


@ct.kernel
def _finalize_token_kernel(
    token_partials_ptr,   # f32 [NUM_BATCH_GROUPS*TOKENS, CHANNELS]
    token_sum_ptr,        # f32 [TOKENS, CHANNELS]
    token0_sum_ptr,       # f32 [CHANNELS]
    TOKENS_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    NUM_BATCH_GROUPS: ct.Constant[int],
    BATCH_GROUP_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    token = ct.bid(0)
    c_blk = ct.bid(1)
    acc = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    for g in range(NUM_BATCH_GROUPS):
        row = g * TOKENS_ + token
        tile = ct.load(token_partials_ptr, index=(row, c_blk),
                       shape=(1, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
        acc = acc + tile
    ct.store(token_sum_ptr, index=(token, c_blk), tile=acc)
    if token == 0:
        ct.store(token0_sum_ptr, index=(c_blk,),
                 tile=ct.reshape(acc, (BLOCK_C,)))


@ct.kernel
def _global_stage1_kernel(
    column_partials_ptr,  # f32 [NUM_COLUMN_GROUPS*3, CHANNELS]
    stage_ptr,            # f32 [NUM_STAGE_GROUPS*3, CHANNELS]
    NUM_COLUMN_GROUPS: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    c_blk = ct.bid(1)
    acc_prod = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_src = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_patch = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    for g in range(GROUP_BLOCK):
        cg = chunk * GROUP_BLOCK + g
        if cg < NUM_COLUMN_GROUPS:
            base = cg * 3
            p0 = ct.load(column_partials_ptr, index=(base + 0, c_blk),
                         shape=(1, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
            p1 = ct.load(column_partials_ptr, index=(base + 1, c_blk),
                         shape=(1, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
            p2 = ct.load(column_partials_ptr, index=(base + 2, c_blk),
                         shape=(1, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
            acc_prod = acc_prod + p0
            acc_src = acc_src + p1
            acc_patch = acc_patch + p2
    stbase = chunk * 3
    ct.store(stage_ptr, index=(stbase + 0, c_blk), tile=acc_prod)
    ct.store(stage_ptr, index=(stbase + 1, c_blk), tile=acc_src)
    ct.store(stage_ptr, index=(stbase + 2, c_blk), tile=acc_patch)


@ct.kernel
def _global_stage2_kernel(
    stage_ptr,            # f32 [NUM_STAGE_GROUPS*3, CHANNELS]
    prod_out_ptr,         # f32 [CHANNELS]
    src_out_ptr,          # f32 [CHANNELS]
    patch_sum_out_ptr,    # f32 [CHANNELS]
    NUM_STAGE_GROUPS: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_blk = ct.bid(0)
    acc_prod = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_src = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_patch = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    for g in range(NUM_STAGE_GROUPS):
        base = g * 3
        p0 = ct.load(stage_ptr, index=(base + 0, c_blk),
                     shape=(1, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
        p1 = ct.load(stage_ptr, index=(base + 1, c_blk),
                     shape=(1, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
        p2 = ct.load(stage_ptr, index=(base + 2, c_blk),
                     shape=(1, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
        acc_prod = acc_prod + p0
        acc_src = acc_src + p1
        acc_patch = acc_patch + p2
    # patch: round through bf16
    acc_patch_bf = ct.astype(ct.astype(acc_patch, ct.bfloat16), ct.float32)
    ct.store(prod_out_ptr, index=(c_blk,), tile=ct.reshape(acc_prod, (BLOCK_C,)))
    ct.store(src_out_ptr, index=(c_blk,), tile=ct.reshape(acc_src, (BLOCK_C,)))
    ct.store(patch_sum_out_ptr, index=(c_blk,), tile=ct.reshape(acc_patch_bf, (BLOCK_C,)))


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
):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     _sh0, _sh1) = inputs
    device = arg0_1.device

    num_batch_groups = (BATCH + B_BLOCK - 1) // B_BLOCK
    num_token_groups = (TOKENS + T_GROUP - 1) // T_GROUP
    num_column_groups = num_batch_groups * num_token_groups
    num_stage_groups = (num_column_groups + GLOBAL_GROUP_BLOCK - 1) // GLOBAL_GROUP_BLOCK

    token_partials = torch.empty(
        (num_batch_groups * TOKENS, CHANNELS),
        device=device, dtype=torch.float32,
    )
    column_partials = torch.empty(
        (num_column_groups * 3, CHANNELS),
        device=device, dtype=torch.float32,
    )
    stage = torch.empty(
        (num_stage_groups * 3, CHANNELS),
        device=device, dtype=torch.float32,
    )
    token_sum_2d = torch.empty((TOKENS, CHANNELS), device=device, dtype=torch.float32)
    token0_sum_1d = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    bf16_patch = torch.empty((BATCH * PATCH_TOKENS, CHANNELS),
                             device=device, dtype=torch.bfloat16)
    prod_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    src_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    patch_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    # arg0 is [175360, 768] bf16 = viewable as [B, T, C].
    x_3d = arg0_1.view(BATCH, TOKENS, CHANNELS)
    # arg4/arg5 have stride (1376, 1, 1) → they're padded scale/center views.
    # Copy them out to dense contiguous (BATCH, TOKENS) storage to avoid
    # storage-size mismatch issues.
    row_center_2d = arg4_1.view(BATCH, TOKENS).contiguous()
    row_scale_2d = arg5_1.view(BATCH, TOKENS).contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_batch_groups, num_token_groups, 1),
        _row_group_kernel,
        (x_3d, arg1_1, arg2_1, arg3_1, row_center_2d, row_scale_2d, arg6_1,
         bf16_patch, token_partials, column_partials,
         BATCH, TOKENS, CHANNELS,
         num_token_groups, B_BLOCK, T_GROUP, BLOCK_C,
         CHANNEL_FACTOR, INV_CHANNEL_FACTOR),
    )

    ct.launch(
        stream, (TOKENS, (CHANNELS + TOKEN_BLOCK_C - 1) // TOKEN_BLOCK_C, 1),
        _finalize_token_kernel,
        (token_partials, token_sum_2d, token0_sum_1d,
         TOKENS, CHANNELS, num_batch_groups,
         _ceil_pow2(num_batch_groups), TOKEN_BLOCK_C),
    )

    ct.launch(
        stream, (num_stage_groups, (CHANNELS + GLOBAL_BLOCK_C - 1) // GLOBAL_BLOCK_C, 1),
        _global_stage1_kernel,
        (column_partials, stage,
         num_column_groups, CHANNELS,
         GLOBAL_GROUP_BLOCK, GLOBAL_BLOCK_C),
    )

    ct.launch(
        stream, ((CHANNELS + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _global_stage2_kernel,
        (stage, prod_out, src_out, patch_sum,
         num_stage_groups, CHANNELS, FINAL_BLOCK_C),
    )

    token_sum = token_sum_2d.view(1, TOKENS, CHANNELS)
    token0_sum = token0_sum_1d.view(1, 1, CHANNELS)
    patch_view = bf16_patch.view(BATCH, PATCH_TOKENS, CHANNELS).permute(0, 2, 1).reshape(BATCH, CHANNELS, 37, 37)

    return prod_out, src_out, token_sum, token0_sum, patch_view, patch_sum
