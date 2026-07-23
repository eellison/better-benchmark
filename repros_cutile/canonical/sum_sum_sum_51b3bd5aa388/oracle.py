"""cuTile port of sum_sum_sum_51b3bd5aa388: DeiT LayerNorm-backward fanout.

Three cuTile kernels mirroring the Triton reference:
  1) _row_ln_bw_kernel: LN-backward per row, accumulates channel partials
     for sum_3 and sum_4.
  2) _token_reduce_kernel: per-token sum over batch, bf16-cast patch slice,
     patch-partial accumulation over batch.
  3) _finalize_vectors_kernel: reduces channel partials + patch partials.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
CHANNELS = 192
ROWS = BATCH * TOKENS  # 25216
PATCH_TOKENS = TOKENS - 1  # 196
INV_CHANNELS = 1.0 / CHANNELS


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


@ct.kernel
def _row_ln_bw_kernel(
    x_ptr,           # bf16 [ROWS, CHANNELS]
    weight_ptr,      # f32 [CHANNELS]
    source_ptr,      # f32 [ROWS, CHANNELS] (arg2 + arg3)  <-- expanded
    mean_ptr,        # f32 [ROWS]
    rsqrt_ptr,       # f32 [ROWS]
    residual_ptr,    # f32 [ROWS, CHANNELS]
    ln_out_ptr,      # f32 [ROWS, CHANNELS]     (arg6 + ln_grad)
    partial_x_norm_ptr,  # f32 [NUM_R, CHANNELS]
    partial_x_ptr,   # f32 [NUM_R, CHANNELS]
    ROWS_C: ct.Constant[int],
    CHANNELS_C: ct.Constant[int],
    INV_CHANNELS_: ct.Constant[float],
    ROW_SPLIT: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    pid = ct.bid(0)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_C))

    c_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask = ct.reshape(c_idx < CHANNELS_C, (1, BLOCK_C))

    acc_x_norm = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_x = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)

    tiles = ROW_SPLIT // BLOCK_R
    for t in range(tiles):
        r_tile = pid * tiles + t
        # Load full row (BLOCK_C == CHANNELS padded to power of 2).
        x_bf = ct.load(x_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
        x = ct.astype(x_bf, ct.float32)
        source = ct.load(source_ptr, index=(r_tile, 0),
                         shape=(BLOCK_R, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
        mean = ct.load(mean_ptr, index=(r_tile,), shape=(BLOCK_R,),
                       padding_mode=ct.PaddingMode.ZERO)
        rsqrt = ct.load(rsqrt_ptr, index=(r_tile,), shape=(BLOCK_R,),
                        padding_mode=ct.PaddingMode.ZERO)
        residual = ct.load(residual_ptr, index=(r_tile, 0),
                           shape=(BLOCK_R, BLOCK_C),
                           padding_mode=ct.PaddingMode.ZERO)

        mean_2d = ct.reshape(mean, (BLOCK_R, 1))
        rsqrt_2d = ct.reshape(rsqrt, (BLOCK_R, 1))
        centered = source - mean_2d
        norm = centered * rsqrt_2d
        weighted = x * weight_2d
        # Mask channel padding out before doing row reductions.
        weighted_masked = ct.where(c_mask, weighted, 0.0)
        row_sum = ct.sum(weighted_masked, axis=1, keepdims=True)
        row_dot = ct.sum(ct.where(c_mask, weighted * norm, 0.0),
                         axis=1, keepdims=True)
        scaled = weighted * float(CHANNELS_C)
        sub1 = scaled - row_sum
        sub2 = sub1 - norm * row_dot
        div = rsqrt_2d * INV_CHANNELS_
        ln_grad = div * sub2
        value = residual + ln_grad
        # Only store cols < CHANNELS (else OOB in the array anyway).
        ct.store(ln_out_ptr, index=(r_tile, 0), tile=value)

        # Accumulate per-channel partials (mask off OOB rows and padded chs).
        rows = r_tile * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
        row_active = ct.reshape(rows < ROWS_C, (BLOCK_R, 1))
        active2d = row_active & c_mask
        acc_x_norm = acc_x_norm + ct.sum(
            ct.where(active2d, x * norm, 0.0),
            axis=0, keepdims=True,
        )
        acc_x = acc_x + ct.sum(
            ct.where(active2d, x, 0.0),
            axis=0, keepdims=True,
        )

    ct.store(partial_x_norm_ptr, index=(pid, 0), tile=acc_x_norm)
    ct.store(partial_x_ptr, index=(pid, 0), tile=acc_x)


@ct.kernel
def _token_reduce_kernel(
    ln_out_ptr,          # f32 [BATCH, TOKENS, CHANNELS]  (viewed as [ROWS, CHANNELS])
    token_sum_ptr,       # f32 [1, TOKENS, CHANNELS]
    patch_flat_ptr,      # bf16 [BATCH*PATCH_TOKENS, CHANNELS]
    patch_partial_ptr,   # f32 [PATCH_TOKENS, CHANNELS]
    BATCH_: ct.Constant[int],
    TOKENS_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    BLOCK_B: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    token = ct.bid(0)
    c_blk = ct.bid(1)
    b_idx = ct.arange(BLOCK_B, dtype=ct.int32)
    b_mask = b_idx < BATCH_
    # rows = b * TOKENS_ + token; treat ln_out as (ROWS, CHANNELS) tile
    # loaded across BATCH stride-TOKENS_ rows -> use per-b load loop.

    acc_sum = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_patch = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)

    for b in range(BATCH_):
        # row index in the (ROWS, CHANNELS) view = b * TOKENS_ + token
        # We tile the (ROWS, CHANNELS) array with tile shape (1, BLOCK_C),
        # so tile-space index is (row, c_blk).
        row_idx = b * TOKENS_ + token
        row = ct.load(ln_out_ptr, index=(row_idx, c_blk),
                      shape=(1, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
        acc_sum = acc_sum + row
        # Only include for patch when token > 0.
        if token != 0:
            row_bf = ct.astype(row, ct.bfloat16)
            # Write bf16 patch slice at [b, token-1, c_blk] in
            # (BATCH*PATCH_TOKENS, CHANNELS) row-major tile view.
            patch_row_idx = b * (TOKENS_ - 1) + (token - 1)
            ct.store(patch_flat_ptr, index=(patch_row_idx, c_blk), tile=row_bf)
            acc_patch = acc_patch + ct.astype(row_bf, ct.float32)

    # token_sum store: pass as 2D (TOKENS, CHANNELS) tensor.
    ct.store(token_sum_ptr, index=(token, c_blk), tile=acc_sum)
    if token != 0:
        # store patch_partial for this token slot
        ct.store(patch_partial_ptr, index=(token - 1, c_blk), tile=acc_patch)


@ct.kernel
def _finalize_vectors_kernel(
    partial_x_norm_ptr,  # f32 [NUM_R, CHANNELS_PAD]
    partial_x_ptr,       # f32 [NUM_R, CHANNELS_PAD]
    patch_partial_ptr,   # f32 [PATCH_TOKENS, CHANNELS]
    out_x_norm_ptr,      # f32 [CHANNELS]
    out_x_ptr,           # f32 [CHANNELS]
    patch_sum_ptr,       # f32 [CHANNELS]
    NUM_R: ct.Constant[int],
    BLOCK_NUM_R: ct.Constant[int],
    PATCH_TOKENS_: ct.Constant[int],
    BLOCK_PATCH: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_blk = ct.bid(0)

    row_idx = ct.arange(BLOCK_NUM_R, dtype=ct.int32)
    row_mask = ct.reshape(row_idx < NUM_R, (BLOCK_NUM_R, 1))
    x_norm = ct.load(partial_x_norm_ptr, index=(0, c_blk),
                     shape=(BLOCK_NUM_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    x_norm = ct.where(row_mask, x_norm, 0.0)
    x_ = ct.load(partial_x_ptr, index=(0, c_blk),
                 shape=(BLOCK_NUM_R, BLOCK_C),
                 padding_mode=ct.PaddingMode.ZERO)
    x_ = ct.where(row_mask, x_, 0.0)
    ct.store(out_x_norm_ptr, index=(c_blk,), tile=ct.sum(x_norm, axis=0))
    ct.store(out_x_ptr, index=(c_blk,), tile=ct.sum(x_, axis=0))

    patch_idx = ct.arange(BLOCK_PATCH, dtype=ct.int32)
    patch_mask = ct.reshape(patch_idx < PATCH_TOKENS_, (BLOCK_PATCH, 1))
    patch_tile = ct.load(patch_partial_ptr, index=(0, c_blk),
                         shape=(BLOCK_PATCH, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
    patch_tile = ct.where(patch_mask, patch_tile, 0.0)
    patch_total = ct.sum(patch_tile, axis=0)
    patch_rounded = ct.astype(ct.astype(patch_total, ct.bfloat16), ct.float32)
    ct.store(patch_sum_ptr, index=(c_blk,), tile=patch_rounded)


@oracle_impl(
    hardware="B200",
    point="c8d56283",
    ROW_SPLIT=32,
    ROW_BLOCK=8,
    ROW_BLOCK_C=256,
    FINAL_BLOCK_C=64,
    TOKEN_BLOCK_C=32,
)
def oracle_forward(
    inputs,
    *,
    ROW_SPLIT: int,
    ROW_BLOCK: int,
    ROW_BLOCK_C: int,
    FINAL_BLOCK_C: int,
    TOKEN_BLOCK_C: int,
):
    flat, weight, source, pos, mean, rsqrt, residual, _s0, _s1 = inputs
    device = flat.device

    # Precompute the (source + pos) tensor once so the per-row kernel doesn't
    # need to figure out the token axis. pos is [1, TOKENS, CHANNELS] and
    # broadcasts over BATCH.
    combined = (source + pos).view(ROWS, CHANNELS)
    residual_2d = residual.view(ROWS, CHANNELS)
    mean_1d = mean.view(ROWS)
    rsqrt_1d = rsqrt.view(ROWS)

    row_blocks = (ROWS + ROW_SPLIT - 1) // ROW_SPLIT

    ln_out = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)
    partial_x_norm = torch.empty((row_blocks, ROW_BLOCK_C), device=device, dtype=torch.float32)
    partial_x = torch.empty((row_blocks, ROW_BLOCK_C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (row_blocks, 1, 1),
        _row_ln_bw_kernel,
        (flat, weight, combined, mean_1d, rsqrt_1d, residual_2d, ln_out,
         partial_x_norm, partial_x,
         ROWS, CHANNELS, INV_CHANNELS, ROW_SPLIT, ROW_BLOCK, ROW_BLOCK_C),
    )

    # Kernel 2: token reduce over batch + bf16 patch cast + patch partial.
    token_sum_2d = torch.empty((TOKENS, CHANNELS), device=device, dtype=torch.float32)
    patch_flat = torch.empty((BATCH * PATCH_TOKENS, CHANNELS), device=device, dtype=torch.bfloat16)
    patch_partial = torch.empty((PATCH_TOKENS, CHANNELS), device=device, dtype=torch.float32)

    ct.launch(
        stream, (TOKENS, CHANNELS // TOKEN_BLOCK_C, 1),
        _token_reduce_kernel,
        (ln_out, token_sum_2d, patch_flat, patch_partial,
         BATCH, TOKENS, CHANNELS, _next_pow2(BATCH), TOKEN_BLOCK_C),
    )
    token_sum = token_sum_2d.view(1, TOKENS, CHANNELS)

    out_x_norm = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    patch_sum = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    ct.launch(
        stream, ((CHANNELS + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _finalize_vectors_kernel,
        (partial_x_norm, partial_x, patch_partial,
         out_x_norm, out_x, patch_sum,
         row_blocks, _next_pow2(row_blocks),
         PATCH_TOKENS, _next_pow2(PATCH_TOKENS), FINAL_BLOCK_C),
    )

    cls_sum = torch.as_strided(
        token_sum, (1, 1, CHANNELS), (TOKENS * CHANNELS, CHANNELS, 1)
    )
    # patch_view = patch_flat.view(B, PATCH_T, C).permute(0, 2, 1).view(B, C, 14, 14)
    patch_view = patch_flat.view(BATCH, PATCH_TOKENS, CHANNELS).permute(0, 2, 1).reshape(BATCH, CHANNELS, 14, 14)

    return out_x_norm, out_x, token_sum, cls_sum, patch_view, patch_sum
