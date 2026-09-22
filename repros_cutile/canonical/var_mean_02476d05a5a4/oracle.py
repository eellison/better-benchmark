"""cuTile port of var_mean_02476d05a5a4: CycleGAN InstanceNorm+ReLU+reflected pad.

Three cuTile kernels matching Triton's structure:
  1. `_partial_channel_stats_kernel`: per-(channel, chunk) fp32 sum & sum_sq.
  2. `_final_channel_stats_kernel`: reduce chunk partials to mean/invstd.
  3. `_reflect_norm_bf16_relu_kernel`: reflect-padded normalize+ReLU output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
CHANNELS = 64
HEIGHT = 256
WIDTH = 256
PAD = 3
HW = HEIGHT * WIDTH
PADDED = HEIGHT + PAD + PAD
OUT_HW = PADDED * PADDED
TOTAL = CHANNELS * OUT_HW


def _next_pow2(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


@ct.kernel
def _partial_channel_stats_kernel(
    x_ptr,               # bf16 [CHANNELS, HW]
    partial_sum_ptr,     # f32 [NUM_CHUNKS, CHANNELS]
    partial_sumsq_ptr,   # f32 [NUM_CHUNKS, CHANNELS]
    HW_: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    c_block = ct.bid(0)
    chunk = ct.bid(1)
    # Load a (C_BLOCK, BLOCK_E) tile of bf16 data.
    vals_bf = ct.load(
        x_ptr, index=(c_block, chunk), shape=(C_BLOCK, BLOCK_E),
        padding_mode=ct.PaddingMode.ZERO,
    )
    vals = ct.astype(vals_bf, ct.float32)
    # Mask columns past HW (last chunk may be partial).
    cols_1d = chunk * BLOCK_E + ct.arange(BLOCK_E, dtype=ct.int32)
    hw_lim = ct.full(shape=(BLOCK_E,), fill_value=HW_, dtype=ct.int32)
    col_valid = cols_1d < hw_lim
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_E))
    zeros_2d = ct.zeros((C_BLOCK, BLOCK_E), dtype=ct.float32)
    vals_masked = ct.where(col_valid_2d, vals, zeros_2d)
    sums = ct.sum(vals_masked, axis=1, keepdims=False)      # (C_BLOCK,)
    sumsqs = ct.sum(vals_masked * vals_masked, axis=1)      # (C_BLOCK,)
    ct.store(partial_sum_ptr, index=(chunk, c_block), tile=ct.reshape(sums, (1, C_BLOCK)))
    ct.store(partial_sumsq_ptr, index=(chunk, c_block), tile=ct.reshape(sumsqs, (1, C_BLOCK)))


@ct.kernel
def _final_channel_stats_kernel(
    partial_sum_ptr,     # f32 [NUM_CHUNKS, CHANNELS]
    partial_sumsq_ptr,   # f32 [NUM_CHUNKS, CHANNELS]
    mean_ptr,            # f32 [CHANNELS]
    invstd_ptr,          # f32 [CHANNELS]
    HW_: ct.Constant[int],
    NUM_CHUNKS_: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    c_block = ct.bid(0)
    # Load [BLOCK_CHUNKS, C_BLOCK] tile of partials.
    sums = ct.load(
        partial_sum_ptr, index=(0, c_block), shape=(BLOCK_CHUNKS, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sumsqs = ct.load(
        partial_sumsq_ptr, index=(0, c_block), shape=(BLOCK_CHUNKS, C_BLOCK),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Mask rows past NUM_CHUNKS.
    rows_1d = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    lim = ct.full(shape=(BLOCK_CHUNKS,), fill_value=NUM_CHUNKS_, dtype=ct.int32)
    row_valid = rows_1d < lim
    row_valid_2d = ct.reshape(row_valid, (BLOCK_CHUNKS, 1))
    zeros = ct.zeros((BLOCK_CHUNKS, C_BLOCK), dtype=ct.float32)
    sums_masked = ct.where(row_valid_2d, sums, zeros)
    sumsqs_masked = ct.where(row_valid_2d, sumsqs, zeros)
    total = ct.sum(sums_masked, axis=0, keepdims=False)    # (C_BLOCK,)
    total_sq = ct.sum(sumsqs_masked, axis=0, keepdims=False)
    inv_hw = 1.0 / HW_
    mean = total * inv_hw
    var = total_sq * inv_hw - mean * mean
    zero_c = ct.zeros((C_BLOCK,), dtype=ct.float32)
    var = ct.where(var > zero_c, var, zero_c)
    invstd = ct.rsqrt(var + EPS)
    ct.store(mean_ptr, index=(c_block,), tile=mean)
    ct.store(invstd_ptr, index=(c_block,), tile=invstd)


@ct.kernel
def _reflect_norm_bf16_relu_kernel(
    x_ptr,          # bf16 flat [CHANNELS*HW]
    mean_ptr,       # f32 [CHANNELS]
    invstd_ptr,     # f32 [CHANNELS]
    out_ptr,        # bf16 flat [TOTAL]
    TOTAL_: ct.Constant[int],
    HW_: ct.Constant[int],
    OUT_HW_: ct.Constant[int],
    HEIGHT_: ct.Constant[int],
    WIDTH_: ct.Constant[int],
    PADDED_: ct.Constant[int],
    PAD_: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK_E + ct.arange(BLOCK_E, dtype=ct.int32)
    valid = offsets < TOTAL_
    safe_offsets = ct.where(valid, offsets, 0)
    channels = safe_offsets // OUT_HW_
    spatial = safe_offsets - channels * OUT_HW_
    oh = spatial // PADDED_
    ow = spatial - oh * PADDED_

    h_shifted = oh - PAD_
    h_abs = ct.where(h_shifted < 0, -h_shifted, h_shifted)
    h_folded = (HEIGHT_ - 1) - h_abs
    h_folded_abs = ct.where(h_folded < 0, -h_folded, h_folded)
    ih = (HEIGHT_ - 1) - h_folded_abs

    w_shifted = ow - PAD_
    w_abs = ct.where(w_shifted < 0, -w_shifted, w_shifted)
    w_folded = (WIDTH_ - 1) - w_abs
    w_folded_abs = ct.where(w_folded < 0, -w_folded, w_folded)
    iw = (WIDTH_ - 1) - w_folded_abs

    input_offsets = channels * HW_ + ih * WIDTH_ + iw
    x = ct.astype(ct.gather(x_ptr, input_offsets), ct.float32)
    mean = ct.gather(mean_ptr, channels)
    invstd = ct.gather(invstd_ptr, channels)
    y = (x - mean) * invstd
    rounded_bf = ct.astype(y, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf, ct.float32)
    is_nan = rounded_f != rounded_f
    positive = rounded_f > 0.0
    relu_val = ct.where(positive | is_nan, rounded_f, 0.0)
    ct.scatter(out_ptr, offsets, ct.astype(relu_val, ct.bfloat16), mask=valid)


@oracle_impl(hardware="B200", point="b9dd6ddf")
def oracle_forward(inputs):
    (x,) = inputs
    device = x.device
    channels = CHANNELS
    height = HEIGHT
    width = WIDTH
    hw = HW

    x_flat_2d = x.contiguous().view(channels, hw)

    BLOCK_E = 1024
    C_BLOCK = 1
    num_chunks = (hw + BLOCK_E - 1) // BLOCK_E
    block_chunks = _next_pow2(num_chunks)

    partial_sum = torch.empty((num_chunks, channels), device=device, dtype=torch.float32)
    partial_sumsq = torch.empty((num_chunks, channels), device=device, dtype=torch.float32)
    mean = torch.empty((channels,), device=device, dtype=torch.float32)
    invstd = torch.empty((channels,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((channels + C_BLOCK - 1) // C_BLOCK, num_chunks, 1),
        _partial_channel_stats_kernel,
        (x_flat_2d, partial_sum, partial_sumsq, hw, BLOCK_E, C_BLOCK),
    )
    ct.launch(
        stream, ((channels + C_BLOCK - 1) // C_BLOCK, 1, 1),
        _final_channel_stats_kernel,
        (partial_sum, partial_sumsq, mean, invstd, hw,
         num_chunks, block_chunks, C_BLOCK),
    )

    out = torch.empty_strided(
        (1, channels, PADDED, PADDED),
        (channels * OUT_HW, OUT_HW, PADDED, 1),
        device=device, dtype=torch.bfloat16,
    )
    x_flat = x.contiguous().view(-1)
    out_flat = out.view(-1)
    OUT_BLOCK = 1024
    ct.launch(
        stream, ((TOTAL + OUT_BLOCK - 1) // OUT_BLOCK, 1, 1),
        _reflect_norm_bf16_relu_kernel,
        (x_flat, mean, invstd, out_flat,
         TOTAL, hw, OUT_HW, height, width, PADDED, PAD, OUT_BLOCK),
    )
    return out
