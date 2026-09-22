"""cuTile port of var_mean_fd789e584775: ShuffleNet BN training + skip + channel-shuffle.

Multi-kernel:
  1. Partial-stats kernel: split-K reduction of x → per-channel sum, sum^2 (f64).
  2. Finalize kernel: mean, invstd, running-stats update via torch.copy_.
  3. BN + concat + channel-shuffle output kernel.

Then torch does the shuffle → split to produce the returned aliases.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.00000996502277


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_stats_kernel(
    x_ptr,           # bf16 [C, N*HW]
    partial_sum_ptr, # f64 [num_chunks, C]
    partial_sq_ptr,  # f64 [num_chunks, C]
    C_C: ct.Constant[int],
    ELEMS_PER_CHANNEL: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """Grid: (C // BLOCK_C, num_chunks, 1)."""
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    r_offsets = r_block * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    r_mask = r_offsets < ELEMS_PER_CHANNEL

    x = ct.load(
        x_ptr, index=(c_block, r_block),
        shape=(BLOCK_C, BLOCK_R), padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float64)
    r_mask_2d = ct.reshape(r_mask, (1, BLOCK_R))
    x_masked = ct.where(r_mask_2d, x_f, ct.astype(ct.full((1, BLOCK_R), 0.0, dtype=ct.float32), ct.float64))
    sq = x_masked * x_masked

    part_sum = ct.sum(x_masked, axis=1)
    part_sq = ct.sum(sq, axis=1)

    ct.store(partial_sum_ptr, index=(r_block, c_block),
             tile=ct.reshape(part_sum, (1, BLOCK_C)))
    ct.store(partial_sq_ptr, index=(r_block, c_block),
             tile=ct.reshape(part_sq, (1, BLOCK_C)))


@ct.kernel
def _finalize_stats_kernel(
    partial_sum_ptr,   # f64 [num_chunks, C]
    partial_sq_ptr,    # f64 [num_chunks, C]
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    var_ptr,           # f32 [C]
    C_C: ct.Constant[int],
    ELEMS_PER_CHANNEL: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    NUM_CHUNKS_PAD: ct.Constant[int],
    EPS_C: ct.Constant[float],
    BLOCK_C: ct.Constant[int],
):
    """Grid: (C // BLOCK_C, 1, 1)."""
    c_block = ct.bid(0)
    chunks_pad = ct.arange(NUM_CHUNKS_PAD, dtype=ct.int32)
    chunk_mask = chunks_pad < NUM_CHUNKS

    part_sum = ct.load(
        partial_sum_ptr, index=(0, c_block),
        shape=(NUM_CHUNKS_PAD, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    part_sq = ct.load(
        partial_sq_ptr, index=(0, c_block),
        shape=(NUM_CHUNKS_PAD, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    chunk_mask_2d = ct.reshape(chunk_mask, (NUM_CHUNKS_PAD, 1))
    zero_f64 = ct.astype(ct.full((NUM_CHUNKS_PAD, BLOCK_C), 0.0, dtype=ct.float32), ct.float64)
    part_sum_masked = ct.where(chunk_mask_2d, part_sum, zero_f64)
    part_sq_masked = ct.where(chunk_mask_2d, part_sq, zero_f64)
    total_sum = ct.sum(part_sum_masked, axis=0)
    total_sq = ct.sum(part_sq_masked, axis=0)

    inv_count = 1.0 / ELEMS_PER_CHANNEL
    mean64 = total_sum * inv_count
    variance64 = total_sq * inv_count - mean64 * mean64
    variance_pos = ct.where(variance64 < 0.0,
                            ct.astype(ct.full((BLOCK_C,), 0.0, dtype=ct.float32), ct.float64),
                            variance64)
    variance = ct.astype(variance_pos, ct.float32)
    mean = ct.astype(mean64, ct.float32)
    invstd = ct.rsqrt(variance + EPS_C)

    ct.store(mean_ptr, index=(c_block,), tile=mean)
    ct.store(invstd_ptr, index=(c_block,), tile=invstd)
    ct.store(var_ptr, index=(c_block,), tile=variance)


@ct.kernel
def _bn_relu_kernel(
    x_ptr,        # bf16 [N, C, HW]
    mean_ptr,     # f32 [C]
    invstd_ptr,   # f32 [C]
    weight_ptr,   # f32 [C]
    bias_ptr,     # f32 [C]
    out_ptr,      # bf16 [N, C, HW]  (relu(BN(x)))
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, C, HW // BLOCK_HW)."""
    n = ct.bid(0)
    c = ct.bid(1)
    hw_b = ct.bid(2)

    x = ct.astype(
        ct.load(x_ptr, index=(n, c, hw_b), shape=(1, 1, BLOCK_HW)),
        ct.float64,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float64)
    invstd = ct.astype(ct.load(invstd_ptr, index=(c,), shape=(1,)), ct.float64)
    weight = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float64)
    bias = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float64)
    mean_s = ct.reshape(mean, ())
    invstd_s = ct.reshape(invstd, ())
    weight_s = ct.reshape(weight, ())
    bias_s = ct.reshape(bias, ())

    y = ((x - mean_s) * invstd_s) * weight_s + bias_s
    y_bf = ct.astype(ct.astype(y, ct.float32), ct.bfloat16)
    y_bf_f = ct.astype(y_bf, ct.float32)
    # ReLU (handling NaN as per triton: keep NaN via ct.where check)
    is_nan = y_bf_f != y_bf_f
    relu = ct.where((y_bf_f > 0.0) | is_nan, y_bf_f, 0.0)
    ct.store(out_ptr, index=(n, c, hw_b), tile=ct.astype(relu, ct.bfloat16))


@oracle_impl(hardware="B200", point="99b3b05e", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C, BLOCK_E):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape0, _shape1 = inputs
    del _shape0, _shape1

    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    out_channels = channels * 2
    elems_per_channel = batch * hw
    device = arg0_1.device

    # arg0 is bf16 contiguous NCHW. arg5 is bf16 with stride (90944, 784, 28, 1), which is
    # already NCHW-contiguous per-batch (stride[1]=784=H*W). It's actually [128, C, H, W] fine.

    x_nchw = arg0_1.contiguous()
    x_perm = x_nchw.permute(1, 0, 2, 3).contiguous().view(channels, elems_per_channel)

    num_chunks = (elems_per_channel + BLOCK_R - 1) // BLOCK_R
    partial_sum = torch.empty((num_chunks, channels), device=device, dtype=torch.float64)
    partial_sq = torch.empty((num_chunks, channels), device=device, dtype=torch.float64)

    if channels % BLOCK_C != 0:
        b = BLOCK_C
        while b > 1 and channels % b != 0:
            b //= 2
        BLOCK_C = b

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (channels // BLOCK_C, num_chunks, 1),
        _partial_stats_kernel,
        (x_perm, partial_sum, partial_sq, channels, elems_per_channel, BLOCK_R, BLOCK_C),
    )

    mean_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    invstd_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    var_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    NUM_CHUNKS_PAD = _next_pow2(num_chunks)
    ct.launch(
        stream,
        (channels // BLOCK_C, 1, 1),
        _finalize_stats_kernel,
        (
            partial_sum, partial_sq, mean_1d, invstd_1d, var_1d,
            channels, elems_per_channel, num_chunks, NUM_CHUNKS_PAD, EPS, BLOCK_C,
        ),
    )

    # Running stats update via torch copy_.
    new_running_mean = arg1_1 * (1.0 - MOMENTUM) + mean_1d * MOMENTUM
    corrected_var = var_1d * RUNNING_VAR_CORRECTION
    new_running_var = arg2_1 * (1.0 - MOMENTUM) + corrected_var * MOMENTUM
    arg1_1.copy_(new_running_mean)
    arg2_1.copy_(new_running_var)

    # BN + ReLU kernel produces bf16 [batch, channels, H, W].
    x_ncf = x_nchw.view(batch, channels, hw)
    relu_out = torch.empty((batch, channels, hw), device=device, dtype=torch.bfloat16)

    BLOCK_HW = 1
    cand = 1
    while cand <= hw and hw % cand == 0:
        BLOCK_HW = cand
        cand *= 2

    ct.launch(
        stream,
        (batch, channels, hw // BLOCK_HW),
        _bn_relu_kernel,
        (x_ncf, mean_1d, invstd_1d, arg3_1, arg4_1, relu_out, channels, hw, BLOCK_HW),
    )
    relu_bf16 = relu_out.view(batch, channels, height, width)

    # cat = [arg5_1, relu] along dim 1  (skip first, bn second)
    cat = torch.cat([arg5_1.contiguous(), relu_bf16], dim=1)  # [batch, 2*C, H, W]
    # view [batch, 2, C, H, W] then permute(0, 2, 1, 3, 4) then view [batch, 2*C, H, W]
    shuffled = cat.view(batch, 2, channels, height, width).permute(0, 2, 1, 3, 4).contiguous().view(batch, out_channels, height, width)
    getitem_2 = shuffled[:, :channels, :, :]
    getitem_3 = shuffled[:, channels:, :, :]

    # Return: getitem_1 (mean unsqueezed), rsqrt (invstd unsqueezed), getitem_2, getitem_3, copy_, copy__1
    mean_out = mean_1d.view(1, channels, 1, 1)
    invstd_out = invstd_1d.view(1, channels, 1, 1)
    return mean_out, invstd_out, getitem_2, getitem_3, arg1_1, arg2_1
