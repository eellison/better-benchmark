"""cuTile port of var_mean_8b2ac790ab50: GhostNet BN training + cat + residual add.

Multi-kernel:
  1. Partial-stats kernel: split-K reduction of x → per-channel sum, sum^2.
  2. Finalize kernel: reduce partial stats to mean/invstd; running-stats update via torch.copy_ outside.
  3. Cat/residual kernel: compute bn(x), concat with arg5, add arg6 (residual).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_stats_kernel(
    x_ptr,           # bf16 [C, N*HW]  (from transpose+reshape)
    partial_sum_ptr, # f32 [num_chunks, C]
    partial_sq_ptr,  # f32 [num_chunks, C]
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
    x_f = ct.astype(x, ct.float32)
    r_mask_2d = ct.reshape(r_mask, (1, BLOCK_R))
    x_masked = ct.where(r_mask_2d, x_f, 0.0)
    sq = x_masked * x_masked

    part_sum = ct.sum(x_masked, axis=1)
    part_sq = ct.sum(sq, axis=1)

    ct.store(partial_sum_ptr, index=(r_block, c_block),
             tile=ct.reshape(part_sum, (1, BLOCK_C)))
    ct.store(partial_sq_ptr, index=(r_block, c_block),
             tile=ct.reshape(part_sq, (1, BLOCK_C)))


@ct.kernel
def _finalize_stats_kernel(
    partial_sum_ptr,   # f32 [num_chunks, C]
    partial_sq_ptr,    # f32 [num_chunks, C]
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    var_ptr,           # f32 [C]  (biased population variance, used for running_var)
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
    part_sum_masked = ct.where(chunk_mask_2d, part_sum, 0.0)
    part_sq_masked = ct.where(chunk_mask_2d, part_sq, 0.0)
    total_sum = ct.sum(part_sum_masked, axis=0)  # (BLOCK_C,)
    total_sq = ct.sum(part_sq_masked, axis=0)

    mean = total_sum / ELEMS_PER_CHANNEL
    variance = total_sq / ELEMS_PER_CHANNEL - mean * mean
    variance_pos = ct.where(variance < 0.0, 0.0, variance)
    invstd = ct.rsqrt(variance_pos + EPS_C)

    ct.store(mean_ptr, index=(c_block,), tile=mean)
    ct.store(invstd_ptr, index=(c_block,), tile=invstd)
    ct.store(var_ptr, index=(c_block,), tile=variance_pos)


@ct.kernel
def _cat_residual_kernel(
    x_ptr,          # bf16 [N, C, HW]
    mean_ptr,       # f32 [C]
    invstd_ptr,     # f32 [C]
    weight_ptr,     # f32 [C]
    bias_ptr,       # f32 [C]
    skip_ptr,       # bf16 [N, C, HW]
    residual_ptr,   # bf16 [N, 2C, HW]  (arg6 — the input to add_4)
    out_ptr,        # bf16 [N, 2C, HW]  (output = cat + residual, both bf16)
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    """Grid: (N, 2*C, HW // BLOCK_HW). c_out < C: skip branch. c_out >= C: BN branch."""
    n = ct.bid(0)
    c_out = ct.bid(1)
    hw_b = ct.bid(2)

    resid = ct.astype(
        ct.load(residual_ptr, index=(n, c_out, hw_b), shape=(1, 1, BLOCK_HW)),
        ct.float32,
    )
    if c_out < C_C:
        # skip branch: cat_value = skip[n, c_out, hw]
        val = ct.astype(
            ct.load(skip_ptr, index=(n, c_out, hw_b), shape=(1, 1, BLOCK_HW)),
            ct.float32,
        )
    else:
        c_bn = c_out - C_C
        x_val = ct.astype(
            ct.load(x_ptr, index=(n, c_bn, hw_b), shape=(1, 1, BLOCK_HW)),
            ct.float32,
        )
        mean = ct.astype(ct.load(mean_ptr, index=(c_bn,), shape=(1,)), ct.float32)
        invstd = ct.astype(ct.load(invstd_ptr, index=(c_bn,), shape=(1,)), ct.float32)
        weight = ct.astype(ct.load(weight_ptr, index=(c_bn,), shape=(1,)), ct.float32)
        bias = ct.astype(ct.load(bias_ptr, index=(c_bn,), shape=(1,)), ct.float32)
        mean_s = ct.reshape(mean, ())
        invstd_s = ct.reshape(invstd, ())
        weight_s = ct.reshape(weight, ())
        bias_s = ct.reshape(bias, ())
        norm = (x_val - mean_s) * invstd_s
        affine = norm * weight_s + bias_s
        val = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)

    out = ct.astype(val + resid, ct.bfloat16)
    ct.store(out_ptr, index=(n, c_out, hw_b), tile=out)


@oracle_impl(hardware="B200", point="26c28da3", BLOCK_R=1024, BLOCK_C=16, BLOCK_E=512)
@oracle_impl(hardware="B200", point="1f6ad890", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024)
@oracle_impl(hardware="B200", point="4585624e", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024)
@oracle_impl(hardware="B200", point="25930a87", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024)
@oracle_impl(hardware="B200", point="c94413ac", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024)
@oracle_impl(hardware="B200", point="23941270", BLOCK_R=4096, BLOCK_C=8, BLOCK_E=1024)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C, BLOCK_E):
    x, running_mean, running_var, weight, bias, skip, residual = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    out_channels = channels * 2
    elems_per_channel = batch * hw
    device = x.device

    # x is bf16 channels-last [batch, C, H, W]. Force NCHW-contiguous for the reduction.
    x_nchw = x.contiguous()  # [B, C, H, W] contiguous
    # Reshape as (C, batch*HW) via permute to expose channel-major layout for reduction.
    x_perm = x_nchw.permute(1, 0, 2, 3).contiguous().view(channels, elems_per_channel)

    num_chunks = (elems_per_channel + BLOCK_R - 1) // BLOCK_R
    partial_sum = torch.empty((num_chunks, channels), device=device, dtype=torch.float32)
    partial_sq = torch.empty((num_chunks, channels), device=device, dtype=torch.float32)

    if channels % BLOCK_C != 0:
        # find largest pow-2 dividing channels
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
    new_running_mean = running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM
    corrected_var = var_1d * RUNNING_VAR_CORRECTION
    new_running_var = running_var * (1.0 - MOMENTUM) + corrected_var * MOMENTUM
    running_mean.copy_(new_running_mean)
    running_var.copy_(new_running_var)

    # cat+residual kernel: produces bf16 [batch, 2*C, H, W] output.
    # Inputs: skip is bf16 channels-last [batch, C, H, W]. residual is bf16 channels-last [batch, 2*C, H, W].
    # For the cuTile kernel we use NCHW-contiguous views.
    skip_ncf = skip.contiguous().view(batch, channels, hw)
    residual_ncf = residual.contiguous().view(batch, out_channels, hw)
    x_ncf = x_nchw.view(batch, channels, hw)

    add_4 = torch.empty_strided(
        (batch, out_channels, height, width),
        (out_channels * hw, hw, width, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_4_ncf = add_4.view(batch, out_channels, hw)

    # BLOCK_HW must divide HW; find best pow-2.
    BLOCK_HW = 1
    cand = 1
    while cand <= hw and hw % cand == 0:
        BLOCK_HW = cand
        cand *= 2

    ct.launch(
        stream,
        (batch, out_channels, hw // BLOCK_HW),
        _cat_residual_kernel,
        (
            x_ncf, mean_1d, invstd_1d, weight, bias, skip_ncf, residual_ncf, add_4_ncf,
            channels, hw, BLOCK_HW,
        ),
    )

    # Return: (invstd (as [80]), add_4, mean [1,80,1,1], running_mean, running_var)
    mean_out = mean_1d.view(1, channels, 1, 1)
    return invstd_1d, add_4, mean_out, running_mean, running_var
