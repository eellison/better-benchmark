"""cuTile port of var_mean_129ad79a9b02: CycleGAN InstanceNorm + ReLU.

Per-channel population var_mean over HxW for a (1, C, H, W) bf16 input, then
normalize (eps=1e-5), round to bf16, ReLU (preserving NaN), and store bf16.

For each channel: if HW fits in one tile use a single kernel; otherwise use a
partial-stats + finalize + apply three-pass pattern.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _instance_norm_relu_kernel_single(
    x_ptr,       # bf16 (C, HW)
    out_ptr,     # bf16 (C, HW)
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c = ct.bid(0)
    x = ct.astype(ct.load(x_ptr, index=(c, 0), shape=(1, BLOCK_HW),
                          padding_mode=ct.PaddingMode.ZERO), ct.float32)
    cols = ct.arange(BLOCK_HW, dtype=ct.int32)
    col_mask = ct.reshape(cols < HW, (1, BLOCK_HW))
    zero_f = ct.zeros((1, BLOCK_HW), dtype=ct.float32)
    x_masked = ct.where(col_mask, x, zero_f)
    total = ct.sum(x_masked)
    mean = total * (1.0 / HW)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, zero_f)
    var = ct.sum(centered_masked * centered_masked) * (1.0 / HW)
    var_safe = ct.where(var > 0.0, var, 0.0)
    invstd = ct.rsqrt(var_safe + EPS)
    y = centered * invstd
    y_bf16 = ct.astype(y, ct.bfloat16)
    y_f = ct.astype(y_bf16, ct.float32)
    is_nan = y_f != y_f
    keep = (y_f > 0.0) | is_nan
    relu = ct.where(keep, y_f, zero_f)
    ct.store(out_ptr, index=(c, 0), tile=ct.astype(relu, ct.bfloat16))


@ct.kernel
def _in_partial_stats_kernel(
    x_ptr,           # bf16 (C, HW)
    partial_sum,     # f32 (num_chunks, C)
    partial_sumsq,   # f32 (num_chunks, C)
    HW: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    c = ct.bid(0)
    chunk = ct.bid(1)
    x = ct.astype(ct.load(x_ptr, index=(c, chunk), shape=(1, BLOCK_E),
                          padding_mode=ct.PaddingMode.ZERO), ct.float32)
    cols = ct.arange(BLOCK_E, dtype=ct.int32)
    global_idx = ct.reshape(cols, (1, BLOCK_E)) + chunk * BLOCK_E
    valid = global_idx < HW
    zero_f = ct.zeros((1, BLOCK_E), dtype=ct.float32)
    x_masked = ct.where(valid, x, zero_f)
    s = ct.sum(x_masked)
    ss = ct.sum(x_masked * x_masked)
    ct.store(partial_sum, index=(chunk, c), tile=ct.reshape(s, (1, 1)))
    ct.store(partial_sumsq, index=(chunk, c), tile=ct.reshape(ss, (1, 1)))


@ct.kernel
def _in_finalize_kernel(
    partial_sum,     # f32 (num_chunks, C)
    partial_sumsq,   # f32 (num_chunks, C)
    mean_ptr,        # f32 (C,)
    invstd_ptr,      # f32 (C,)
    HW: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
):
    c = ct.bid(0)
    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    valid_chunk = ct.reshape(chunks < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    zero_f = ct.zeros((BLOCK_CHUNKS, 1), dtype=ct.float32)
    s = ct.load(partial_sum, index=(0, c), shape=(BLOCK_CHUNKS, 1),
                padding_mode=ct.PaddingMode.ZERO)
    ss = ct.load(partial_sumsq, index=(0, c), shape=(BLOCK_CHUNKS, 1),
                 padding_mode=ct.PaddingMode.ZERO)
    s_masked = ct.where(valid_chunk, s, zero_f)
    ss_masked = ct.where(valid_chunk, ss, zero_f)
    total = ct.sum(s_masked)
    total_sq = ct.sum(ss_masked)
    mean = total * (1.0 / HW)
    var = total_sq * (1.0 / HW) - mean * mean
    var_safe = ct.where(var > 0.0, var, 0.0)
    invstd = ct.rsqrt(var_safe + EPS)
    ct.store(mean_ptr, index=(c,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(c,), tile=ct.reshape(invstd, (1,)))


@ct.kernel
def _in_apply_kernel(
    x_ptr,           # bf16 (C, HW)
    mean_ptr,        # f32 (C,)
    invstd_ptr,      # f32 (C,)
    out_ptr,         # bf16 (C, HW)
    HW: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    c = ct.bid(0)
    chunk = ct.bid(1)
    x = ct.astype(ct.load(x_ptr, index=(c, chunk), shape=(1, BLOCK_E),
                          padding_mode=ct.PaddingMode.ZERO), ct.float32)
    mean_tile = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    invstd_tile = ct.astype(ct.load(invstd_ptr, index=(c,), shape=(1,)), ct.float32)
    mean_v = ct.reshape(mean_tile, (1, 1))
    invstd_v = ct.reshape(invstd_tile, (1, 1))
    y = (x - mean_v) * invstd_v
    y_bf16 = ct.astype(y, ct.bfloat16)
    y_f = ct.astype(y_bf16, ct.float32)
    zero_f = ct.zeros((1, BLOCK_E), dtype=ct.float32)
    is_nan = y_f != y_f
    keep = (y_f > 0.0) | is_nan
    relu = ct.where(keep, y_f, zero_f)
    ct.store(out_ptr, index=(c, chunk), tile=ct.astype(relu, ct.bfloat16))


@oracle_impl(hardware="B200", point="a9dd97a3", BLOCK_HW=16384)
@oracle_impl(hardware="B200", point="b9dd6ddf", BLOCK_HW=1024)
def oracle_forward(inputs, *, BLOCK_HW: int):
    (x,) = inputs
    _batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    device = x.device

    out = torch.empty_strided(
        (1, channels, height, width),
        (channels * hw, hw, width, 1),
        device=device, dtype=torch.bfloat16,
    )

    x_2d = x.view(channels, hw)
    out_2d = out.view(channels, hw)

    stream = torch.cuda.current_stream()
    if hw <= BLOCK_HW and (BLOCK_HW & (BLOCK_HW - 1)) == 0:
        ct.launch(
            stream, (channels, 1, 1), _instance_norm_relu_kernel_single,
            (x_2d, out_2d, hw, BLOCK_HW),
        )
    else:
        block_e = 1024
        num_chunks = (hw + block_e - 1) // block_e
        block_chunks = 1
        while block_chunks < num_chunks:
            block_chunks *= 2
        partial_sum = torch.empty((num_chunks, channels), device=device, dtype=torch.float32)
        partial_sumsq = torch.empty((num_chunks, channels), device=device, dtype=torch.float32)
        mean = torch.empty((channels,), device=device, dtype=torch.float32)
        invstd = torch.empty((channels,), device=device, dtype=torch.float32)
        ct.launch(
            stream, (channels, num_chunks, 1), _in_partial_stats_kernel,
            (x_2d, partial_sum, partial_sumsq, hw, block_e),
        )
        ct.launch(
            stream, (channels, 1, 1), _in_finalize_kernel,
            (partial_sum, partial_sumsq, mean, invstd, hw, num_chunks, block_chunks),
        )
        ct.launch(
            stream, (channels, num_chunks, 1), _in_apply_kernel,
            (x_2d, mean, invstd, out_2d, hw, block_e),
        )
    return out
