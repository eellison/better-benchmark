"""cuTile port of var_mean_4b4d0668d167: ResNet/ShuffleNet BN training +
affine + ReLU + 3x3 stride-2 maxpool-with-offsets.

Strategy:
  1. Two-pass BN stats: `_partial_stats_kernel` computes per-chunk raw
     sum / sumsq (fp32) over each channel; `_finalize_stats_kernel` reduces
     partials, produces mean/invstd (fp32).
  2. Torch epilogue updates running_mean / running_var via copy_ (matches
     eager Repro).
  3. `_bn_relu_kernel`: bf16 affine + ReLU with bf16 rounding, over flat
     NCHW tiles.
  4. Torch `prims._low_memory_max_pool_with_offsets` on the ReLU tensor for
     the maxpool step (kernel=3, stride=2, padding=1) — matches Triton's
     manual stencil numerically (fp32 -> bf16 cast, NaN-preserving max).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1


@ct.kernel
def _partial_stats_kernel(
    x_ptr,             # bf16 [C, elements_per_channel]  (gathered per-channel)
    partial_sum_ptr,   # f32 [num_chunks, C]
    partial_sumsq_ptr, # f32 [num_chunks, C]
    E_: ct.Constant[int],
    BLOCK_E_: ct.Constant[int],
):
    c = ct.bid(0)
    chunk = ct.bid(1)
    x = ct.astype(
        ct.load(x_ptr, index=(c, chunk), shape=(1, BLOCK_E_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    cols = ct.arange(BLOCK_E_, dtype=ct.int32)
    global_idx = ct.reshape(cols, (1, BLOCK_E_)) + chunk * BLOCK_E_
    valid = global_idx < E_
    zero_f = ct.zeros((1, BLOCK_E_), dtype=ct.float32)
    x_masked = ct.where(valid, x, zero_f)
    s = ct.sum(x_masked)
    ss = ct.sum(x_masked * x_masked)
    ct.store(partial_sum_ptr, index=(chunk, c), tile=ct.reshape(s, (1, 1)))
    ct.store(partial_sumsq_ptr, index=(chunk, c), tile=ct.reshape(ss, (1, 1)))


@ct.kernel
def _finalize_stats_kernel(
    partial_sum_ptr,   # f32 [num_chunks, C]
    partial_sumsq_ptr, # f32 [num_chunks, C]
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    E_: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
):
    c = ct.bid(0)
    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    valid_chunk = ct.reshape(chunks < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    zero_f = ct.zeros((BLOCK_CHUNKS, 1), dtype=ct.float32)
    s = ct.load(partial_sum_ptr, index=(0, c), shape=(BLOCK_CHUNKS, 1),
                padding_mode=ct.PaddingMode.ZERO)
    ss = ct.load(partial_sumsq_ptr, index=(0, c), shape=(BLOCK_CHUNKS, 1),
                 padding_mode=ct.PaddingMode.ZERO)
    s_masked = ct.where(valid_chunk, s, zero_f)
    ss_masked = ct.where(valid_chunk, ss, zero_f)
    total = ct.sum(s_masked)
    total_sq = ct.sum(ss_masked)
    mean = total * (1.0 / E_)
    var = total_sq * (1.0 / E_) - mean * mean
    zero_scalar = ct.full((1, 1), 0.0, dtype=ct.float32)
    var_safe = ct.where(var < 0.0, zero_scalar, var)
    invstd = ct.rsqrt(var_safe + EPS)
    ct.store(mean_ptr, index=(c,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(c,), tile=ct.reshape(invstd, (1,)))


@ct.kernel
def _bn_relu_kernel(
    x_ptr,             # bf16 [num_tiles, BLOCK]
    mean_bc_ptr,       # f32
    invstd_bc_ptr,     # f32
    weight_bc_ptr,     # f32
    bias_bc_ptr,       # f32
    out_ptr,           # bf16
    BLOCK_: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_)),
        ct.float32,
    )
    mean = ct.load(mean_bc_ptr, index=(row, 0), shape=(1, BLOCK_))
    invstd = ct.load(invstd_bc_ptr, index=(row, 0), shape=(1, BLOCK_))
    weight = ct.load(weight_bc_ptr, index=(row, 0), shape=(1, BLOCK_))
    bias = ct.load(bias_bc_ptr, index=(row, 0), shape=(1, BLOCK_))

    centered = x - mean
    normalized = centered * invstd
    scaled = normalized * weight
    affine = scaled + bias
    rounded_bf16 = ct.astype(affine, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf16, ct.float32)
    zero_f = ct.zeros((1, BLOCK_), dtype=ct.float32)
    is_nan = rounded_f != rounded_f
    keep = (rounded_f > 0.0) | is_nan
    relu = ct.where(keep, rounded_f, zero_f)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(relu, ct.bfloat16))


def _broadcast_channel_1d(v, batch, channels, height, width, total):
    return (
        v.view(1, channels, 1, 1)
         .expand(batch, channels, height, width)
         .contiguous()
         .view(total)
    )


def _run(inputs, *, BLOCK_E: int, BLOCK: int):
    x, running_mean, running_var, weight, bias, _kernel_size, _stride = inputs
    device = x.device
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    elements_per_channel = batch * hw
    total = batch * channels * hw
    out_h = height // 2
    out_w = width // 2

    # 1. Gather per-channel [C, N*H*W]
    x_gathered = (
        x.permute(1, 0, 2, 3).contiguous().reshape(channels, elements_per_channel)
    )

    num_chunks = (elements_per_channel + BLOCK_E - 1) // BLOCK_E
    block_chunks = 1
    while block_chunks < num_chunks:
        block_chunks *= 2

    partial_sum = torch.empty((num_chunks, channels), device=device, dtype=torch.float32)
    partial_sumsq = torch.empty((num_chunks, channels), device=device, dtype=torch.float32)

    mean = torch.empty_strided(
        (1, channels, 1, 1), (channels, 1, 1, 1),
        device=device, dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (1, channels, 1, 1), (channels, 1, 1, 1),
        device=device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (channels, num_chunks, 1), _partial_stats_kernel,
        (x_gathered, partial_sum, partial_sumsq, elements_per_channel, BLOCK_E),
    )
    ct.launch(
        stream, (channels, 1, 1), _finalize_stats_kernel,
        (partial_sum, partial_sumsq, mean.view(channels), invstd.view(channels),
         elements_per_channel, num_chunks, block_chunks),
    )

    # 2. Running-stat update (fp32 arithmetic, matches Triton).
    mean_1d = mean.view(channels)
    invstd_1d = invstd.view(channels)
    variance_1d = 1.0 / (invstd_1d * invstd_1d) - EPS
    # Bessel correction: E / (E - 1)
    correction = elements_per_channel / (elements_per_channel - 1)
    corrected_var = variance_1d * correction
    new_running_mean = mean_1d * MOMENTUM + running_mean * (1.0 - MOMENTUM)
    new_running_var = corrected_var * MOMENTUM + running_var * (1.0 - MOMENTUM)
    torch.ops.aten.copy_(running_mean, new_running_mean)
    torch.ops.aten.copy_(running_var, new_running_var)

    # 3. BN affine + ReLU on flat [N*C*H*W]
    x_flat = x.contiguous().view(total)
    mean_bc = _broadcast_channel_1d(mean_1d, batch, channels, height, width, total)
    invstd_bc = _broadcast_channel_1d(invstd_1d, batch, channels, height, width, total)
    weight_bc = _broadcast_channel_1d(weight, batch, channels, height, width, total)
    bias_bc = _broadcast_channel_1d(bias, batch, channels, height, width, total)

    num_tiles = (total + BLOCK - 1) // BLOCK
    assert num_tiles * BLOCK == total, (
        f"total={total} must be divisible by BLOCK={BLOCK}"
    )

    x_2d = x_flat.view(num_tiles, BLOCK)
    mean_2d = mean_bc.view(num_tiles, BLOCK)
    invstd_2d = invstd_bc.view(num_tiles, BLOCK)
    weight_2d = weight_bc.view(num_tiles, BLOCK)
    bias_2d = bias_bc.view(num_tiles, BLOCK)
    relu_out = torch.empty(num_tiles, BLOCK, device=device, dtype=torch.bfloat16)

    ct.launch(
        stream, (num_tiles, 1, 1), _bn_relu_kernel,
        (x_2d, mean_2d, invstd_2d, weight_2d, bias_2d, relu_out, BLOCK),
    )
    relu_bnchw = relu_out.view(total).view(batch, channels, height, width)

    # 4. Torch max_pool_with_offsets (kernel=3, stride=2, padding=1).
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu_bnchw, [3, 3], [2, 2], [1, 1], [1, 1], False,
    )

    return mean, invstd, values, offsets, running_mean, running_var


# 5e4f83d3: (16, 64, 112, 112)  total=12,845,056
@oracle_impl(hardware="B200", point="5e4f83d3", BLOCK_E=4096, BLOCK=1024)
# 22d6d5c7: (32, 64, 112, 112)  total=25,690,112
@oracle_impl(hardware="B200", point="22d6d5c7", BLOCK_E=4096, BLOCK=1024)
# 4ee39e3f: (8, 64, 112, 112)   total=6,422,528
@oracle_impl(hardware="B200", point="4ee39e3f", BLOCK_E=4096, BLOCK=1024)
# 75543dfe: (128, 24, 112, 112) total=38,535,168
@oracle_impl(hardware="B200", point="75543dfe", BLOCK_E=4096, BLOCK=1024)
# eaa8fe86: (128, 64, 112, 112) total=102,760,448
@oracle_impl(hardware="B200", point="eaa8fe86", BLOCK_E=4096, BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_E: int, BLOCK: int):
    return _run(inputs, BLOCK_E=BLOCK_E, BLOCK=BLOCK)
