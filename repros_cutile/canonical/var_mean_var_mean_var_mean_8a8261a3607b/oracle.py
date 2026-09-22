"""cuTile port of var_mean_var_mean_var_mean_8a8261a3607b: Inception v3
4-branch BN training + affine + ReLU + channels-last channel cat.

For each of the 4 branches:
  1. `_partial_stats_kernel` computes per-chunk raw sum / sumsq (fp32) over
     the flattened [N*H*W, C] view of the channels-last input.
  2. `_finalize_stats_kernel` reduces partials to (mean, invstd) with
     eps=0.001, matching the graph's baked eps.
  3. Torch running-stat update via `copy_` — matches the eager Repro's
     mutation semantics. The Bessel correction constant `1.0000270336027683`
     is baked into the graph at capture time and used for both shape points.
  4. `_bn_relu_flat_kernel` does bf16 affine + ReLU as a 1D kernel over the
     flat (N*H*W*C) buffer, gathering per-channel stats by `col = idx % C`.

Final `cat`: 4 branch outputs concatenated along channel dim, materialized
as channels-last bf16 (N, TOTAL_C, H, W).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 0.001
MOMENTUM = 0.1
BAKED_VAR_CORRECTION = 1.0000270336027683


@ct.kernel
def _partial_stats_kernel(
    x_ptr,             # bf16 [C, K]  (gathered per-channel [C, N*H*W])
    partial_sum_ptr,   # f32 [num_chunks, C]
    partial_sumsq_ptr, # f32 [num_chunks, C]
    K_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
):
    c = ct.bid(0)
    chunk = ct.bid(1)
    x = ct.astype(
        ct.load(x_ptr, index=(c, chunk), shape=(1, BLOCK_K_),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    cols = ct.arange(BLOCK_K_, dtype=ct.int32)
    global_idx = ct.reshape(cols, (1, BLOCK_K_)) + chunk * BLOCK_K_
    valid = global_idx < K_
    zero_f = ct.zeros((1, BLOCK_K_), dtype=ct.float32)
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
    var_ptr,           # f32 [C]  (biased var for running-stat update)
    K_: ct.Constant[int],
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
    mean = total * (1.0 / K_)
    var = total_sq * (1.0 / K_) - mean * mean
    zero_scalar = ct.full((1, 1), 0.0, dtype=ct.float32)
    var_safe = ct.where(var < 0.0, zero_scalar, var)
    invstd = ct.rsqrt(var_safe + EPS)
    ct.store(mean_ptr, index=(c,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_ptr, index=(c,), tile=ct.reshape(invstd, (1,)))
    ct.store(var_ptr, index=(c,), tile=ct.reshape(var_safe, (1,)))


@ct.kernel
def _bn_relu_flat_kernel(
    x_ptr,           # bf16 flat [total]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    out_ptr,         # bf16 flat [total]
    C_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.astype(ct.load(x_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    lane = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    col = lane % C_
    mean = ct.gather(mean_ptr, col)
    invstd = ct.gather(invstd_ptr, col)
    weight = ct.gather(weight_ptr, col)
    bias = ct.gather(bias_ptr, col)
    centered = x - mean
    normalized = centered * invstd
    scaled = normalized * weight
    affine = scaled + bias
    rounded_bf16 = ct.astype(affine, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf16, ct.float32)
    is_nan = rounded_f != rounded_f
    keep = (rounded_f > 0.0) | is_nan
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    relu = ct.where(keep, rounded_f, zero_f)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(relu, ct.bfloat16))


def _bn_branch(x, running_mean, running_var, weight, bias, *, BLOCK_K: int, BLOCK: int):
    """Run one branch: returns (mean_4d, invstd_4d, out_flat_2d, C).

    `x` is channels-last (N, C, H, W).
    `out_flat_2d` is contiguous (N*H*W, C).
    """
    device = x.device
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    k = batch * hw
    total = k * channels

    # Gather per-channel [C, K] contiguous.
    # channels-last physical layout is NHWC contiguous; permute+reshape.
    x_gathered = (
        x.permute(1, 0, 2, 3).contiguous().reshape(channels, k)
    )

    num_chunks = (k + BLOCK_K - 1) // BLOCK_K
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
    var_biased = torch.empty((channels,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (channels, num_chunks, 1), _partial_stats_kernel,
        (x_gathered, partial_sum, partial_sumsq, k, BLOCK_K),
    )
    ct.launch(
        stream, (channels, 1, 1), _finalize_stats_kernel,
        (partial_sum, partial_sumsq,
         mean.view(channels), invstd.view(channels), var_biased,
         k, num_chunks, block_chunks),
    )

    # Running-stat updates (baked correction constant, fp32 arithmetic).
    corrected_var = var_biased * BAKED_VAR_CORRECTION
    new_running_mean = mean.view(channels) * MOMENTUM + running_mean * (1.0 - MOMENTUM)
    new_running_var = corrected_var * MOMENTUM + running_var * (1.0 - MOMENTUM)
    torch.ops.aten.copy_(running_mean, new_running_mean)
    torch.ops.aten.copy_(running_var, new_running_var)

    # BN affine + ReLU on flat (N*H*W*C) buffer laid out as NHWC contiguous.
    # x_flat is the NHWC contiguous 1D view of the channels-last input.
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    x_flat = x_nhwc.view(total)
    out_flat = torch.empty(total, device=device, dtype=torch.bfloat16)

    num_tiles = (total + BLOCK - 1) // BLOCK
    assert num_tiles * BLOCK == total, (
        f"total={total} must be divisible by BLOCK={BLOCK}"
    )

    ct.launch(
        stream, (num_tiles, 1, 1), _bn_relu_flat_kernel,
        (x_flat, mean.view(channels), invstd.view(channels), weight, bias,
         out_flat, channels, BLOCK),
    )

    # Reshape to (N*H*W, C) for cat.
    out_2d = out_flat.view(batch * hw, channels)
    return mean, invstd, out_2d, channels


def _run(inputs, *, BLOCK_K: int, BLOCK: int):
    (
        x0, running_mean0, running_var0, weight0, bias0,
        x1, running_mean1, running_var1, weight1, bias1,
        x2, running_mean2, running_var2, weight2, bias2,
        x3, running_mean3, running_var3, weight3, bias3,
    ) = inputs
    device = x0.device
    batch = int(x0.shape[0])
    height = int(x0.shape[2])
    width = int(x0.shape[3])

    mean0, invstd0, out0, c0 = _bn_branch(
        x0, running_mean0, running_var0, weight0, bias0, BLOCK_K=BLOCK_K, BLOCK=BLOCK)
    mean1, invstd1, out1, c1 = _bn_branch(
        x1, running_mean1, running_var1, weight1, bias1, BLOCK_K=BLOCK_K, BLOCK=BLOCK)
    mean2, invstd2, out2, c2 = _bn_branch(
        x2, running_mean2, running_var2, weight2, bias2, BLOCK_K=BLOCK_K, BLOCK=BLOCK)
    mean3, invstd3, out3, c3 = _bn_branch(
        x3, running_mean3, running_var3, weight3, bias3, BLOCK_K=BLOCK_K, BLOCK=BLOCK)

    total_c = c0 + c1 + c2 + c3
    # cat along the C axis of each (N*H*W, C) tensor.
    cat_2d = torch.cat([out0, out1, out2, out3], dim=1)  # (N*H*W, total_c)
    # Reshape to (N, H, W, total_c) and permute to (N, total_c, H, W) — this
    # yields the channels-last stride pattern (total_c*H*W, 1, total_c*W, total_c).
    cat_nhwc = cat_2d.view(batch, height, width, total_c)
    cat_nchw = cat_nhwc.permute(0, 3, 1, 2)

    return (
        mean0, invstd0,
        mean1, invstd1,
        mean2, invstd2,
        mean3, invstd3,
        cat_nchw,
        running_mean0, running_var0,
        running_mean1, running_var1,
        running_mean2, running_var2,
        running_mean3, running_var3,
    )


# c9aa364e: four bf16[128, 192, 17, 17] channels-last.  N*H*W = 36992
@oracle_impl(hardware="B200", point="c9aa364e", BLOCK_K=4096, BLOCK=1024)
# d0d42d59: bf16[128, 64/64/96/64, 35, 35] channels-last.  N*H*W = 156800
@oracle_impl(hardware="B200", point="d0d42d59", BLOCK_K=8192, BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK: int):
    return _run(inputs, BLOCK_K=BLOCK_K, BLOCK=BLOCK)
