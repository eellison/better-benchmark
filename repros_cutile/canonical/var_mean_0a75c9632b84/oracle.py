"""cuTile port of var_mean_0a75c9632b84: DenseNet 6-branch cat + training BN + ReLU.

Uses cuTile for the per-channel var_mean reduction over (N, H, W). Torch handles
the cat producer, affine + ReLU epilogue, and running-stat mutations.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
VAR_CORRECTION = 1.005128205128205
MOMENTUM_NEW = 0.1
MOMENTUM_OLD = 0.9


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _channel_var_mean_kernel(
    x_ptr,          # bf16 [C, N_ELEMS] (viewed contiguously)
    mean_ptr,       # f32  [C]
    var_ptr,        # f32  [C]
    N_ELEMS: ct.Constant[int],
    N_ELEMS_F: ct.Constant[float],
    BLOCK_R: ct.Constant[int],
):
    channel = ct.bid(0)
    x = ct.load(
        x_ptr, index=(channel, 0), shape=(1, BLOCK_R),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    cols = ct.arange(BLOCK_R, dtype=ct.int32)
    valid = cols < N_ELEMS
    valid_2d = ct.reshape(valid, (1, BLOCK_R))
    zero_f = ct.full((1, BLOCK_R), 0.0, dtype=ct.float32)
    x_masked = ct.where(valid_2d, x_f, zero_f)
    total = ct.sum(x_masked)
    total_sq = ct.sum(x_masked * x_masked)
    mean = total * (1.0 / N_ELEMS_F)
    var = total_sq * (1.0 / N_ELEMS_F) - mean * mean
    ct.store(mean_ptr, index=(channel,), tile=ct.reshape(mean, (1,)))
    ct.store(var_ptr, index=(channel,), tile=ct.reshape(var, (1,)))


@oracle_impl(hardware="B200", point="c422a09b")
@oracle_impl(hardware="B200", point="c5a93fac")
@oracle_impl(hardware="B200", point="bed86ee6")
@oracle_impl(hardware="B200", point="da84b51a")
def oracle_forward(inputs):
    x0, x1, x2, x3, x4, x5, running_mean, running_var, weight, bias = inputs
    device = x0.device
    n = int(x0.shape[0])
    c0 = int(x0.shape[1])
    branch_c = int(x1.shape[1])
    h = int(x0.shape[2])
    w = int(x0.shape[3])
    channels = c0 + 5 * branch_c
    hw = h * w
    n_elems = n * hw
    block_r = _next_pow2(n_elems)

    # cat producer: (N, C, H, W) bf16 contiguous
    cat = torch.cat([x0, x1, x2, x3, x4, x5], dim=1).contiguous()

    # For per-channel reduction across (N, H, W): permute to (C, N, H, W) then
    # materialize into a contiguous (C, N*H*W) view.
    x_2d = cat.permute(1, 0, 2, 3).contiguous().reshape(channels, n_elems)

    mean_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    var_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (channels, 1, 1), _channel_var_mean_kernel,
        (x_2d, mean_1d, var_1d, n_elems, float(n_elems), block_r),
    )
    # Guard against tiny negative variance from bf16-derived rounding.
    var_1d = torch.clamp_min(var_1d, 0.0)

    mean_4d = mean_1d.view(1, channels, 1, 1)
    var_4d = var_1d.view(1, channels, 1, 1)
    rsqrt_4d = torch.rsqrt(var_4d + EPS)
    rsqrt_1d = rsqrt_4d.view(channels)

    # Affine + ReLU epilogue
    normalized = (cat.to(torch.float32) - mean_4d) * rsqrt_4d
    weight_ = weight.view(1, channels, 1, 1)
    bias_ = bias.view(1, channels, 1, 1)
    affine = normalized * weight_ + bias_
    affine_bf16 = affine.to(torch.bfloat16)
    relu = torch.relu(affine_bf16)

    # Running stat mutation
    new_running_mean = mean_1d * MOMENTUM_NEW + running_mean * MOMENTUM_OLD
    new_running_var = (var_1d * VAR_CORRECTION) * MOMENTUM_NEW + running_var * MOMENTUM_OLD
    torch.ops.aten.copy_(running_mean, new_running_mean)
    torch.ops.aten.copy_(running_var, new_running_var)

    return cat, rsqrt_1d, relu, mean_4d, running_mean, running_var
