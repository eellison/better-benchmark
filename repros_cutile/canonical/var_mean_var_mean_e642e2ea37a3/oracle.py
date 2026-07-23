"""cuTile port of var_mean_var_mean_e642e2ea37a3: ShuffleNet dual BN training + channel-shuffle.

Uses cuTile for the per-channel var_mean reduction on each branch, then computes
the rest (BN affine + ReLU + cat + view/permute/clone/view/split + running-stat
mutation) in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
VAR_CORRECTION = 1.0001594642002871
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


def _bn_branch(x, running_mean, running_var, weight, bias):
    """Compute var_mean + affine + ReLU for one branch. Mutates running_mean/var."""
    device = x.device
    n = int(x.shape[0])
    channels = int(x.shape[1])
    h = int(x.shape[2])
    w = int(x.shape[3])
    n_elems = n * h * w
    block_r = _next_pow2(n_elems)
    # Permute to (C, N*H*W) contiguous so kernel can load [channel, :].
    x_2d = x.permute(1, 0, 2, 3).contiguous().reshape(channels, n_elems)
    mean_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    var_1d = torch.empty((channels,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (channels, 1, 1), _channel_var_mean_kernel,
        (x_2d, mean_1d, var_1d, n_elems, float(n_elems), block_r),
    )
    var_1d = torch.clamp_min(var_1d, 0.0)
    mean = mean_1d.view(1, channels, 1, 1)
    var = var_1d.view(1, channels, 1, 1)
    rsqrt = torch.rsqrt(var + EPS)
    normalized = (x.to(torch.float32) - mean) * rsqrt
    affine = normalized * weight.view(1, channels, 1, 1) + bias.view(1, channels, 1, 1)
    affine_bf16 = affine.to(torch.bfloat16)
    relu = torch.relu(affine_bf16)
    # Update running stats
    new_running_mean = mean_1d * MOMENTUM_NEW + running_mean * MOMENTUM_OLD
    new_running_var = (var_1d * VAR_CORRECTION) * MOMENTUM_NEW + running_var * MOMENTUM_OLD
    running_mean.copy_(new_running_mean)
    running_var.copy_(new_running_var)
    return mean, rsqrt, relu


@oracle_impl(hardware="B200", point="24b29630")
def oracle_forward(inputs):
    (
        x0, rm0, rv0, w0, b0,
        x1, rm1, rv1, w1, b1,
        shape0, shape1,
    ) = inputs

    mean0, rsqrt0, relu0 = _bn_branch(x0, rm0, rv0, w0, b0)
    mean1, rsqrt1, relu1 = _bn_branch(x1, rm1, rv1, w1, b1)

    # cat + shuffle + split
    cat = torch.cat([relu0, relu1], dim=1)          # (128, 464, 7, 7)
    view0 = cat.view(*shape0)                       # (128, 2, 232, 7, 7)
    permute = view0.permute(0, 2, 1, 3, 4)          # (128, 232, 2, 7, 7)
    clone = permute.contiguous()
    view1 = clone.view(*shape1)                     # (128, 464, 7, 7)
    split = torch.split(view1, 232, dim=1)
    getitem_4, getitem_5 = split[0], split[1]

    return (
        mean0, rsqrt0, mean1, rsqrt1,
        getitem_4, getitem_5,
        rm0, rv0, rm1, rv1,
    )
