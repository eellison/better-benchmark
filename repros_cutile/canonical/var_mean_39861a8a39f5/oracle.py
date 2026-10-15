"""cuTile port of var_mean_39861a8a39f5: GroupNorm-style residual+ReLU+mask kernel.

Per group-row: fp32 var/mean, rsqrt+eps=1e-5, affine (per-channel), residual add,
ReLU, bf16 store, bool le<=0 mask. Groups = 32. group_elems in {16, 64, 128}.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


GROUPS = 32
EPS = 1.0e-5


@ct.kernel
def _groupnorm_kernel(
    x_ptr,        # bf16 [TOTAL_GROUPS, GROUP_ELEMS]
    weight_ptr,   # bf16 [CHANNELS]
    bias_ptr,     # bf16 [CHANNELS]
    residual_ptr, # bf16 [TOTAL_GROUPS, GROUP_ELEMS]
    mean_ptr,     # f32 [TOTAL_GROUPS]
    rsqrt_ptr,    # f32 [TOTAL_GROUPS]
    bf16_ptr,     # bf16 [TOTAL_GROUPS, GROUP_ELEMS]
    mask_ptr,     # bool [TOTAL_GROUPS, GROUP_ELEMS]
    CHANNELS: ct.Constant[int],
    HW: ct.Constant[int],
    GROUP_ELEMS: ct.Constant[int],
    GROUPS_C: ct.Constant[int],
    CHANNELS_PER_GROUP: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.astype(ct.load(x_ptr, index=(row, 0), shape=(1, GROUP_ELEMS)), ct.float32)
    mean = ct.sum(x) * (1.0 / GROUP_ELEMS)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / GROUP_ELEMS)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    # Per-element channel index: group = row % GROUPS_C, channel = group*CPG + elem//HW
    elems = ct.arange(GROUP_ELEMS, dtype=ct.int64)
    row_scalar = ct.full(shape=(1,), fill_value=row, dtype=ct.int64)
    group = row_scalar - (row_scalar // GROUPS_C) * GROUPS_C  # row % GROUPS_C
    channel = group * CHANNELS_PER_GROUP + elems // HW  # shape (GROUP_ELEMS,) via broadcasting
    channel_2d = ct.reshape(channel, (1, GROUP_ELEMS))

    weight = ct.astype(ct.gather(weight_ptr, channel_2d), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, channel_2d), ct.float32)
    residual = ct.astype(ct.load(residual_ptr, index=(row, 0), shape=(1, GROUP_ELEMS)), ct.float32)

    affine = normalized * weight + bias
    summed = affine + residual
    relu = ct.where(summed > 0.0, summed, ct.astype(ct.full(shape=(1, GROUP_ELEMS), fill_value=0.0, dtype=ct.float32), ct.float32))

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.store(bf16_ptr, index=(row, 0), tile=ct.astype(relu, ct.bfloat16))
    ct.store(mask_ptr, index=(row, 0), tile=relu <= 0.0)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _launch(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    group_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    groups = int(group_shape[1])
    group_elems = int(group_shape[2]) * int(group_shape[3])
    total_groups = batch * groups
    cpg = channels // groups

    mean = torch.empty_strided(
        (batch, groups), (groups, 1), device=arg0_1.device, dtype=torch.float32
    )
    rsqrt = torch.empty_strided(
        (batch, groups), (groups, 1), device=arg0_1.device, dtype=torch.float32
    )
    bf16 = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape), device=arg0_1.device, dtype=torch.bfloat16
    )
    le = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape), device=arg0_1.device, dtype=torch.bool
    )

    x_view = arg0_1.view(total_groups, group_elems)
    residual_view = arg3_1.view(total_groups, group_elems)
    bf16_view = bf16.view(total_groups, group_elems)
    le_view = le.view(total_groups, group_elems)
    mean_flat = mean.view(total_groups)
    rsqrt_flat = rsqrt.view(total_groups)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_groups, 1, 1),
        _groupnorm_kernel,
        (x_view, arg1_1, arg2_1, residual_view, mean_flat, rsqrt_flat,
         bf16_view, le_view, channels, hw, group_elems, groups, cpg),
    )
    return mean, rsqrt, bf16, le


@oracle_impl(hardware="B200", point="af9e4718")
@oracle_impl(hardware="B200", point="b79ebf0a")
@oracle_impl(hardware="B200", point="a488e261")
def oracle_forward(inputs):
    return _launch(inputs)
