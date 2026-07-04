"""cuTile port of var_mean_e405decd6535: bf16 GroupNorm-affine + residual + ReLU."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _groupnorm_kernel(
    x_ptr,        # bf16 [total_groups, GROUP_ELEMS]
    weight_ptr,   # bf16 [CHANNELS]
    bias_ptr,     # bf16 [CHANNELS]
    residual_ptr, # bf16 [total_groups, GROUP_ELEMS]
    out_ptr,      # bf16 [total_groups, GROUP_ELEMS]
    CHANNELS: ct.Constant[int],
    HW: ct.Constant[int],
    GROUP_ELEMS: ct.Constant[int],
    CHANNELS_PER_GROUP: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, GROUP_ELEMS))
    x_f = ct.astype(x, ct.float32)
    mean = ct.sum(x_f) * (1.0 / GROUP_ELEMS)
    centered = x_f - mean
    variance = ct.sum(centered * centered) * (1.0 / GROUP_ELEMS)
    invstd = ct.rsqrt(variance + 1.0e-5)
    normalized = centered * invstd

    # channel index = group_id * CHANNELS_PER_GROUP + (elems // HW)
    group_id = row % 32
    elems = ct.arange(GROUP_ELEMS, dtype=ct.int32)
    channel_1d = group_id * CHANNELS_PER_GROUP + elems // HW
    # channel_1d is (GROUP_ELEMS,); reshape to (1, GROUP_ELEMS)
    channel_2d = ct.reshape(channel_1d, (1, GROUP_ELEMS))

    # We need to gather weight[channel] and bias[channel]. cuTile has no
    # gather primitive, but since channels_per_group is small, we can iterate
    # or use static reindexing. However, this pattern is a per-element gather.
    # Since we don't have gather, load the full weight vector once and use
    # where-based selection — impractical for C=256.
    # Better: reshape input as [row, channels_per_group, HW] and load weight[c].
    # For B200 shapes: (C=256, HW=4), (C=128, HW=16), (C=64, HW=64).
    # channels_per_group = C/32 which is 8, 4, 2 resp.
    # Load weight per group: weight[group_id*channels_per_group : ...] gives shape (channels_per_group,).
    weight_group = ct.load(
        weight_ptr,
        index=(group_id,),
        shape=(CHANNELS_PER_GROUP,),
    )
    bias_group = ct.load(
        bias_ptr,
        index=(group_id,),
        shape=(CHANNELS_PER_GROUP,),
    )
    weight_group_f = ct.astype(weight_group, ct.float32)
    bias_group_f = ct.astype(bias_group, ct.float32)

    # normalized has shape (1, GROUP_ELEMS) = (1, channels_per_group * HW).
    # Reshape to (1, channels_per_group, HW) then broadcast weight_group_f (channels_per_group,) -> (1, channels_per_group, 1).
    normalized_3d = ct.reshape(normalized, (1, CHANNELS_PER_GROUP, HW))
    weight_3d = ct.reshape(weight_group_f, (1, CHANNELS_PER_GROUP, 1))
    bias_3d = ct.reshape(bias_group_f, (1, CHANNELS_PER_GROUP, 1))
    affine_3d = normalized_3d * weight_3d + bias_3d
    affine = ct.reshape(affine_3d, (1, GROUP_ELEMS))

    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, GROUP_ELEMS))
    rounded_affine = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    rounded_sum = ct.astype(
        ct.astype(rounded_affine + ct.astype(residual, ct.float32), ct.bfloat16),
        ct.float32,
    )
    zero = ct.zeros(shape=(1, GROUP_ELEMS), dtype=ct.float32)
    # ReLU preserving NaN
    is_pos_or_nan = (rounded_sum > 0.0) | (rounded_sum != rounded_sum)
    relu = ct.where(is_pos_or_nan, rounded_sum, zero)
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(relu, ct.bfloat16))


@oracle_impl(hardware="B200", point="49e390ef", BLOCK_M=4, BLOCK_K=32)
@oracle_impl(hardware="B200", point="d6ffe01a", BLOCK_M=4, BLOCK_K=64)
@oracle_impl(hardware="B200", point="f9823638", BLOCK_M=2, BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_K: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    channels_per_group = channels // 32
    group_elems = channels_per_group * hw
    total_groups = batch * 32

    out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        tuple(int(stride) for stride in arg3_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    x_2d = arg0_1.view(total_groups, group_elems)
    residual_2d = arg3_1.view(total_groups, group_elems)
    out_2d = out.view(total_groups, group_elems)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total_groups, 1, 1),
        _groupnorm_kernel,
        (x_2d, arg1_1, arg2_1, residual_2d, out_2d, channels, hw, group_elems, channels_per_group),
    )
    return out
