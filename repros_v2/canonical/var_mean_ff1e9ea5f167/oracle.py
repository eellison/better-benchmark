"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16-input GroupNorm affine ReLU Repro.forward scope in one shape-specialized Triton row kernel, converting the input tile to fp32, reducing each exposed group to fp32 mean and inverse standard deviation, applying per-channel fp32 scale/bias and ReLU, and storing the final activation as bf16 while returning the fp32 stat tensors, whereas Inductor lowers the decomposed convert/view/var_mean/rsqrt/affine/ReLU/cast graph through generic normalization scheduling; Inductor cannot do this today because its norm template canonicalizer does not recognize singleton-spatial GroupNorm forward with exposed mean/rsqrt side outputs and a bf16 output-cast epilogue as one dedicated small-group template; the fix is NEW_PATTERN: add a guarded GroupNorm affine ReLU lowering that emits stats and the bf16 activation from the same fixed-group reduction tile."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _groupnorm_affine_relu_bf16_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_out_ptr,
    rsqrt_out_ptr,
    act_out_ptr,
    groups: tl.constexpr,
    channels: tl.constexpr,
    channels_per_group: tl.constexpr,
    hw: tl.constexpr,
    group_elems: tl.constexpr,
    total_groups: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    group_rows = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    elems = tl.arange(0, BLOCK_ELEMS)
    row_mask = group_rows < total_groups
    elem_mask = elems < group_elems

    batch = group_rows // groups
    group = group_rows - batch * groups
    offsets = batch[:, None] * channels * hw + group[:, None] * group_elems + elems[None, :]
    mask = row_mask[:, None] & elem_mask[None, :]
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    x_sum = tl.sum(tl.where(elem_mask[None, :], x, 0.0), axis=1)
    mean = x_sum / group_elems
    centered = tl.where(elem_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / group_elems
    inv_std = tl.rsqrt(var + 1.0e-5)

    channel = group[:, None] * channels_per_group + elems[None, :] // hw
    weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    act = tl.maximum(centered * inv_std[:, None] * weight + bias, 0.0)

    stats_offsets = batch * groups + group
    tl.store(mean_out_ptr + stats_offsets, mean, mask=row_mask)
    tl.store(rsqrt_out_ptr + stats_offsets, inv_std, mask=row_mask)
    tl.store(act_out_ptr + offsets, act, mask=mask)


@oracle_impl(hardware="B200", point="2744156c", BLOCK_GROUPS=8, BLOCK_ELEMS=16, num_warps=1)
@oracle_impl(hardware="B200", point="09b1da2c", BLOCK_GROUPS=4, BLOCK_ELEMS=32, num_warps=1)
@oracle_impl(hardware="B200", point="6756477c", BLOCK_GROUPS=2, BLOCK_ELEMS=64, num_warps=1)
@oracle_impl(hardware="B200", point="5ebcc8cf", BLOCK_GROUPS=1, BLOCK_ELEMS=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_GROUPS: int, BLOCK_ELEMS: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    n = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    groups = int(_shape_param_0[1])
    channels_per_group = int(_shape_param_0[2])
    hw = int(_shape_param_0[3])
    group_elems = channels_per_group * hw

    mean_out = torch.empty_strided(
        (n, groups, 1, 1),
        (groups, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt_out = torch.empty_strided(
        (n, groups, 1, 1),
        (groups, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    act_out = torch.empty_strided(
        (n, channels, height, width),
        (channels * height * width, height * width, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _groupnorm_affine_relu_bf16_kernel[(triton.cdiv(n * groups, BLOCK_GROUPS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        mean_out,
        rsqrt_out,
        act_out,
        groups=groups,
        channels=channels,
        channels_per_group=channels_per_group,
        hw=hw,
        group_elems=group_elems,
        total_groups=n * groups,
        BLOCK_GROUPS=BLOCK_GROUPS,
        BLOCK_ELEMS=BLOCK_ELEMS,
        num_warps=num_warps,
    )
    return mean_out, rsqrt_out, act_out
