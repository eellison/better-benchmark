"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 GroupNorm-style training fragment in one shape-specialized Triton group-tile kernel, including the bf16-to-fp32 input promotion, population `var_mean(..., correction=0)` over each 16-channel group, eps=1e-5 rsqrt, returned squeezed mean and rsqrt tensors, fp32 affine plus residual add, ReLU, degenerate 1x1 spatial mean, final bf16 `[128,512]` view, and bool `le(relu, 0)` mask, whereas Inductor lowers the decomposed graph through generic normalization, pointwise, singleton mean, cast, and mask schedules; Inductor cannot do this today because its norm pattern library does not canonicalize this multi-output small-group GroupNorm affine/residual/ReLU/mask idiom with exposed stats and a singleton spatial-mean consumer; the fix is NEW_PATTERN: add a canonical GroupNorm forward lowering that returns auxiliary stats and fuses same-tile consumers such as affine residual ReLU, singleton spatial mean/views, final bf16 casts, and ReLU-derived masks."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 512
GROUPS = 32
GROUP_SIZE = 16
TOTAL_GROUPS = BATCH * GROUPS


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _groupnorm_stats_final_mask_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    mean_out_ptr,
    rsqrt_out_ptr,
    final_out_ptr,
    mask_out_ptr,
    TOTAL_GROUPS_N: tl.constexpr,
    GROUPS_N: tl.constexpr,
    CHANNELS_N: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group_rows = tl.program_id(0) * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    lanes = tl.arange(0, BLOCK_C)
    valid_rows = group_rows < TOTAL_GROUPS_N

    batch = group_rows // GROUPS_N
    group = group_rows - batch * GROUPS_N
    channel = group[:, None] * GROUP_SIZE_N + lanes[None, :]
    elem_offsets = batch[:, None] * CHANNELS_N + channel
    elem_mask = valid_rows[:, None] & (lanes[None, :] < GROUP_SIZE_N)

    x = tl.load(x_ptr + elem_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    x_for_sum = tl.where(elem_mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=1) * (1.0 / GROUP_SIZE_N)
    centered = _f32_sub(x, mean[:, None])
    centered_for_var = tl.where(elem_mask, centered, 0.0)
    var = tl.sum(_f32_mul(centered_for_var, centered_for_var), axis=1) * (1.0 / GROUP_SIZE_N)
    invstd = libdevice.rsqrt(_f32_add(var, 1.0e-5))

    weight = tl.load(weight_ptr + channel, mask=elem_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channel, mask=elem_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + elem_offsets, mask=elem_mask, other=0.0).to(tl.float32)

    normalized = _f32_mul(centered, invstd[:, None])
    scaled = _f32_mul(normalized, weight)
    biased = _f32_add(scaled, bias)
    affine_residual = _f32_add(biased, residual)
    relu = tl.where((affine_residual > 0.0) | (affine_residual != affine_residual), affine_residual, 0.0)

    stats_offsets = batch * GROUPS_N + group
    tl.store(mean_out_ptr + stats_offsets, mean, mask=valid_rows)
    tl.store(rsqrt_out_ptr + stats_offsets, invstd, mask=valid_rows)
    tl.store(final_out_ptr + elem_offsets, relu.to(tl.bfloat16), mask=elem_mask)
    tl.store(mask_out_ptr + elem_offsets, relu <= 0.0, mask=elem_mask)


# 61898e90: bf16 input, batch=128, groups=32, group size=16, four-output GroupNorm stats/final/mask.
@oracle_impl(hardware="B200", point="61898e90", BLOCK_GROUPS=16, BLOCK_C=16, num_warps=1, num_stages=3)
def oracle_forward(inputs, *, BLOCK_GROUPS: int, BLOCK_C: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    mean_out = torch.empty_strided((BATCH, GROUPS), (GROUPS, 1), device=arg0_1.device, dtype=torch.float32)
    rsqrt_out = torch.empty_strided((BATCH, GROUPS), (GROUPS, 1), device=arg0_1.device, dtype=torch.float32)
    final_out = torch.empty_strided((BATCH, CHANNELS), (CHANNELS, 1), device=arg0_1.device, dtype=torch.bfloat16)
    mask_out = torch.empty_strided(
        (BATCH, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    _groupnorm_stats_final_mask_kernel[(triton.cdiv(TOTAL_GROUPS, BLOCK_GROUPS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        mean_out,
        rsqrt_out,
        final_out,
        mask_out,
        TOTAL_GROUPS_N=TOTAL_GROUPS,
        GROUPS_N=GROUPS,
        CHANNELS_N=CHANNELS,
        GROUP_SIZE_N=GROUP_SIZE,
        BLOCK_GROUPS=BLOCK_GROUPS,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return mean_out, rsqrt_out, final_out, mask_out
