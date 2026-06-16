"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete channels-last MobileNetV3 bf16 BN-inference affine, explicit bf16 round-trip, hard-swish, second bf16 round-trip, and 7x7 spatial mean directly into the returned keepdim/as_strided view, whereas Inductor lowers the decomposed broadcast affine, activation, cast, and mean through generic scheduled fragments; Inductor cannot do this today because its normalization/activation scheduler does not keep the per-channel affine constants and small spatial tile resident while emitting the required bf16 cast boundaries and final mean-only output; the fix is SCHEDULER_FUSION: teach BN-inference hard-swish spatial-mean scheduling to fuse the broadcast affine, required bf16 boundaries, activation epilogue, and keepdim mean store without changing sqrt/reciprocal/cast semantics."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 512
CHANNELS = 960
HW = 49
EPS = 1.0e-5


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bn_hardswish_spatial_mean_kernel(
    running_mean_ptr,
    x_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    TOTAL_ROWS: tl.constexpr,
    CHANNELS_C: tl.constexpr,
    HW_C: tl.constexpr,
    HW_F: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw_offsets = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw_offsets < HW_C

    batches = rows // CHANNELS_C
    channels = rows - batches * CHANNELS_C
    x_offsets = batches[:, None] * (CHANNELS_C * HW_C) + hw_offsets[None, :] * CHANNELS_C + channels[:, None]

    x = tl.load(
        x_ptr + x_offsets,
        mask=row_mask[:, None] & hw_mask[None, :],
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    running_mean = tl.load(running_mean_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    running_var = tl.load(running_var_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=row_mask, other=0.0).to(tl.float32)

    centered = _f32_sub(x, running_mean[:, None])
    sqrt_var = libdevice.sqrt(_f32_add(running_var, EPS_C))
    invstd = _f32_mul(_f32_div(1.0, sqrt_var), 1.0)
    normalized = _f32_mul(centered, invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    affine_f32 = _round_to_bf16_f32(affine)

    relu6 = _f32_add(affine_f32, 3.0)
    relu6 = tl.where(relu6 < 0.0, 0.0, relu6)
    relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
    hardswish = _f32_div(_f32_mul(affine_f32, relu6), 6.0)
    out_bf16_f32 = _round_to_bf16_f32(hardswish)

    mean_acc = tl.sum(tl.where(hw_mask[None, :], out_bf16_f32, 0.0), axis=1)
    mean_value = _f32_div(mean_acc, HW_F)
    tl.store(out_ptr + rows, mean_value.to(tl.bfloat16), mask=row_mask)


@oracle_impl(hardware="B200", point="3e244c1d", BLOCK_ROWS=8, BLOCK_HW=64, num_warps=2, num_stages=3)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int, num_stages: int):
    running_mean, x, running_var, weight, bias, shape, stride = inputs
    out = torch.empty_strided(
        (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])),
        (int(stride[0]), int(stride[1]), int(stride[2]), int(stride[3])),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _bn_hardswish_spatial_mean_kernel[(triton.cdiv(BATCH * CHANNELS, BLOCK_ROWS),)](
        running_mean,
        x,
        running_var,
        weight,
        bias,
        out,
        TOTAL_ROWS=BATCH * CHANNELS,
        CHANNELS_C=CHANNELS,
        HW_C=HW,
        HW_F=float(HW),
        EPS_C=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
