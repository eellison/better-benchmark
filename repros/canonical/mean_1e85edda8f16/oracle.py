"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 BN-inference affine, explicit bf16 round, hard-swish, returned bf16 activation, and bf16 spatial mean in one Triton row kernel for each captured point, whereas Inductor lowers the decomposed broadcast affine, activation, cast, and mean through generic scheduled fragments; Inductor cannot do this today because its normalization/activation scheduler does not keep the per-channel affine constants and small spatial tile resident while emitting both the rounded activation and its mean side output; the fix is SCHEDULER_FUSION: teach BN-inference hard-swish spatial-mean scheduling to fuse the broadcast affine, required bf16 boundary, activation epilogue, full activation store, and keepdim mean store without changing the sqrt/reciprocal/cast semantics."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 0.001
BATCH = 32


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
def _bn_hardswish_mean_kernel(
    running_mean_ptr,
    x_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    HW_F: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    EPS_C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw_offsets = tl.arange(0, BLOCK_HW)
    channels = row_offsets - (row_offsets // C) * C
    row_mask = row_offsets < TOTAL_ROWS
    hw_mask = hw_offsets < HW

    x_offsets = row_offsets[:, None] * HW + hw_offsets[None, :]
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
    invstd = _f32_div(1.0, sqrt_var)
    normalized = _f32_mul(centered, invstd[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    affine_bf16 = affine.to(tl.bfloat16)
    affine_f32 = affine_bf16.to(tl.float32)

    relu6 = _f32_add(affine_f32, 3.0)
    relu6 = tl.where(relu6 < 0.0, 0.0, relu6)
    relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
    hardswish = _f32_div(_f32_mul(affine_f32, relu6), 6.0)
    out_bf16 = hardswish.to(tl.bfloat16)

    tl.store(out_ptr + x_offsets, out_bf16, mask=row_mask[:, None] & hw_mask[None, :])

    mean_acc = tl.sum(
        tl.where(hw_mask[None, :], out_bf16.to(tl.float32), 0.0),
        axis=1,
    )
    mean_value = _f32_div(mean_acc, HW_F)
    tl.store(mean_out_ptr + row_offsets, mean_value.to(tl.bfloat16), mask=row_mask)


@oracle_impl(hardware="B200", point="2c1989e8", C=960, HW=49, BLOCK_ROWS=8, BLOCK_HW=64, num_warps=2, num_stages=3)
@oracle_impl(hardware="B200", point="509d8143", C=672, HW=49, BLOCK_ROWS=8, BLOCK_HW=64, num_warps=2, num_stages=3)
@oracle_impl(hardware="B200", point="d9aaabff", C=672, HW=196, BLOCK_ROWS=4, BLOCK_HW=256, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="6cc76740", C=480, HW=196, BLOCK_ROWS=4, BLOCK_HW=256, num_warps=4, num_stages=3)
def oracle_forward(
    inputs,
    *,
    C: int,
    HW: int,
    BLOCK_ROWS: int,
    BLOCK_HW: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs
    height = int(arg1_1.shape[2])
    width = int(arg1_1.shape[3])
    out = torch.empty_strided(
        (BATCH, C, height, width),
        (C * HW, HW, width, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    mean = torch.empty_strided(
        (BATCH, C, 1, 1),
        (C, 1, 1, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _bn_hardswish_mean_kernel[(triton.cdiv(BATCH * C, BLOCK_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out,
        mean,
        C=C,
        HW=HW,
        HW_F=float(HW),
        TOTAL_ROWS=BATCH * C,
        EPS_C=EPS,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, mean
