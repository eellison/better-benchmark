"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 BN-inference affine, Inductor-fused fp32 hard-swish producer, returned bf16 activation, and bf16 spatial mean in one Triton row-tile kernel for each captured point, whereas Inductor emits a separate channels-last pointwise materialization kernel followed by a generic persistent mean reduction over the materialized activation; Inductor cannot do this today because its scheduler does not keep the per-channel BN affine constants and small spatial tile resident across the activation store and mean side-output, and the fix is SCHEDULER_FUSION: teach the BN-inference hard-swish spatial-mean pattern to fuse the broadcast affine, activation epilogue, final bf16 activation store, and keepdim mean store while preserving Inductor's sqrt/reciprocal and bf16 output boundaries."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 512
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
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw = tl.arange(0, BLOCK_HW)
    row_mask = rows < TOTAL_ROWS
    hw_mask = hw < HW

    channels = rows - (rows // C) * C
    batches = rows // C
    base_offsets = batches * (C * HW) + channels
    offsets = base_offsets[:, None] + hw[None, :] * C

    x = tl.load(
        x_ptr + offsets,
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
    relu6 = _f32_add(affine, 3.0)
    relu6 = tl.where(relu6 < 0.0, 0.0, relu6)
    relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
    hardswish = _f32_div(_f32_mul(affine, relu6), 6.0)
    out_bf16 = hardswish.to(tl.bfloat16)

    tl.store(out_ptr + offsets, out_bf16, mask=row_mask[:, None] & hw_mask[None, :])

    mean_sum = tl.sum(tl.where(hw_mask[None, :], out_bf16.to(tl.float32), 0.0), axis=1)
    mean_value = _f32_div(mean_sum, HW_F)
    tl.store(mean_out_ptr + rows, mean_value.to(tl.bfloat16), mask=row_mask)


@oracle_impl(hardware="B200", point="3e244c1d", C=960, HW=49, BLOCK_ROWS=8, BLOCK_HW=64, num_warps=2, num_stages=3)
@oracle_impl(hardware="B200", point="57e42e70", C=672, HW=49, BLOCK_ROWS=8, BLOCK_HW=64, num_warps=2, num_stages=3)
@oracle_impl(hardware="B200", point="c37163dc", C=672, HW=196, BLOCK_ROWS=4, BLOCK_HW=256, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="86f01d63", C=480, HW=196, BLOCK_ROWS=4, BLOCK_HW=256, num_warps=4, num_stages=3)
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
        (C * HW, 1, width * C, C),
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
