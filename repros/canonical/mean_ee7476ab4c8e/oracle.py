"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DenseNet bf16 BN-inference affine, explicit bf16 rounding, NaN-preserving ReLU, 7x7 spatial mean, and final contiguous [64,1024] view in one Triton reduction, whereas Inductor lowers the decomposed broadcast affine/ReLU/mean graph through a generic reduction that repeats channel-only sqrt/reciprocal/affine work for every spatial element; Inductor cannot do this today because its reduction codegen does not canonicalize inference-BN parameters into per-channel scale/shift inside activation-fed spatial means while preserving the bf16 cast boundary before ReLU; the fix is ALGEBRAIC_ELIMINATION: hoist the channel-only BN algebra into the reduction kernel and emit the final mean layout directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _bn_relu_spatial_mean_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    hw = tl.arange(0, BLOCK_HW)
    row_mask = rows < 65536
    hw_mask = hw < 49
    valid = row_mask[:, None] & hw_mask[None, :]

    channels = 1024
    c = rows - (rows // channels) * channels
    offsets = rows[:, None] * 49 + hw[None, :]

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    channel_mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    variance = tl.load(var_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

    invstd = 1.0 / libdevice.sqrt(variance + 1.0e-5)
    y = ((x - channel_mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
    y_bf16 = y.to(tl.bfloat16, fp_downcast_rounding="rtne")
    zero = tl.full((BLOCK_ROWS, BLOCK_HW), 0.0, tl.float32).to(tl.bfloat16)
    relu = tl.where(y_bf16 < zero, zero, y_bf16)
    reduced = tl.sum(tl.where(hw_mask[None, :], relu.to(tl.float32), 0.0), axis=1) * (1.0 / 49.0)

    tl.store(out_ptr + rows, reduced.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=row_mask)


@oracle_impl(hardware="B200", point="2adc7e85", BLOCK_ROWS=16, BLOCK_HW=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int, num_warps: int):
    mean, x, var, weight, bias, _shape = inputs
    out = torch.empty_strided((64, 1024), (1024, 1), device=x.device, dtype=torch.bfloat16)
    _bn_relu_spatial_mean_kernel[(triton.cdiv(64 * 1024, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
