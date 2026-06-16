"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXt mixed bf16/fp32 residual channel LayerNorm training scope in one Triton row kernel, including the NHWC view over channels-last NCHW storage, bf16-to-fp32 residual add, population `var_mean(..., dim=3, correction=0, keepdim=True)`, eps=1e-6 rsqrt, returned normalized fp32 tensor, fp32 scale/bias affine with final bf16 rounding after the logical NCHW permute, and `rsqrt / 80` side output, whereas Inductor lowers the captured add/permute/var_mean/affine/permute/cast/div graph through generic normalization, pointwise, and layout scheduling fragments; Inductor cannot do this today because its normalization scheduler does not keep a visible mixed-dtype residual producer, normalized side output, affine cast, and inverse-std side output resident in one channel-layernorm schedule for this NHWC-over-NCHW pattern; the fix is SCHEDULER_FUSION: add a guarded ConvNeXt channel-layernorm schedule that fuses same-layout residual adds and emits normalized, rounded-affine, and inverse-std outputs directly for channels-last points."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _channel_residual_layernorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    norm_ptr,
    out_ptr,
    div_ptr,
    C: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
    channels = tl.arange(0, BLOCK_C)[None, :]
    valid = (rows < TOTAL_ROWS) & (channels < C)
    offsets = rows * C + channels

    x = tl.load(x_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
    values = x + residual

    values_masked = tl.where(valid, values, 0.0)
    mean = tl.sum(values_masked, axis=1)[:, None] / C
    centered = values - mean
    centered_masked = tl.where(valid, centered, 0.0)
    variance = tl.sum(centered_masked * centered_masked, axis=1)[:, None] / C
    invstd = libdevice.rsqrt(variance + 1.0e-6)
    normalized = centered * invstd

    channel_ids = tl.arange(0, BLOCK_C)
    channel_mask = channel_ids < C
    weight = tl.load(weight_ptr + channel_ids, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    bias = tl.load(bias_ptr + channel_ids, mask=channel_mask, other=0.0).to(tl.float32)[None, :]
    affine = normalized * weight + bias

    row_ids = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    tl.store(norm_ptr + offsets, normalized, mask=valid)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=valid)
    tl.store(div_ptr + row_ids[:, None], invstd / C, mask=(row_ids < TOTAL_ROWS)[:, None])


@oracle_impl(hardware="B200", point="8b7d5a32", BLOCK_C=128, BLOCK_ROWS=32, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_ROWS: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    total_rows = n * h * w

    normalized = torch.empty_strided(
        (n, h, w, c),
        (h * w * c, w * c, c, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (n, c, h, w),
        (c * h * w, 1, c * w, c),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    div = torch.empty_strided(
        (n, h, w, 1),
        (h * w, w, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _channel_residual_layernorm_kernel[(triton.cdiv(total_rows, BLOCK_ROWS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        normalized,
        out,
        div,
        C=c,
        TOTAL_ROWS=total_rows,
        BLOCK_C=BLOCK_C,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
        num_stages=3,
    )
    return normalized, out, div
