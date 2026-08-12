"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 exact-erf GELU, required bf16 activation rounding, population var_mean side outputs, epsilon rsqrt, affine epilogue, final bf16 cast, and metadata-only views in one Triton row-reduction kernel, whereas Inductor lowers the decomposed view/cast/GELU/cast/var_mean/rsqrt/affine/view graph through generic normalization reduction scheduling; Inductor cannot do this today because the norm scheduler lacks a guarded exact-GELU-producer LayerNorm template that preserves the intermediate bf16 rounding boundary and live mean/rsqrt side outputs while keeping the row tile resident through the affine store; the fix is SCHEDULER_FUSION: add a bf16 exact-GELU-producing LayerNorm template that fuses the pointwise producer, population row moments, eps rsqrt, side-output stores, affine epilogue, final bf16 rounding, and output-view emission."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _gelu_bf16_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rsqrt_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_H)
    row_mask = row_ids[:, None] < ROWS
    col_mask = cols[None, :] < HIDDEN
    mask = row_mask & col_mask
    offsets = row_ids[:, None] * HIDDEN + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gelu = (x * 0.5) * (libdevice.erf(x * 0.7071067811865476) + 1.0)
    gelu_bf16_f32 = gelu.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)

    reduced = tl.where(mask, gelu_bf16_f32, 0.0)
    mean = tl.sum(reduced, axis=1) / HIDDEN
    centered = gelu_bf16_f32 - mean[:, None]
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / HIDDEN
    invstd = tl.rsqrt(variance + 1.0e-7)

    weight = tl.load(
        weight_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=cols < HIDDEN,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    out = centered * invstd[:, None] * weight[None, :] + bias[None, :]

    tl.store(mean_ptr + row_ids, mean, mask=row_ids < ROWS)
    tl.store(rsqrt_ptr + row_ids, invstd, mask=row_ids < ROWS)
    tl.store(
        out_ptr + offsets,
        out.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


def _launch(inputs, *, BLOCK_M: int, BLOCK_H: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    out_shape = tuple(int(dim) for dim in _shape_param_1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    stat_shape = (*view_shape[:-1], 1)
    stat_stride = (view_shape[1], 1, 1)

    mean = torch.empty_strided(
        stat_shape,
        stat_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        stat_shape,
        stat_stride,
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _gelu_bf16_layernorm_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        mean,
        rsqrt,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_M=BLOCK_M,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return mean, rsqrt, out


# cbab746f: (T([4096,1536], bf16), T([1536], f32), T([1536], f32), S([8,512,1536]), S([4096,1536]))
@oracle_impl(hardware="B200", point="cbab746f", BLOCK_M=1, BLOCK_H=2048, num_warps=8)
# b2a77780: (T([32768,768], bf16), T([768], f32), T([768], f32), S([256,128,768]), S([32768,768]))
@oracle_impl(hardware="B200", point="b2a77780", BLOCK_M=1, BLOCK_H=1024, num_warps=4)
# ca0eabd2: (T([16384,768], bf16), T([768], f32), T([768], f32), S([32,512,768]), S([16384,768]))
@oracle_impl(hardware="B200", point="ca0eabd2", BLOCK_M=1, BLOCK_H=1024, num_warps=4)
# 1398f333: (T([32768,128], bf16), T([128], f32), T([128], f32), S([64,512,128]), S([32768,128]))
@oracle_impl(hardware="B200", point="1398f333", BLOCK_M=4, BLOCK_H=128, num_warps=4)
# a565199e: (T([8192,768], bf16), T([768], f32), T([768], f32), S([8,1024,768]), S([8192,768]))
@oracle_impl(hardware="B200", point="a565199e", BLOCK_M=1, BLOCK_H=1024, num_warps=4)
# 3871c2e1: (T([8192,1024], bf16), T([1024], f32), T([1024], f32), S([16,512,1024]), S([8192,1024]))
@oracle_impl(hardware="B200", point="3871c2e1", BLOCK_M=1, BLOCK_H=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int, num_warps: int):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H, num_warps=num_warps)
