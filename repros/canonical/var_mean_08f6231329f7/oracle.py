"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ALBERT bf16 tanh-GELU producer plus hidden-size-128 LayerNorm inference scope in one Triton row kernel, including the metadata-only `[4096,128] -> [8,512,128]` view, every captured bf16 rounding boundary in `0.5*x*(1+tanh(0.7978845608028654*(x+0.044715*x^3)))`, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt, bf16 scale/bias affine epilogue, final bf16 cast, and contiguous `[4096,128]` view return, whereas Inductor lowers the dtype-sensitive GELU producer, row statistics, affine cast, and output view through generic pointwise and normalization-template scheduling; Inductor cannot fuse this full scope today because the fixed-hidden normalization scheduler does not keep bf16-rounded tanh-GELU producer values resident across row statistics and affine output stores; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline bf16 tanh-GELU producers with exact dtype boundaries into one guarded row-normalization plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-12


@triton.jit
def _bf16_round(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _gelu_layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_offsets[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    valid = (row_offsets[:, None] < ROWS) & (cols < HIDDEN)
    offsets = rows * HIDDEN + cols

    x = tl.load(
        x_ptr + offsets,
        mask=valid,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    half = _bf16_round(x * 0.5)
    square = _bf16_round(x * x)
    cubic = _bf16_round(square * x)
    cubic_scaled = _bf16_round(cubic * 0.044715)
    poly = _bf16_round(x + cubic_scaled)
    tanh_arg = _bf16_round(poly * 0.7978845608028654)
    tanh_val = _bf16_round(libdevice.tanh(tanh_arg))
    one_plus_tanh = _bf16_round(tanh_val + 1.0)
    gelu = _bf16_round(half * one_plus_tanh)

    reduce_input = tl.where(valid, gelu, 0.0)
    mean = tl.sum(reduce_input, axis=1) * (1.0 / HIDDEN)
    centered = gelu - mean[:, None]
    centered_masked = tl.where(valid, centered, 0.0)
    variance = tl.sum(centered_masked * centered_masked, axis=1) * (1.0 / HIDDEN)
    invstd = libdevice.rsqrt(variance + EPSILON)
    normalized = centered * invstd[:, None]

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
    affine = normalized * weight + bias
    tl.store(
        out_ptr + offsets,
        affine.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=valid,
    )


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# 699e8097: AlbertForMaskedLM infer tanh-GELU + LayerNorm, hidden=128, rows=4096.
@oracle_impl(
    hardware="B200",
    point="699e8097",
    BLOCK_H=128,
    ROW_BLOCK=4,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    ROW_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    x, weight, bias, _view_shape, out_shape = inputs
    output_shape = _shape_tuple(out_shape)
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    out = torch.empty_strided(
        output_shape,
        _contiguous_stride(output_shape),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _gelu_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        x,
        weight,
        bias,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
