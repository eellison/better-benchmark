"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DINOv2 bf16 scaled-residual LayerNorm scope in one shape-specialized Triton row kernel, including the `[175360,768] -> [128,1370,768]` metadata view, bf16 scaled residual-add side output, fp32 correction=0 `var_mean` over hidden size 768, `rsqrt(var + 1e-6)`, affine scale/bias, final bf16 cast, and returned contiguous `[175360,768]` view, whereas Inductor lowers this canonicalized normalization graph through its generic norm-template schedule with separately visible producer/output stores; Inductor cannot do this today because the scheduler does not keep the visible bf16 scaled residual-add producer and dependent LayerNorm epilogue in one full-scope row template while preserving the returned side tensor; the fix is SCHEDULER_FUSION: extend the normalization template to sink visible bf16 producer stores and final affine materialization into one guarded hidden-size row schedule."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _scaled_residual_layernorm_bf16_kernel(
    source_ptr,
    scale_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    scaled = _f32_mul(source, scale).to(tl.bfloat16).to(tl.float32)
    x = _f32_add(residual, scaled).to(tl.bfloat16).to(tl.float32)
    tl.store(add_out_ptr + offsets, x, mask=mask)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=0) / HIDDEN
    centered = _f32_sub(x, mean)
    variance = tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=0) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-6))

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(norm_out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


# (T([175360,768], bf16), T([768], bf16), T([128,1370,768], bf16), T([768], bf16), T([768], bf16), S([128,1370,768]), S([175360,768]))
@oracle_impl(hardware="B200", point="01a8f9c7", BLOCK_H=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    source, scale, residual, weight, bias, add_shape, norm_shape = inputs
    add_shape = _shape_tuple(add_shape)
    norm_shape = _shape_tuple(norm_shape)
    hidden = int(source.shape[1])
    rows = int(source.shape[0])

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=source.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape,
        _contiguous_stride(norm_shape),
        device=source.device,
        dtype=torch.bfloat16,
    )

    _scaled_residual_layernorm_bf16_kernel[(rows,)](
        source,
        scale,
        residual,
        weight,
        bias,
        add_out,
        norm_out,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return add_out, norm_out
