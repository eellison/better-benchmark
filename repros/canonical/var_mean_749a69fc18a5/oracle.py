"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16-plus-f32 GPT-Neo residual LayerNorm scope in one hidden-size-2048 Triton row kernel, including the `[4096,2048] -> [32,128,2048]` view, f32 residual add side output, correction=0 var_mean, eps=1e-5 rsqrt, f32 normalized side output, affine scale/bias, final bf16 `[4096,2048]` view, and `rsqrt / 2048` side output, whereas Inductor lowers the captured graph through a generic var_mean normalization schedule with the returned producer and normalized tensors handled as general side outputs; Inductor cannot do this today because its correction=0 var_mean lowering keeps the general reduction form instead of selecting a fixed-width LayerNorm algebra that keeps the row tile resident while sinking all dependent side-output and bf16 affine stores; the fix is ALGEBRAIC_ELIMINATION: add a guarded hidden-size LayerNorm lowering that replaces generic var_mean bookkeeping with direct mean and centered-variance reductions while preserving every returned tensor and cast boundary."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


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
def _residual_layernorm_kernel(
    bf16_input_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    bf16_out_ptr,
    invstd_div_ptr,
    HIDDEN: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    offsets = row * HIDDEN + cols

    x0 = tl.load(bf16_input_ptr + offsets).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    x = _f32_add(x0, residual)
    tl.store(add_out_ptr + offsets, x)

    mean = tl.sum(x, axis=0) / HIDDEN
    centered = _f32_sub(x, mean)
    variance = tl.sum(_f32_mul(centered, centered), axis=0) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPSILON))

    normalized = _f32_mul(centered, invstd)
    tl.store(norm_out_ptr + offsets, normalized)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols).to(tl.float32)
    scaled = _f32_mul(normalized, weight)
    affine = _f32_add(scaled, bias)
    tl.store(bf16_out_ptr + offsets, affine.to(tl.bfloat16))
    tl.store(invstd_div_ptr + row, invstd / HIDDEN)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# 5d43e450: (T([4096,2048], bf16), T([32,128,2048], f32), T([2048], f32), T([2048], f32), S([32,128,2048]), S([4096,2048]))
@oracle_impl(hardware="B200", point="5d43e450", BLOCK_H=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    bf16_input, residual, weight, bias, add_shape, out_shape = inputs
    add_shape = tuple(int(dim) for dim in add_shape)
    out_shape = tuple(int(dim) for dim in out_shape)
    rows = int(bf16_input.shape[0])
    hidden = int(bf16_input.shape[1])

    add_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=bf16_input.device,
        dtype=torch.float32,
    )
    norm_out = torch.empty_strided(
        add_shape,
        _contiguous_stride(add_shape),
        device=bf16_input.device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=bf16_input.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        (add_shape[0], add_shape[1], 1),
        (add_shape[1], 1, 1),
        device=bf16_input.device,
        dtype=torch.float32,
    )

    _residual_layernorm_kernel[(rows,)](
        bf16_input,
        residual,
        weight,
        bias,
        add_out,
        norm_out,
        bf16_out,
        invstd_div,
        HIDDEN=hidden,
        EPSILON=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return add_out, norm_out, bf16_out, invstd_div
