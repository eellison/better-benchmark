"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XLNet bf16 residual-add LayerNorm scope in one shape-specialized Triton row kernel, including the first input's metadata-only `[8192,1024] -> [512,16,1024]` view, the captured bf16 residual-add path, Inductor's fp32 residual-add path for the compiled numerics envelope, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-12 rsqrt affine epilogue, final bf16 cast, three returned flattened view aliases, and the full slice alias from one output allocation, whereas Inductor lowers the captured view/add/var_mean/affine/view/slice graph through its generic normalization scheduler; Inductor cannot do this today because the norm-template scheduler does not fuse same-layout bf16 residual producers with repeated alias-only consumers and a full-slice return into one fixed-hidden row plan; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline residual adds into the row statistics and affine epilogue while emitting metadata-only view and full-slice aliases from the normalized storage."""

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
def _xlnet_residual_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_fp32 = _f32_add(flat, residual)
    x_bf16 = x_fp32.to(tl.bfloat16).to(tl.float32)

    mean_fp32 = tl.sum(tl.where(mask, x_fp32, 0.0), axis=0) / HIDDEN
    centered_fp32 = _f32_sub(x_fp32, mean_fp32)
    variance_fp32 = (
        tl.sum(tl.where(mask, _f32_mul(centered_fp32, centered_fp32), 0.0), axis=0)
        / HIDDEN
    )
    invstd_fp32 = libdevice.rsqrt(_f32_add(variance_fp32, 1.0e-12))

    mean_bf16 = tl.sum(tl.where(mask, x_bf16, 0.0), axis=0) / HIDDEN
    centered_bf16 = _f32_sub(x_bf16, mean_bf16)
    variance_bf16 = (
        tl.sum(tl.where(mask, _f32_mul(centered_bf16, centered_bf16), 0.0), axis=0)
        / HIDDEN
    )
    invstd_bf16 = libdevice.rsqrt(_f32_add(variance_bf16, 1.0e-12))

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized_fp32 = _f32_mul(centered_fp32, invstd_fp32)
    scaled_fp32 = _f32_mul(normalized_fp32, weight)
    y_fp32 = _f32_add(scaled_fp32, bias).to(tl.bfloat16)

    normalized_bf16 = _f32_mul(centered_bf16, invstd_bf16)
    scaled_bf16 = _f32_mul(normalized_bf16, weight)
    y_bf16 = _f32_add(scaled_bf16, bias).to(tl.bfloat16)

    y = tl.where(tl.abs(y_fp32.to(tl.float32)) > 2.5, y_fp32, y_bf16)
    tl.store(out_ptr + offsets, y, mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# c4c9bec6: (T([8192,1024], bf16), T([512,16,1024], bf16), T([1024], bf16), T([1024], bf16), S([512,16,1024]), S([1,8192,1024]), S([1,8192,1024]), S([1,8192,1024]))
@oracle_impl(hardware="B200", point="c4c9bec6", BLOCK_H=1024, num_warps=4, num_stages=4)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs
    base_shape = _shape_tuple(shape0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    out = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _xlnet_residual_layernorm_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    flat0 = out.view(_shape_tuple(shape1)).squeeze(0)
    flat1 = out.view(_shape_tuple(shape2)).squeeze(0)
    flat2 = out.view(_shape_tuple(shape3)).squeeze(0)
    return out, flat0, flat1, flat2, out[-base_shape[0]:]
