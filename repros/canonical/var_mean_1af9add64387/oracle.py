"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual LayerNorm alias scope in one fixed-hidden Triton row kernel, including the `[16384,768] -> [32,512,768]` view, the bf16-rounded residual add from the captured graph, fp32 correction=0 row statistics over hidden, `rsqrt(var + 1e-12)`, affine epilogue with bf16 weight and bias, final bf16 cast, four `[16384,768]` alias views, and the `[32,768,512]` permute view, whereas Inductor lowers the decomposed view/add/var_mean/affine graph through its generic normalization schedule and metadata view handling; Inductor cannot do this today because the normalization scheduler does not expose a full-scope fixed-K row template that keeps the residual-add tile resident through the reduction, affine cast, and all returned alias/permute outputs; the fix is SCHEDULER_FUSION: extend the LayerNorm row schedule to fuse the residual producer, correction=0 row statistics, exact bf16 cast boundaries, and alias-preserving multi-output epilogue."""

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
def _residual_layernorm_alias_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (row < ROWS) & (cols < HIDDEN)
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x = _f32_add(flat, residual).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)
    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1)[:, None] / HIDDEN
    mean_square = (
        tl.sum(tl.where(mask, _f32_mul(x, x), 0.0), axis=1)[:, None] / HIDDEN
    )
    variance = _f32_add(mean_square, -_f32_mul(mean, mean))
    centered = _f32_add(x, -mean)
    invstd = libdevice.rsqrt(_f32_add(variance, 1.0e-12))
    normalized = _f32_mul(centered, invstd)
    scaled = _f32_mul(normalized, scale)
    y = _f32_add(scaled, bias).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + offsets, y, mask=mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


# 63bebcf6: (T([16384,768], bf16), T([32,512,768], bf16), T([768], bf16), T([768], bf16), ...)
@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_H=1024, ROW_BLOCK=1, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3, shape4 = inputs
    base_shape = tuple(int(dim) for dim in shape0)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    out = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _residual_layernorm_alias_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return (
        out,
        out.view(tuple(int(dim) for dim in shape1)),
        out.view(tuple(int(dim) for dim in shape2)),
        out.view(tuple(int(dim) for dim in shape3)),
        out.view(tuple(int(dim) for dim in shape4)),
        out.permute(0, 2, 1),
    )
