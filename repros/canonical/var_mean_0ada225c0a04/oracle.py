"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual-add LayerNorm scope in one shape-specialized Triton row kernel, including the metadata-only view of the flat input, Inductor's fused fp32 residual-add path for the unreturned producer, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 rsqrt affine epilogue, final bf16 cast, and returned contiguous flattened view, whereas Inductor lowers the captured add/var_mean/affine/view graph through its generic normalization scheduler; Inductor cannot do this today because the norm-template scheduler does not fuse the same-layout residual producer into the fixed-hidden row reduction and final store for every recorded model shape; the fix is SCHEDULER_FUSION: teach LayerNorm scheduling to inline residual adds into row statistics and emit the flattened output directly from the normalization epilogue."""

import torch
import triton
import triton.language as tl

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
def _residual_layernorm_kernel(
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
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    mask = (rows < ROWS) & (cols < HIDDEN)
    offsets = rows * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    x = _f32_add(residual.to(tl.float32), flat.to(tl.float32)).to(tl.bfloat16).to(tl.float32)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=1)[:, None] / HIDDEN
    centered = _f32_sub(x, mean)
    variance = tl.sum(tl.where(mask, _f32_mul(centered, centered), 0.0), axis=1)[:, None] / HIDDEN
    invstd = tl.rsqrt(_f32_add(variance, 1.0e-6))

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    tl.store(out_ptr + offsets, affine.to(tl.bfloat16), mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="c4bf51cc", BLOCK_H=4096, ROW_BLOCK=1, USE_EXACT=False, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="7f824027", BLOCK_H=4096, ROW_BLOCK=1, USE_EXACT=False, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="f2c837cd", BLOCK_H=2048, ROW_BLOCK=1, USE_EXACT=False, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="17affd46", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="0b3dc49f", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="ceab07f0", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="a3e95c29", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="d4cc3e3e", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="f2e11670", BLOCK_H=1024, ROW_BLOCK=2, USE_EXACT=False, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="9801ab6a", BLOCK_H=1024, ROW_BLOCK=4, USE_EXACT=True, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="1cea4d76", BLOCK_H=1024, ROW_BLOCK=4, USE_EXACT=True, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="d1f40ce2", BLOCK_H=1024, ROW_BLOCK=4, USE_EXACT=False, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="324149d9", BLOCK_H=512, ROW_BLOCK=1, USE_EXACT=False, num_warps=1, num_stages=4)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK, USE_EXACT, num_warps, num_stages):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    out_shape = _shape_tuple(_shape_param_1)
    out = torch.empty_strided(
        out_shape,
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _residual_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
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
    return out
