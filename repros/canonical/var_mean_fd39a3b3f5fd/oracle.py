"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete FNet train real-select residual LayerNorm scope in one shape-specialized Triton row kernel, including `arg0_1[..., 0]`, the residual add, correction=0 hidden-size-768 var_mean, eps=1e-12 rsqrt, returned normalized tensor, returned f32 affine tensor, metadata-only affine view alias, and `rsqrt / 768` side output, whereas Inductor lowers the captured select/add/var_mean/affine/view graph through generic normalization scheduling; Inductor cannot do this today because correction=0 var_mean lowering keeps generic Welford reduction/codegen instead of selecting a fixed-hidden LayerNorm algebra that reuses the row tile for the affine and saved-scale epilogues; the fix is ALGEBRAIC_ELIMINATION: add a guarded fixed-hidden correction=0 LayerNorm lowering that replaces generic Welford bookkeeping with direct mean and centered-variance reductions while preserving all returned side outputs and alias-only views."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-12
HIDDEN = 768


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
def _fnet_train_layernorm_kernel(
    real_imag_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    normalized_ptr,
    affine_ptr,
    invstd_div_ptr,
    HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN

    dense_offsets = row * HIDDEN + cols
    real_offsets = dense_offsets * 2

    real = tl.load(
        real_imag_ptr + real_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    residual = tl.load(
        residual_ptr + dense_offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = _f32_add(real, residual)

    x_for_reduce = tl.where(mask, x, 0.0)
    mean = tl.sum(x_for_reduce, axis=0) / HIDDEN
    centered = _f32_sub(x, mean)
    centered_for_reduce = tl.where(mask, centered, 0.0)
    variance = tl.sum(_f32_mul(centered_for_reduce, centered_for_reduce), axis=0) / HIDDEN
    invstd = libdevice.rsqrt(_f32_add(variance, EPS))
    normalized = _f32_mul(centered, invstd)

    weight = tl.load(
        weight_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    bias = tl.load(
        bias_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    scaled = _f32_mul(normalized, weight)
    affine = _f32_add(scaled, bias)

    tl.store(normalized_ptr + dense_offsets, normalized, mask=mask)
    tl.store(affine_ptr + dense_offsets, affine, mask=mask)
    tl.store(invstd_div_ptr + row, invstd / HIDDEN)


def _shape(shape):
    return tuple(int(dim) for dim in shape)


# 98ade792: (T([32,512,768,2], f32), T([32,512,768], f32), T([768], f32), T([768], f32), S([16384,768]))
@oracle_impl(hardware="B200", point="98ade792", BLOCK_H=1024, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int, num_stages: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0 = inputs
    batch = int(arg1_1.shape[0])
    seq_len = int(arg1_1.shape[1])
    rows = batch * seq_len

    normalized = torch.empty_like(arg1_1)
    affine = torch.empty_like(arg1_1)
    invstd_div = torch.empty_strided(
        (batch, seq_len, 1),
        (seq_len, 1, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    _fnet_train_layernorm_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        normalized,
        affine,
        invstd_div,
        HIDDEN=HIDDEN,
        EPS=EPS,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return normalized, affine, affine.view(_shape(shape0)), invstd_div
