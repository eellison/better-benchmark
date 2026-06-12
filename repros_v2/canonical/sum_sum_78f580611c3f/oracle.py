"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete NFNet weight-standardization backward fragment in one Triton row kernel per output channel, sharing the bf16-to-f32 weight-gradient load, sibling per-channel sums, and dependent full-tensor epilogue for both returned tensors, whereas Inductor schedules the reductions and their full-size consumer as separate generic reduction/pointwise work across reshape-only views; Inductor cannot do this today because its scheduler lacks a full-scope multi-output reduction template that keeps small channel rows resident while sinking finalized reduction scalars into a materializing sibling epilogue; the fix is SCHEDULER_FUSION: add scheduler/codegen support for compatible sibling reductions with dependent full-output stores while preserving captured f32 cast and operation order."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sum_sum_full_scope_kernel(
    grad_ptr,
    weight_ptr,
    mean_ptr,
    invstd_ptr,
    gain_ptr,
    reduced_out_ptr,
    full_out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    row_offsets = c * K + offs

    grad = tl.load(grad_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    gain = tl.load(gain_ptr + c).to(tl.float32)

    centered = _sub_rn_f32(weight, mean)
    sum_grad = tl.sum(tl.where(mask, grad, 0.0), axis=0)
    sum_grad_centered = tl.sum(tl.where(mask, _mul_rn_f32(grad, centered), 0.0), axis=0)

    mean_grad = _mul_rn_f32(sum_grad, 0.0078125)
    correction = _mul_rn_f32(sum_grad_centered, 0.0078125)
    correction = _mul_rn_f32(correction, invstd)
    correction = _mul_rn_f32(correction, invstd)

    scaled_gain = _mul_rn_f32(gain, 0.08838834764831845)
    out_scale = _mul_rn_f32(invstd, scaled_gain)
    reduced = _mul_rn_f32(_mul_rn_f32(sum_grad_centered, invstd), 0.08838834764831845)

    full = _sub_rn_f32(grad, _mul_rn_f32(centered, correction))
    full = _sub_rn_f32(full, mean_grad)
    full = _mul_rn_f32(full, out_scale)

    tl.store(reduced_out_ptr + c, reduced)
    tl.store(full_out_ptr + row_offsets, full, mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="9cebd270", BLOCK_K=128, num_warps=4)
@oracle_impl(hardware="B200", point="fcb0a01d", BLOCK_K=128, num_warps=4)
@oracle_impl(hardware="B200", point="a28caacd", BLOCK_K=256, num_warps=4)
@oracle_impl(hardware="B200", point="4035c1ca", BLOCK_K=256, num_warps=4)
@oracle_impl(hardware="B200", point="a9796ac0", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="48a71583", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="27f8d48f", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="536e5c9c", BLOCK_K=1024, num_warps=8)
@oracle_impl(hardware="B200", point="f1e6452c", BLOCK_K=2048, num_warps=8)
@oracle_impl(hardware="B200", point="5274e21b", BLOCK_K=2048, num_warps=8)
@oracle_impl(hardware="B200", point="5b5eaa2a", BLOCK_K=2048, num_warps=8)
@oracle_impl(hardware="B200", point="60bfd0d5", BLOCK_K=128, num_warps=4)
@oracle_impl(hardware="B200", point="7da1502a", BLOCK_K=64, num_warps=4)
@oracle_impl(hardware="B200", point="91971ed7", BLOCK_K=256, num_warps=4)
@oracle_impl(hardware="B200", point="97760c9d", BLOCK_K=128, num_warps=4)
@oracle_impl(hardware="B200", point="1d8ef5f6", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="1d9c9eb5", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="6db1d2fa", BLOCK_K=512, num_warps=4)
@oracle_impl(hardware="B200", point="73422ca5", BLOCK_K=2048, num_warps=8)
@oracle_impl(hardware="B200", point="d2f96b40", BLOCK_K=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K, num_warps):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        _shape_param_0,
        _shape_param_1,
        shape_param_2,
        shape_param_3,
    ) = inputs
    c = int(arg0_1.shape[0])
    k = int(arg0_1.shape[1])

    reduced_out = torch.empty_strided(
        _shape_tuple(shape_param_2),
        (1, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    full_out = torch.empty_strided(
        _shape_tuple(shape_param_3),
        (k, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _sum_sum_full_scope_kernel[(c,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        reduced_out,
        full_out,
        K=k,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )
    return (reduced_out, full_out)
