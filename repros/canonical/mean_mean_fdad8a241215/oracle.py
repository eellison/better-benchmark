"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete two-stage bf16 RMSNorm chain, including fp32 mean-square reductions, eps=1e-6 rsqrt, `(weight + 1)` affine multiplies, the explicit bf16 rounding before the returned residual add, the second RMSNorm over that bf16-rounded add, and the duplicate final alias views in one Triton row kernel, whereas Inductor lowers the decomposed graph as generic reduction and pointwise regions around a materialized intermediate; Inductor cannot do this today because its scheduler does not keep a fixed-hidden row tile live across dependent RMSNorm reductions while also preserving the returned side output and alias-view epilogue; the fix is SCHEDULER_FUSION: add a guarded dual-RMSNorm residual schedule that fuses both row reductions, the bf16 intermediate store, and the final alias-view store."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _dual_rmsnorm_kernel(
    x_ptr,
    weight0_ptr,
    residual_ptr,
    weight1_ptr,
    add_out_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_sq0 = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    inv_rms0 = tl.rsqrt(sum_sq0 * (1.0 / HIDDEN) + 1.0e-6)
    weight0 = tl.load(weight0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    first = (x * inv_rms0) * (weight0 + 1.0)
    first_bf16 = first.to(tl.bfloat16)

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_bf16 = (residual + first_bf16.to(tl.float32)).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    add_f32 = add_bf16.to(tl.float32)
    sum_sq1 = tl.sum(tl.where(mask, add_f32 * add_f32, 0.0), axis=0)
    inv_rms1 = tl.rsqrt(sum_sq1 * (1.0 / HIDDEN) + 1.0e-6)
    weight1 = tl.load(weight1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    second = (add_f32 * inv_rms1) * (weight1 + 1.0)
    tl.store(norm_out_ptr + offsets, second, mask=mask)


# ab924692: (T([1000, 2304], bf16), T([2304], bf16), T([1, 1000, 2304], bf16), T([2304], bf16), ...)
# 5c119e0a: (T([1000, 2560], bf16), T([2560], bf16), T([1, 1000, 2560], bf16), T([2560], bf16), ...)
@oracle_impl(hardware="B200", point="ab924692", BLOCK_H=4096, num_warps=8)
@oracle_impl(hardware="B200", point="5c119e0a", BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    x, weight0, residual, weight1, _shape0, out_shape0, out_shape1 = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])

    add_out = torch.empty_like(residual)
    norm_base = torch.empty_like(residual)
    _dual_rmsnorm_kernel[(rows,)](
        x,
        weight0,
        residual,
        weight1,
        add_out,
        norm_base,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=2,
    )
    return (
        add_out,
        norm_base.view(tuple(int(dim) for dim in out_shape0)),
        norm_base.view(tuple(int(dim) for dim in out_shape1)),
    )
