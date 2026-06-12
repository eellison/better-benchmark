"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete two-stage bf16 RMSNorm scope in one shape-specialized Triton row kernel, including the first fp32 mean-square reduction, `(weight + 1)` affine, bf16 cast, visible bf16 residual add output, second fp32 mean-square reduction over that rounded add, second `(weight + 1)` affine, bf16 cast, and three alias-only flattened views from one final buffer, whereas Inductor lowers the chained add/mean/rsqrt/affine/add/mean/rsqrt/affine/view graph through generic reduction scheduling with separate producer/consumer regions; Inductor cannot do this today because the normalization scheduler does not keep the first normalized tile and required bf16 rounded residual add live across the second fixed-hidden reduction while also preserving the returned intermediate and alias-only view outputs; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to fuse chained RMSNorm rows with explicit dtype boundaries, visible residual side outputs, and repeated view aliases into one full-scope row plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _dual_rmsnorm_bf16_kernel(
    x0_ptr,
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

    x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum0 = tl.sum(tl.where(mask, x0 * x0, 0.0), axis=0)
    inv0 = tl.rsqrt(sum0 / HIDDEN + 1.0e-6)
    w0 = tl.load(weight0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    norm0_bf16 = ((x0 * inv0) * (w0 + 1.0)).to(tl.bfloat16)

    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_bf16 = (residual + norm0_bf16.to(tl.float32)).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    x1 = add_bf16.to(tl.float32)
    sum1 = tl.sum(tl.where(mask, x1 * x1, 0.0), axis=0)
    inv1 = tl.rsqrt(sum1 / HIDDEN + 1.0e-6)
    w1 = tl.load(weight1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    norm1_bf16 = ((x1 * inv1) * (w1 + 1.0)).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, norm1_bf16, mask=mask)


# ab924692: (T([1000,2304], bf16), T([2304], bf16), T([1,1000,2304], bf16), T([2304], bf16), ...)
# 5c119e0a: (T([1000,2560], bf16), T([2560], bf16), T([1,1000,2560], bf16), T([2560], bf16), ...)
@oracle_impl(hardware="B200", point="ab924692", BLOCK_H=4096, num_warps=8)
@oracle_impl(hardware="B200", point="5c119e0a", BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1, shape2, shape3 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in shape0)

    add_out = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        add_shape,
        (add_shape[1] * add_shape[2], add_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _dual_rmsnorm_bf16_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        add_out,
        norm_base,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return (
        add_out,
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
        norm_base.view(tuple(int(dim) for dim in shape3)),
    )
