"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual-add RMSNorm scope in one shape-specialized Triton row kernel, including the returned bf16 add tensor, fp32 mean-square reduction, eps=1e-6 rsqrt normalization, bf16 cast boundary, bf16 affine weight multiply, and the three returned flattened view aliases from one normalized output buffer, whereas Inductor lowers the captured add/mean/rsqrt/affine/view graph through its generic fused reduction scheduler; Inductor cannot do this today because the normalization scheduler does not fuse a live returned residual producer, fixed-hidden RMSNorm reduction, required bf16 rounding points, and repeated alias-only view returns into one full-scope row plan; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to inline same-layout residual adds with visible side outputs and emit alias-only view returns from the single normalized storage."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _residual_rmsnorm_bf16_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    add_out_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN
    offsets = row * HIDDEN + cols

    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    added_bf16 = (residual + flat).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, added_bf16, mask=mask)

    x = added_bf16.to(tl.float32)
    square_sum = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
    inv_rms = tl.rsqrt(square_sum / HIDDEN + 1.0e-6)
    normalized_bf16 = (x * inv_rms).to(tl.bfloat16)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (normalized_bf16.to(tl.float32) * weight).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, out, mask=mask)


# (T([1000,1024], bf16), T([1,1000,1024], bf16), T([1024], bf16), ...)
@oracle_impl(hardware="B200", point="da8b94aa", BLOCK_H=1024, num_warps=4)
# (T([1000,4096], bf16), T([1,1000,4096], bf16), T([4096], bf16), ...)
@oracle_impl(hardware="B200", point="111af936", BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H, num_warps):
    arg0_1, arg1_1, arg2_1, _shape_param_0, shape1, shape2, shape3 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = tuple(int(dim) for dim in _shape_param_0)

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

    _residual_rmsnorm_bf16_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        add_out,
        norm_base,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
    )
    return (
        add_out,
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
        norm_base.view(tuple(int(dim) for dim in shape3)),
    )
