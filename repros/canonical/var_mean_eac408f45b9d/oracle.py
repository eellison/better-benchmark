"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GPT-J bf16 residual-add LayerNorm scope in one shape-specialized Triton row kernel, including both bf16 residual-add rounding points, the returned `[1,128,4096]` add tensor, a correction=0 raw-moment fp32 var_mean, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and four returned `[128,4096]` view aliases from one normalized storage, whereas Inductor lowers the captured add/var_mean/affine/view-alias graph through its generic normalization scheduler with Welford-style reduction bookkeeping; Inductor cannot do this today because correction=0 LayerNorm lowering does not select a guarded raw-moment row template that preserves the visible bf16 producer store and duplicate alias-view contract; the fix is ALGEBRAIC_ELIMINATION: add a fixed-hidden correction=0 LayerNorm lowering that uses sum and sum-of-squares moments when the numeric policy allows it and sinks the returned producer plus alias-only outputs into the same schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _triple_residual_layernorm_bf16_kernel(
    x0_ptr,
    x1_ptr,
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
    offsets = row * HIDDEN + cols

    x0 = tl.load(x0_ptr + offsets).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets).to(tl.float32)
    residual = tl.load(residual_ptr + offsets).to(tl.float32)

    add0 = (x0 + x1).to(tl.bfloat16)
    add1 = (add0.to(tl.float32) + residual).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add1)

    x = add1.to(tl.float32)
    mean = tl.sum(x, axis=0) / HIDDEN
    var = tl.sum(x * x, axis=0) / HIDDEN - mean * mean
    var = tl.maximum(var, 0.0)
    centered = x - mean
    invstd = tl.rsqrt(var + 1.0e-5)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols).to(tl.float32)
    y = centered * invstd * weight + bias
    tl.store(norm_out_ptr + offsets, y.to(tl.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# (T([128,4096], bf16), T([128,4096], bf16), T([1,128,4096], bf16), T([4096], bf16), T([4096], bf16), ...)
@oracle_impl(hardware="B200", point="d9611874", BLOCK_H=4096, num_warps=8, num_stages=3)
def oracle_forward(inputs, *, BLOCK_H, num_warps, num_stages):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, _shape1, shape2, shape3, shape4, shape5 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    add_shape = _as_shape(shape0)

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

    _triple_residual_layernorm_bf16_kernel[(rows,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        add_out,
        norm_base,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return (
        add_out,
        norm_base.view(_as_shape(shape2)),
        norm_base.view(_as_shape(shape3)),
        norm_base.view(_as_shape(shape4)),
        norm_base.view(_as_shape(shape5)),
    )
