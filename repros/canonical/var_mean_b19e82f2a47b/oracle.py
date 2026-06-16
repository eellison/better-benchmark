"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 residual-add LayerNorm scope in one shape-specialized Triton row kernel, including the metadata view of the first input, bf16-rounded add side output, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 rsqrt affine epilogue, final bf16 cast, and the three returned flattened view aliases from one normalized output buffer, whereas Inductor lowers the captured add/var_mean/affine/view graph through its generic normalization scheduler; Inductor cannot do this today because the norm-template scheduler does not fuse the bf16 residual producer, required side-output store, fixed-hidden row reduction, and repeated alias-view returns into one full-scope plan; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to inline same-layout residual adds with visible side outputs and emit alias-only view returns from the single normalized storage."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _residual_layernorm_bf16_h2560_kernel(
    x0_ptr,
    x1_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
):
    row = tl.program_id(0)
    cols0 = tl.arange(0, 2048)
    cols1 = tl.arange(0, 512) + 2048
    base = row * 2560
    offsets0 = base + cols0
    offsets1 = base + cols1

    x0_0 = tl.load(x0_ptr + offsets0).to(tl.float32)
    x1_0 = tl.load(x1_ptr + offsets0).to(tl.float32)
    x0_1 = tl.load(x0_ptr + offsets1).to(tl.float32)
    x1_1 = tl.load(x1_ptr + offsets1).to(tl.float32)

    add0 = (x1_0 + x0_0).to(tl.bfloat16)
    add1 = (x1_1 + x0_1).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets0, add0)
    tl.store(add_out_ptr + offsets1, add1)

    x_0 = add0.to(tl.float32)
    x_1 = add1.to(tl.float32)
    mean = (tl.sum(x_0, axis=0) + tl.sum(x_1, axis=0)) / 2560.0
    centered0 = x_0 - mean
    centered1 = x_1 - mean
    var = (
        tl.sum(centered0 * centered0, axis=0)
        + tl.sum(centered1 * centered1, axis=0)
    ) / 2560.0
    invstd = tl.rsqrt(var + 1.0e-5)

    weight0 = tl.load(weight_ptr + cols0).to(tl.float32)
    bias0 = tl.load(bias_ptr + cols0).to(tl.float32)
    weight1 = tl.load(weight_ptr + cols1).to(tl.float32)
    bias1 = tl.load(bias_ptr + cols1).to(tl.float32)
    y0 = centered0 * invstd * weight0 + bias0
    y1 = centered1 * invstd * weight1 + bias1
    tl.store(norm_out_ptr + offsets0, y0.to(tl.bfloat16))
    tl.store(norm_out_ptr + offsets1, y1.to(tl.bfloat16))


@triton.jit
def _residual_layernorm_bf16_kernel(
    x0_ptr,
    x1_ptr,
    weight_ptr,
    bias_ptr,
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
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_bf16 = (x1 + x0).to(tl.bfloat16)
    tl.store(add_out_ptr + offsets, add_bf16, mask=mask)

    x = add_bf16.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / HIDDEN
    centered = x - mean
    var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / HIDDEN
    invstd = tl.rsqrt(var + 1.0e-5)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = centered * invstd * weight + bias
    tl.store(norm_out_ptr + offsets, y.to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="c4bf51cc", BLOCK_H=4096, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="7f824027", BLOCK_H=4096, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="f2c837cd", BLOCK_H=2048, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="17affd46", BLOCK_H=1024, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="0b3dc49f", BLOCK_H=1024, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="ceab07f0", BLOCK_H=1024, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="a3e95c29", BLOCK_H=1024, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="f2e11670", BLOCK_H=1024, num_warps=4, num_stages=4)
@oracle_impl(hardware="B200", point="324149d9", BLOCK_H=512, num_warps=1, num_stages=4)
def oracle_forward(inputs, *, BLOCK_H, num_warps, num_stages):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, shape1, shape2, shape3 = inputs
    add_out = torch.empty_like(arg1_1)
    norm_base = torch.empty_like(arg1_1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])

    if hidden == 2560:
        _residual_layernorm_bf16_h2560_kernel[(rows,)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            add_out,
            norm_base,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _residual_layernorm_bf16_kernel[(rows,)](
            arg0_1,
            arg1_1,
            arg2_1,
            arg3_1,
            add_out,
            norm_base,
            HIDDEN=hidden,
            BLOCK_H=BLOCK_H,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return (
        add_out,
        norm_base.view(tuple(shape1)),
        norm_base.view(tuple(shape2)),
        norm_base.view(tuple(shape3)),
    )
