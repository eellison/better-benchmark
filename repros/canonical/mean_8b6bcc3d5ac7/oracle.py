"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Inception six-branch BN-inference affine, explicit bf16 rounding, NaN-preserving ReLU, fixed channel concatenations, 8x8 spatial mean, as_strided, and final contiguous bf16 [128,2048] view in one branch-selecting Triton reduction kernel, whereas Inductor schedules the branch pointwise producers, inner cats, outer cat, and spatial mean as generic producer/consumer regions around materialized concat-shaped intermediates; Inductor cannot do this today because its scheduler does not keep fixed channel-cat producers virtual across a downstream reduction that needs per-output-channel source selection while preserving the bf16 cast boundary before ReLU; the fix is SCHEDULER_FUSION: represent fixed-shape aten.cat as a virtual multi-source layout and sink each branch's BN/ReLU epilogue into the fused spatial-mean reduction that writes the final reshaped output."""

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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _inception_bn_relu_spatial_mean_kernel(
    mean0_ptr,
    x0_ptr,
    var0_ptr,
    weight0_ptr,
    bias0_ptr,
    mean1_ptr,
    x1_ptr,
    var1_ptr,
    weight1_ptr,
    bias1_ptr,
    mean2_ptr,
    x2_ptr,
    var2_ptr,
    weight2_ptr,
    bias2_ptr,
    mean3_ptr,
    x3_ptr,
    var3_ptr,
    weight3_ptr,
    bias3_ptr,
    mean4_ptr,
    x4_ptr,
    var4_ptr,
    weight4_ptr,
    bias4_ptr,
    mean5_ptr,
    x5_ptr,
    var5_ptr,
    weight5_ptr,
    bias5_ptr,
    out_ptr,
    BLOCK_ROWS: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = row_offsets < 262144
    n_offsets = row_offsets // 2048
    out_c_offsets = row_offsets - n_offsets * 2048

    hw_offsets = tl.arange(0, 64)
    h_offsets = hw_offsets // 8
    w_offsets = hw_offsets - h_offsets * 8

    x = tl.full((BLOCK_ROWS, 64), 0.0, tl.float32)
    mean = tl.full((BLOCK_ROWS,), 0.0, tl.float32)
    var = tl.full((BLOCK_ROWS,), 0.0, tl.float32)
    weight = tl.full((BLOCK_ROWS,), 0.0, tl.float32)
    bias = tl.full((BLOCK_ROWS,), 0.0, tl.float32)

    in0 = out_c_offsets < 320
    c0 = tl.where(in0, out_c_offsets, 0)
    mask0 = row_mask & in0
    x0_offsets = (
        n_offsets[:, None] * 20480
        + c0[:, None]
        + h_offsets[None, :] * 2560
        + w_offsets[None, :] * 320
    )
    x = tl.where(
        in0[:, None],
        tl.load(x0_ptr + x0_offsets, mask=mask0[:, None], other=0.0).to(tl.float32),
        x,
    )
    mean = tl.where(in0, tl.load(mean0_ptr + c0, mask=mask0, other=0.0).to(tl.float32), mean)
    var = tl.where(in0, tl.load(var0_ptr + c0, mask=mask0, other=0.0).to(tl.float32), var)
    weight = tl.where(
        in0, tl.load(weight0_ptr + c0, mask=mask0, other=0.0).to(tl.float32), weight
    )
    bias = tl.where(in0, tl.load(bias0_ptr + c0, mask=mask0, other=0.0).to(tl.float32), bias)

    in1 = (out_c_offsets >= 320) & (out_c_offsets < 704)
    c1 = tl.where(in1, out_c_offsets - 320, 0)
    mask1 = row_mask & in1
    x1_offsets = (
        n_offsets[:, None] * 24576
        + c1[:, None]
        + h_offsets[None, :] * 3072
        + w_offsets[None, :] * 384
    )
    x = tl.where(
        in1[:, None],
        tl.load(x1_ptr + x1_offsets, mask=mask1[:, None], other=0.0).to(tl.float32),
        x,
    )
    mean = tl.where(in1, tl.load(mean1_ptr + c1, mask=mask1, other=0.0).to(tl.float32), mean)
    var = tl.where(in1, tl.load(var1_ptr + c1, mask=mask1, other=0.0).to(tl.float32), var)
    weight = tl.where(
        in1, tl.load(weight1_ptr + c1, mask=mask1, other=0.0).to(tl.float32), weight
    )
    bias = tl.where(in1, tl.load(bias1_ptr + c1, mask=mask1, other=0.0).to(tl.float32), bias)

    in2 = (out_c_offsets >= 704) & (out_c_offsets < 1088)
    c2 = tl.where(in2, out_c_offsets - 704, 0)
    mask2 = row_mask & in2
    x2_offsets = (
        n_offsets[:, None] * 24576
        + c2[:, None]
        + h_offsets[None, :] * 3072
        + w_offsets[None, :] * 384
    )
    x = tl.where(
        in2[:, None],
        tl.load(x2_ptr + x2_offsets, mask=mask2[:, None], other=0.0).to(tl.float32),
        x,
    )
    mean = tl.where(in2, tl.load(mean2_ptr + c2, mask=mask2, other=0.0).to(tl.float32), mean)
    var = tl.where(in2, tl.load(var2_ptr + c2, mask=mask2, other=0.0).to(tl.float32), var)
    weight = tl.where(
        in2, tl.load(weight2_ptr + c2, mask=mask2, other=0.0).to(tl.float32), weight
    )
    bias = tl.where(in2, tl.load(bias2_ptr + c2, mask=mask2, other=0.0).to(tl.float32), bias)

    in3 = (out_c_offsets >= 1088) & (out_c_offsets < 1472)
    c3 = tl.where(in3, out_c_offsets - 1088, 0)
    mask3 = row_mask & in3
    x3_offsets = (
        n_offsets[:, None] * 24576
        + c3[:, None]
        + h_offsets[None, :] * 3072
        + w_offsets[None, :] * 384
    )
    x = tl.where(
        in3[:, None],
        tl.load(x3_ptr + x3_offsets, mask=mask3[:, None], other=0.0).to(tl.float32),
        x,
    )
    mean = tl.where(in3, tl.load(mean3_ptr + c3, mask=mask3, other=0.0).to(tl.float32), mean)
    var = tl.where(in3, tl.load(var3_ptr + c3, mask=mask3, other=0.0).to(tl.float32), var)
    weight = tl.where(
        in3, tl.load(weight3_ptr + c3, mask=mask3, other=0.0).to(tl.float32), weight
    )
    bias = tl.where(in3, tl.load(bias3_ptr + c3, mask=mask3, other=0.0).to(tl.float32), bias)

    in4 = (out_c_offsets >= 1472) & (out_c_offsets < 1856)
    c4 = tl.where(in4, out_c_offsets - 1472, 0)
    mask4 = row_mask & in4
    x4_offsets = (
        n_offsets[:, None] * 24576
        + c4[:, None]
        + h_offsets[None, :] * 3072
        + w_offsets[None, :] * 384
    )
    x = tl.where(
        in4[:, None],
        tl.load(x4_ptr + x4_offsets, mask=mask4[:, None], other=0.0).to(tl.float32),
        x,
    )
    mean = tl.where(in4, tl.load(mean4_ptr + c4, mask=mask4, other=0.0).to(tl.float32), mean)
    var = tl.where(in4, tl.load(var4_ptr + c4, mask=mask4, other=0.0).to(tl.float32), var)
    weight = tl.where(
        in4, tl.load(weight4_ptr + c4, mask=mask4, other=0.0).to(tl.float32), weight
    )
    bias = tl.where(in4, tl.load(bias4_ptr + c4, mask=mask4, other=0.0).to(tl.float32), bias)

    in5 = out_c_offsets >= 1856
    c5 = tl.where(in5, out_c_offsets - 1856, 0)
    mask5 = row_mask & in5
    x5_offsets = (
        n_offsets[:, None] * 12288
        + c5[:, None]
        + h_offsets[None, :] * 1536
        + w_offsets[None, :] * 192
    )
    x = tl.where(
        in5[:, None],
        tl.load(x5_ptr + x5_offsets, mask=mask5[:, None], other=0.0).to(tl.float32),
        x,
    )
    mean = tl.where(in5, tl.load(mean5_ptr + c5, mask=mask5, other=0.0).to(tl.float32), mean)
    var = tl.where(in5, tl.load(var5_ptr + c5, mask=mask5, other=0.0).to(tl.float32), var)
    weight = tl.where(
        in5, tl.load(weight5_ptr + c5, mask=mask5, other=0.0).to(tl.float32), weight
    )
    bias = tl.where(in5, tl.load(bias5_ptr + c5, mask=mask5, other=0.0).to(tl.float32), bias)

    centered = _f32_sub(x, mean[:, None])
    var_eps = _f32_add(var, 0.001)
    sqrt = tl.sqrt_rn(var_eps)
    recip = _f32_div(1.0, sqrt)
    scale = _f32_mul(recip, 1.0)
    normalized = _f32_mul(centered, scale[:, None])
    affine = _f32_add(_f32_mul(normalized, weight[:, None]), bias[:, None])
    rounded = affine.to(tl.bfloat16).to(tl.float32)
    relu = tl.where(rounded != rounded, rounded, tl.maximum(rounded, 0.0))
    total = tl.sum(tl.where(row_mask[:, None], relu, 0.0), axis=1)
    pooled = _f32_div(total, 64.0)

    tl.store(out_ptr + row_offsets, pooled.to(tl.bfloat16), mask=row_mask)


# 13ab6fb6: six bf16 channels-last Inception BN/ReLU branches -> bf16 [128,2048]
@oracle_impl(hardware="B200", point="13ab6fb6", BLOCK_ROWS=32, num_warps=2)
def oracle_forward(inputs, *, BLOCK_ROWS: int, num_warps: int):
    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        mean3,
        x3,
        var3,
        weight3,
        bias3,
        mean4,
        x4,
        var4,
        weight4,
        bias4,
        mean5,
        x5,
        var5,
        weight5,
        bias5,
        _shape,
        _stride,
        _view,
    ) = inputs
    out = torch.empty_strided(
        (128, 2048),
        (2048, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    _inception_bn_relu_spatial_mean_kernel[(triton.cdiv(128 * 2048, BLOCK_ROWS),)](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        mean3,
        x3,
        var3,
        weight3,
        bias3,
        mean4,
        x4,
        var4,
        weight4,
        bias4,
        mean5,
        x5,
        var5,
        weight5,
        bias5,
        out,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
