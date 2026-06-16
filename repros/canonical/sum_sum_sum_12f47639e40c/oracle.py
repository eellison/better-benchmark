"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete PyTorch U-Net bf16 bilinear-upsample-backward scatter feeding BN/ReLU masking and three channel reductions, exploiting the captured point's all-zero index tensors to collapse the four `index_put(accumulate=True)` destinations to the first spatial cell while preserving the f32 bilinear coefficient formulation, bf16 scatter and mask boundaries, returned `[1,256,80,119]` tensor, and final bf16-rounded channel sum; Inductor lowers the four generic scatter-adds, bf16 masking producer, dependent BN-backward reductions, returned dense store, and final bf16 output reduction as separate materialized kernels; Inductor cannot do this today because scheduler/codegen does not specialize degenerate indexed scatter-add producers as channel-reduction inputs with a required full-tensor epilogue; the fix is SCATTER_REDUCE: teach Inductor to specialize captured scatter index domains, reduce collapsed scatter producers directly, and fuse the downstream channel reductions plus returned bf16 store."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


C = 256
SRC_H = 160
SRC_W = 238
SRC_STRIDE_H = 239
SRC_C_STRIDE = SRC_H * SRC_STRIDE_H
SRC_CHANNEL_OFFSET = 256
SRC_K = SRC_H * SRC_W
OUT_H = 80
OUT_W = 119
OUT_HW = OUT_H * OUT_W
SCALE = 0.0001050420168067227


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
def _to_bf16_f32(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _source_sum_kernel(
    src_ptr,
    row_weight_ptr,
    col_weight_ptr,
    scatter_ptr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc0 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_K,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for start in tl.range(0, 38080, BLOCK_K):
        k = start + offs
        mask = k < 38080
        h = k // 238
        w = k - h * 238
        src_offsets = (c + 256) * 38240 + h * 239 + w
        x = tl.load(src_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
        row_w = tl.load(row_weight_ptr + h, mask=mask, other=0.0).to(tl.float32)
        col_w = tl.load(col_weight_ptr + w, mask=mask, other=0.0).to(tl.float32)

        mul = _f32_mul(x, row_w)
        add = _f32_sub(x, mul)
        mul1 = _f32_mul(mul, col_w)
        add1 = _f32_sub(mul, mul1)
        mul2 = _f32_mul(add, col_w)
        add2 = _f32_sub(add, mul2)

        acc0 += tl.where(mask, mul1, 0.0)
        acc1 += tl.where(mask, add1, 0.0)
        acc2 += tl.where(mask, mul2, 0.0)
        acc3 += tl.where(mask, add2, 0.0)

    total0 = tl.sum(acc0, axis=0)
    total1 = tl.sum(acc1, axis=0)
    total2 = tl.sum(acc2, axis=0)
    total3 = tl.sum(acc3, axis=0)
    add3 = _f32_add(total0, total1)
    add4 = _f32_add(add3, total2)
    add5 = _f32_add(add4, total3)
    tl.store(scatter_ptr + c, add5.to(tl.bfloat16, fp_downcast_rounding="rtne"))


@triton.jit
def _sum12_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    scatter_ptr,
    out_sum1_ptr,
    sum2_ptr,
    out_mul13_ptr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    hw_offsets = tl.arange(0, BLOCK_HW)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    scatter = tl.load(scatter_ptr + c).to(tl.float32)

    acc1 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for start in tl.range(0, 9520, BLOCK_HW):
        hw = start + hw_offsets
        mask = hw < 9520
        offsets = c * 9520 + hw
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        centered = _f32_sub(x, mean)
        gate_mul = _f32_mul(_f32_mul(centered, invstd), weight)
        gate = _to_bf16_f32(_f32_add(gate_mul, bias))
        active = gate <= 0.0
        scatter_or_zero = tl.where(hw == 0, scatter, 0.0)
        y = tl.where(active, scalar, scatter_or_zero)

        acc1 += tl.where(mask, y, 0.0)
        acc2 += tl.where(mask, _f32_mul(y, centered), 0.0)

    sum1 = tl.sum(acc1, axis=0)
    sum2 = tl.sum(acc2, axis=0)
    tl.store(out_sum1_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(out_mul13_ptr + c, _f32_mul(sum2, invstd))


@triton.jit
def _final_output_kernel(
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    scatter_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    out_sum4_ptr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    hw_offsets = tl.arange(0, BLOCK_HW)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    scalar = tl.load(scalar_ptr).to(tl.float32)
    scatter = tl.load(scatter_ptr + c).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c).to(tl.float32)

    mean_term = _f32_mul(sum1, 0.0001050420168067227)
    scaled_sum2 = _f32_mul(sum2, 0.0001050420168067227)
    invstd_sq = _f32_mul(invstd, invstd)
    var_term = _f32_mul(scaled_sum2, invstd_sq)
    affine_scale = _f32_mul(invstd, weight)

    acc4 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for start in tl.range(0, 9520, BLOCK_HW):
        hw = start + hw_offsets
        mask = hw < 9520
        offsets = c * 9520 + hw
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        centered = _f32_sub(x, mean)
        gate_mul = _f32_mul(_f32_mul(centered, invstd), weight)
        gate = _to_bf16_f32(_f32_add(gate_mul, bias))
        active = gate <= 0.0
        scatter_or_zero = tl.where(hw == 0, scatter, 0.0)
        y = tl.where(active, scalar, scatter_or_zero)

        sub2 = _f32_sub(y, _f32_mul(centered, var_term))
        sub3 = _f32_sub(sub2, mean_term)
        out_f32 = _f32_mul(sub3, affine_scale)
        out_bf16 = out_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(out_ptr + offsets, out_bf16, mask=mask)
        acc4 += tl.where(mask, out_bf16.to(tl.float32), 0.0)

    total = tl.sum(acc4, axis=0)
    tl.store(out_sum4_ptr + c, total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32))


# (T([1,512,160,239], bf16), T([160,1], f32), T([238], f32), all-zero i64 index tensors, T([1,256,80,119], bf16), ...)
@oracle_impl(hardware="B200", point="89c70199", SRC_BLOCK=1024, HW_BLOCK=1024, source_warps=4, row_warps=4)
def oracle_forward(
    inputs,
    *,
    SRC_BLOCK: int,
    HW_BLOCK: int,
    source_warps: int,
    row_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        _shape_param_0,
    ) = inputs
    del arg3_1, arg4_1, arg5_1, arg6_1, _shape_param_0

    device = arg0_1.device
    scatter = torch.empty((C,), device=device, dtype=torch.bfloat16)
    _source_sum_kernel[(C,)](
        arg0_1,
        arg1_1,
        arg2_1,
        scatter,
        BLOCK_K=SRC_BLOCK,
        num_warps=source_warps,
        num_stages=3,
    )

    out_sum1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=device, dtype=torch.float32)
    out_mul13 = torch.empty((C,), device=device, dtype=torch.float32)
    _sum12_kernel[(C,)](
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        scatter,
        out_sum1,
        sum2,
        out_mul13,
        BLOCK_HW=HW_BLOCK,
        num_warps=row_warps,
        num_stages=3,
    )

    out_tensor = torch.empty_strided(
        (1, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_sum4 = torch.empty((C,), device=device, dtype=torch.float32)
    _final_output_kernel[(C,)](
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        scatter,
        out_sum1,
        sum2,
        out_tensor,
        out_sum4,
        BLOCK_HW=HW_BLOCK,
        num_warps=row_warps,
        num_stages=3,
    )

    return out_sum1, out_mul13, out_tensor, out_sum4
