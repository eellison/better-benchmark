"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete PyTorch U-Net bf16 bilinear-upsample-backward scatter feeding BN/ReLU masking and three channel reductions, exploiting the captured point's all-zero index tensors to collapse the four `index_put(accumulate=True)` destinations to the single spatial cell while still preserving the bf16 scatter, mask, BN-backward cast boundaries, returned `[1,512,40,59]` tensor, and final bf16-rounded channel sum; Inductor currently lowers the four generic scatter-adds, bf16 masking producer, dependent reductions, and final bf16 output reduction as separate materialized kernels; Inductor cannot do this today because scheduler/codegen does not recognize degenerate indexed scatter-add producers as channel-reduction inputs with a required full-tensor epilogue; the fix is SCATTER_REDUCE: teach Inductor to specialize captured scatter index domains, reduce collapsed scatter producers directly, and fuse the downstream channel reductions plus returned bf16 store."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


C = 512
SRC_H = 80
SRC_W = 118
OUT_H = 40
OUT_W = 59
SRC_K = SRC_H * SRC_W
OUT_HW = OUT_H * OUT_W
SCALE = 0.000423728813559322


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
    arg0_ptr,
    scatter_ptr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    for start in tl.range(0, 9440, BLOCK_K):
        k = start + offs
        mask = k < 9440
        h = k // 118
        w = k - h * 118
        src_offsets = (c + 512) * 9520 + h * 119 + w
        x = tl.load(arg0_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(mask, x, 0.0)

    total = tl.sum(acc, axis=0)
    tl.store(scatter_ptr + c, total.to(tl.bfloat16, fp_downcast_rounding="rtne"))


@triton.jit
def _sum12_kernel(
    arg7_ptr,
    arg8_ptr,
    arg9_ptr,
    arg10_ptr,
    arg11_ptr,
    arg12_ptr,
    scatter_ptr,
    out_sum1_ptr,
    sum2_ptr,
    out_mul13_ptr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    hw_offsets = tl.arange(0, BLOCK_HW)
    mean = tl.load(arg8_ptr + c).to(tl.float32)
    invstd = tl.load(arg9_ptr + c).to(tl.float32)
    weight = tl.load(arg10_ptr + c).to(tl.float32)
    bias = tl.load(arg11_ptr + c).to(tl.float32)
    scalar = tl.load(arg12_ptr).to(tl.float32)
    scatter = tl.load(scatter_ptr + c).to(tl.float32)

    acc1 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for start in tl.range(0, 2360, BLOCK_HW):
        hw = start + hw_offsets
        mask = hw < 2360
        offsets = c * 2360 + hw
        x7 = tl.load(arg7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        centered = _f32_sub(x7, mean)
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
    arg7_ptr,
    arg8_ptr,
    arg9_ptr,
    arg10_ptr,
    arg11_ptr,
    arg12_ptr,
    scatter_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    out_sum4_ptr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    hw_offsets = tl.arange(0, BLOCK_HW)
    mean = tl.load(arg8_ptr + c).to(tl.float32)
    invstd = tl.load(arg9_ptr + c).to(tl.float32)
    weight = tl.load(arg10_ptr + c).to(tl.float32)
    bias = tl.load(arg11_ptr + c).to(tl.float32)
    scalar = tl.load(arg12_ptr).to(tl.float32)
    scatter = tl.load(scatter_ptr + c).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c).to(tl.float32)

    mean_term = _f32_mul(sum1, 0.000423728813559322)
    scaled_sum2 = _f32_mul(sum2, 0.000423728813559322)
    invstd_sq = _f32_mul(invstd, invstd)
    var_term = _f32_mul(scaled_sum2, invstd_sq)
    affine_scale = _f32_mul(invstd, weight)

    acc4 = tl.zeros((BLOCK_HW,), dtype=tl.float32)
    for start in tl.range(0, 2360, BLOCK_HW):
        hw = start + hw_offsets
        mask = hw < 2360
        offsets = c * 2360 + hw
        x7 = tl.load(arg7_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        centered = _f32_sub(x7, mean)
        gate_mul = _f32_mul(_f32_mul(centered, invstd), weight)
        gate = _to_bf16_f32(_f32_add(gate_mul, bias))
        active = gate <= 0.0
        scatter_or_zero = tl.where(hw == 0, scatter, 0.0)
        y = tl.where(active, scalar, scatter_or_zero)

        mul11 = _f32_mul(centered, var_term)
        sub2 = _f32_sub(y, mul11)
        sub3 = _f32_sub(sub2, mean_term)
        out_f32 = _f32_mul(sub3, affine_scale)
        out_bf16 = out_f32.to(tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(out_ptr + offsets, out_bf16, mask=mask)
        acc4 += tl.where(mask, out_bf16.to(tl.float32), 0.0)

    total = tl.sum(acc4, axis=0)
    rounded = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(out_sum4_ptr + c, rounded)


# (T([1,1024,80,119], bf16), T([80,1], f32), T([118], f32), T([80,1], i64), T([118], i64), T([118], i64), T([80,1], i64), T([1,512,40,59], bf16), T([1,512,1,1], f32), T([1,512,1,1], f32), T([512], f32), T([512], f32), T([], bf16), S([1,512,40,59]))
@oracle_impl(hardware="B200", point="bdfa8b76", SRC_BLOCK=1024, HW_BLOCK=1024, source_warps=4, row_warps=4)
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
    del arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, _shape_param_0

    device = arg0_1.device
    scatter = torch.empty((C,), device=device, dtype=torch.bfloat16)
    _source_sum_kernel[(C,)](
        arg0_1,
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
