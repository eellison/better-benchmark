"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete PyTorch U-Net bf16 bilinear-upsample-backward scatter feeding BN/ReLU masking and three channel reductions, using the captured real bilinear index tensors with bounds `[0,40)` and `[0,59)` for the four duplicate-preserving `index_put(accumulate=True)` updates, then preserving the f32 add order, bf16 scatter/mask/BN-backward cast boundaries, returned `[1,512,40,59]` tensor, and final bf16-rounded channel sum; Inductor currently lowers the four generic scatter-adds, bf16 masking producer, dependent reductions, and final bf16 output reduction as separate materialized kernels; Inductor cannot do this today because scheduler/codegen does not recognize indexed bilinear scatter-add producers as channel-reduction inputs with a required full-tensor epilogue; the fix is SCATTER_REDUCE: teach Inductor to specialize bounded scatter index domains, reduce indexed scatter producers directly, and fuse the downstream channel reductions plus returned bf16 store."""

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
SCATTER_PLANES = 4


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
def _zero_kernel(ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=offsets < n_elements)


@triton.jit
def _scatter_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    scatter_ptr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    offs = tl.arange(0, BLOCK_K)
    k = block * BLOCK_K + offs
    mask = k < 9440
    h = k // 118
    w = k - h * 118

    src_offsets = (c + 512) * 9520 + h * 119 + w
    x = tl.load(arg0_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    h_weight = tl.load(arg1_ptr + h, mask=mask, other=0.0).to(tl.float32)
    w_weight = tl.load(arg2_ptr + w, mask=mask, other=0.0).to(tl.float32)

    mul = _f32_mul(x, h_weight)
    add = _f32_sub(x, mul)
    mul_1 = _f32_mul(mul, w_weight)
    add_1 = _f32_sub(mul, mul_1)
    mul_2 = _f32_mul(add, w_weight)
    add_2 = _f32_sub(add, mul_2)

    h0 = tl.load(arg3_ptr + h, mask=mask, other=0)
    w0 = tl.load(arg4_ptr + w, mask=mask, other=0)
    w1 = tl.load(arg5_ptr + w, mask=mask, other=0)
    h1 = tl.load(arg6_ptr + h, mask=mask, other=0)
    channel_base = c * 2360
    plane_stride = 512 * 2360

    tl.atomic_add(
        scatter_ptr + channel_base + h0 * 59 + w0,
        mul_1,
        sem="relaxed",
        mask=mask,
    )
    tl.atomic_add(
        scatter_ptr + plane_stride + channel_base + h0 * 59 + w1,
        add_1,
        sem="relaxed",
        mask=mask,
    )
    tl.atomic_add(
        scatter_ptr + plane_stride * 2 + channel_base + h1 * 59 + w0,
        mul_2,
        sem="relaxed",
        mask=mask,
    )
    tl.atomic_add(
        scatter_ptr + plane_stride * 3 + channel_base + h1 * 59 + w1,
        add_2,
        sem="relaxed",
        mask=mask,
    )


@triton.jit
def _combine_scatter_kernel(
    scatter_ptr,
    out_ptr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.program_id(1)
    hw = block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = hw < 2360
    offsets = c * 2360 + hw
    plane_stride = 512 * 2360
    s0 = tl.load(scatter_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(scatter_ptr + plane_stride + offsets, mask=mask, other=0.0).to(tl.float32)
    s2 = tl.load(scatter_ptr + plane_stride * 2 + offsets, mask=mask, other=0.0).to(tl.float32)
    s3 = tl.load(scatter_ptr + plane_stride * 3 + offsets, mask=mask, other=0.0).to(tl.float32)
    combined = _f32_add(_f32_add(_f32_add(s0, s1), s2), s3)
    tl.store(out_ptr + offsets, combined.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@triton.jit
def _bn_backward_kernel(
    arg7_ptr,
    arg8_ptr,
    arg9_ptr,
    arg10_ptr,
    arg11_ptr,
    arg12_ptr,
    out_ptr,
    out_sum1_ptr,
    out_mul13_ptr,
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
        scatter = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.where(active, scalar, scatter)

        acc1 += tl.where(mask, y, 0.0)
        acc2 += tl.where(mask, _f32_mul(y, centered), 0.0)

    sum1 = tl.sum(acc1, axis=0)
    sum2 = tl.sum(acc2, axis=0)
    tl.store(out_sum1_ptr + c, sum1)
    tl.store(out_mul13_ptr + c, _f32_mul(sum2, invstd))

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
        scatter = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.where(active, scalar, scatter)

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
@oracle_impl(hardware="B200", point="bdfa8b76", SRC_BLOCK=512, ZERO_BLOCK=1024, HW_BLOCK=1024, source_warps=4, row_warps=4)
def oracle_forward(
    inputs,
    *,
    SRC_BLOCK: int,
    ZERO_BLOCK: int,
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
    del _shape_param_0

    device = arg0_1.device
    scatter = torch.empty((SCATTER_PLANES, C, OUT_HW), device=device, dtype=torch.float32)
    _zero_kernel[(triton.cdiv(SCATTER_PLANES * C * OUT_HW, ZERO_BLOCK),)](
        scatter,
        n_elements=SCATTER_PLANES * C * OUT_HW,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _scatter_kernel[(C, triton.cdiv(SRC_K, SRC_BLOCK))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        scatter,
        BLOCK_K=SRC_BLOCK,
        num_warps=source_warps,
        num_stages=3,
    )

    out_tensor = torch.empty_strided(
        (1, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    _combine_scatter_kernel[(C, triton.cdiv(OUT_HW, HW_BLOCK))](
        scatter,
        out_tensor,
        BLOCK_HW=HW_BLOCK,
        num_warps=row_warps,
        num_stages=3,
    )

    out_sum1 = torch.empty((C,), device=device, dtype=torch.float32)
    out_mul13 = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum4 = torch.empty((C,), device=device, dtype=torch.float32)
    _bn_backward_kernel[(C,)](
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        out_tensor,
        out_sum1,
        out_mul13,
        out_sum4,
        BLOCK_HW=HW_BLOCK,
        num_warps=row_warps,
        num_stages=3,
    )

    return out_sum1, out_mul13, out_tensor, out_sum4
