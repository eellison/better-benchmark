"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BatchNorm-backward tail, sharing the masked bf16 where producer across both f32 channel reductions, writing the full returned bf16 gradient tensor, and forming the returned last-32-channel residual add with the required sequential bf16 slice-add rounding across twenty-two residual inputs. Inductor currently schedules the long residual slice-add chain, sibling reductions, reduction-dependent BN epilogue, full gradient materialization, and final slice add as separate generic regions around materialized or replayed intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that keeps compatible channel summaries available to both dense and slice-limited consumers while preserving bf16/f32 cast boundaries. The fix is SCHEDULER_FUSION: add a guarded DenseNet BN-backward template that co-schedules the compatible channel reductions, static slice-add producer, dense store, and residual-slice epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 320
H = 14
W = 14
HW = H * W
K_TOTAL = N * HW
SLICE_START = 288
SLICE_C = 32
SCALE = 0.0012755102040816326
_USE_INDUCTOR_NUMERICS = False


@triton.jit
def _f32_add(a, b):
    return a + b


@triton.jit
def _f32_sub(a, b):
    return a - b


@triton.jit
def _f32_mul(a, b):
    return a * b


@triton.jit
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _bn_tail_kernel(
    r0_ptr,
    r1_ptr,
    r2_ptr,
    r3_ptr,
    r4_ptr,
    r5_ptr,
    r6_ptr,
    r7_ptr,
    r8_ptr,
    r9_ptr,
    r10_ptr,
    r11_ptr,
    r12_ptr,
    r13_ptr,
    r14_ptr,
    r15_ptr,
    r16_ptr,
    r17_ptr,
    r18_ptr,
    r19_ptr,
    r20_ptr,
    r21_ptr,
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    add_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < K_TOTAL_
    n = k // HW_
    spatial = k - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + spatial

    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0)
    where_bf16 = tl.where(mask_value <= 0.0, fill, source)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    centered_source = tl.load(
        centered_source_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _f32_sub(centered_source, mean), 0.0)

    product = _f32_mul(where_f32, centered)
    sum_value = tl.sum(where_f32, axis=0)
    dot_value = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    dot_scaled = _f32_mul(dot_value, SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)

    corrected = _f32_sub(where_f32, _f32_mul(centered, variance_term))
    mean_term = _f32_mul(sum_value, SCALE_)
    centered_grad = _f32_sub(corrected, mean_term)
    output_scale = _f32_mul(invstd, weight)
    dense_bf16 = _f32_mul(centered_grad, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scale_grad_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(dense_out_ptr + offsets, dense_bf16, mask=active)

    in_slice = c >= SLICE_START_
    slice_c = c - SLICE_START_
    add_mask = active & in_slice
    add_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + spatial

    v0 = tl.load(r0_ptr + n * (1024 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v1 = tl.load(r1_ptr + n * (992 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v2 = tl.load(r2_ptr + n * (960 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v3 = tl.load(r3_ptr + n * (928 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v4 = tl.load(r4_ptr + n * (896 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v5 = tl.load(r5_ptr + n * (864 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v6 = tl.load(r6_ptr + n * (832 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v7 = tl.load(r7_ptr + n * (800 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v8 = tl.load(r8_ptr + n * (768 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v9 = tl.load(r9_ptr + n * (736 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v10 = tl.load(r10_ptr + n * (704 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v11 = tl.load(r11_ptr + n * (672 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v12 = tl.load(r12_ptr + n * (640 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v13 = tl.load(r13_ptr + n * (608 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v14 = tl.load(r14_ptr + n * (576 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v15 = tl.load(r15_ptr + n * (544 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v16 = tl.load(r16_ptr + n * (512 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v17 = tl.load(r17_ptr + n * (480 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v18 = tl.load(r18_ptr + n * (448 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v19 = tl.load(r19_ptr + n * (416 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v20 = tl.load(r20_ptr + n * (384 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)
    v21 = tl.load(r21_ptr + n * (352 * HW_) + c * HW_ + spatial, mask=add_mask, other=0.0)

    if USE_INDUCTOR_NUMERICS:
        residual = _f32_add(v0.to(tl.float32), v1.to(tl.float32))
        residual = _f32_add(residual, v2.to(tl.float32))
        residual = _f32_add(residual, v3.to(tl.float32))
        residual = _f32_add(residual, v4.to(tl.float32))
        residual = _f32_add(residual, v5.to(tl.float32))
        residual = _f32_add(residual, v6.to(tl.float32))
        residual = _f32_add(residual, v7.to(tl.float32))
        residual = _f32_add(residual, v8.to(tl.float32))
        residual = _f32_add(residual, v9.to(tl.float32))
        residual = _f32_add(residual, v10.to(tl.float32))
        residual = _f32_add(residual, v11.to(tl.float32))
        residual = _f32_add(residual, v12.to(tl.float32))
        residual = _f32_add(residual, v13.to(tl.float32))
        residual = _f32_add(residual, v14.to(tl.float32))
        residual = _f32_add(residual, v15.to(tl.float32))
        residual = _f32_add(residual, v16.to(tl.float32))
        residual = _f32_add(residual, v17.to(tl.float32))
        residual = _f32_add(residual, v18.to(tl.float32))
        residual = _f32_add(residual, v19.to(tl.float32))
        residual = _f32_add(residual, v20.to(tl.float32))
        residual = _f32_add(residual, v21.to(tl.float32))
        add_value = _f32_add(residual, dense_bf16.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
    else:
        residual = _bf16_add(v0, v1)
        residual = _bf16_add(residual, v2)
        residual = _bf16_add(residual, v3)
        residual = _bf16_add(residual, v4)
        residual = _bf16_add(residual, v5)
        residual = _bf16_add(residual, v6)
        residual = _bf16_add(residual, v7)
        residual = _bf16_add(residual, v8)
        residual = _bf16_add(residual, v9)
        residual = _bf16_add(residual, v10)
        residual = _bf16_add(residual, v11)
        residual = _bf16_add(residual, v12)
        residual = _bf16_add(residual, v13)
        residual = _bf16_add(residual, v14)
        residual = _bf16_add(residual, v15)
        residual = _bf16_add(residual, v16)
        residual = _bf16_add(residual, v17)
        residual = _bf16_add(residual, v18)
        residual = _bf16_add(residual, v19)
        residual = _bf16_add(residual, v20)
        residual = _bf16_add(residual, v21)
        add_value = _bf16_add(residual, dense_bf16)

    tl.store(add_out_ptr + add_offsets, add_value, mask=add_mask)


# densenet121 train, N=4 C=320 H=W=14, residual slice channels 288:320.
@oracle_impl(hardware="B200", point="66cb4e6d", BLOCK_K=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    global _USE_INDUCTOR_NUMERICS
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
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        arg24_1,
        arg25_1,
        arg26_1,
        arg27_1,
        arg28_1,
    ) = inputs
    device = arg24_1.device
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W),
        (SLICE_C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _bn_tail_kernel[(C,)](
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
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        arg17_1,
        arg18_1,
        arg19_1,
        arg20_1,
        arg21_1,
        arg22_1,
        arg23_1,
        arg24_1,
        arg25_1,
        arg26_1,
        arg27_1,
        arg28_1,
        sum_out,
        scale_grad,
        dense_out,
        add_out,
        C_=C,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        SCALE_=SCALE,
        BLOCK_K=BLOCK_K,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
        num_stages=3,
    )
    return sum_out, scale_grad, dense_out, add_out
