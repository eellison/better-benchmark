"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DenseNet transition backward tail returned by Repro.forward, sharing the masked `where` producer across both f32 channel reductions, preserving the sequential sixteen-input bf16 slice-add chain, and writing the reduction-dependent BN epilogue directly through the fixed 2x2 avg_pool2d_backward expansion. Inductor currently schedules the sliced residual producer, sibling reductions, broadcast BN epilogue, residual add, and pool-backward expansion as separate generic regions with materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan whose finalized channel scalars feed a layout-changing pool-backward consumer while preserving bf16 cast boundaries. The fix is SCHEDULER_FUSION: add a guarded DenseNet BN-backward/pool-backward template that co-schedules the compatible channel reductions, static slice-add producer, and structured 2x2 pool epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 512
H = 7
W = 7
HW = H * W
OUT_H = 14
OUT_W = 14
OUT_HW = OUT_H * OUT_W
R = N * HW
SCALE = 0.00510204081632653


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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _densenet_pool_kernel(
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
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    pool_out_ptr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < 196
    n = rows // 49
    spatial = rows - n * 49
    base = n * (512 * 49) + c * 49 + spatial

    mask_value = tl.load(mask_ptr + base, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    source_value = tl.load(source_ptr + base, mask=active, other=0.0)
    where_bf16 = tl.where(mask_value <= 0.0, fill_value, source_value)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    centered_source = tl.load(
        centered_source_ptr + base, mask=active, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _f32_sub(centered_source, mean), 0.0)

    product = _f32_mul(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_centered = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_where, 0.00510204081632653)
    dot_scaled = _f32_mul(sum_centered, 0.00510204081632653)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    after_variance = _f32_sub(where_f32, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    bn_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    local = c * 49 + spatial
    v0 = tl.load(r0_ptr + n * (1024 * 49) + local, mask=active, other=0.0)
    v1 = tl.load(r1_ptr + n * (992 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(v0, v1)
    v2 = tl.load(r2_ptr + n * (960 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v2)
    v3 = tl.load(r3_ptr + n * (928 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v3)
    v4 = tl.load(r4_ptr + n * (896 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v4)
    v5 = tl.load(r5_ptr + n * (864 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v5)
    v6 = tl.load(r6_ptr + n * (832 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v6)
    v7 = tl.load(r7_ptr + n * (800 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v7)
    v8 = tl.load(r8_ptr + n * (768 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v8)
    v9 = tl.load(r9_ptr + n * (736 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v9)
    v10 = tl.load(r10_ptr + n * (704 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v10)
    v11 = tl.load(r11_ptr + n * (672 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v11)
    v12 = tl.load(r12_ptr + n * (640 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v12)
    v13 = tl.load(r13_ptr + n * (608 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v13)
    v14 = tl.load(r14_ptr + n * (576 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v14)
    v15 = tl.load(r15_ptr + n * (544 * 49) + local, mask=active, other=0.0)
    residual = _bf16_add(residual, v15)
    pool_parent = _bf16_add(residual, bn_bf16)
    pool_value = _f32_mul(pool_parent.to(tl.float32), 0.25).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    h = spatial // 7
    w = spatial - h * 7
    out_base = n * (512 * 196) + c * 196 + h * 28 + w * 2

    tl.store(sum_out_ptr + c, sum_where)
    tl.store(scale_grad_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(pool_out_ptr + out_base, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 1, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 14, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 15, pool_value, mask=active)


# densenet121 train, N=4 C=512 H=W=7, sixteen residual slices plus BN backward and 2x2 pool backward.
@oracle_impl(hardware="B200", point="6f10577d", BLOCK_R=256, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
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
        _arg23_1,
    ) = inputs
    device = arg0_1.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    pool_out = torch.empty_strided(
        (N, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _densenet_pool_kernel[(C,)](
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
        sum_out,
        scale_grad,
        pool_out,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )

    return sum_out, scale_grad, pool_out
