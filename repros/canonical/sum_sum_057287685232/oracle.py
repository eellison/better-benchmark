"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DenseNet transition-backward tail returned by Repro.forward, sharing the masked bf16 `where` producer across both f32 channel reductions, preserving the sliced residual add, and writing the reduction-dependent BN epilogue directly through the fixed 2x2 avg_pool2d_backward expansion, whereas Inductor currently schedules the mask, sibling reductions, broadcast BN epilogue, residual add, and pool-backward expansion as separate generic regions with materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan whose finalized channel scalars feed a layout-changing pool-backward consumer while preserving bf16 cast boundaries; the fix is SCHEDULER_FUSION: add a guarded DenseNet BN-backward/pool-backward template that co-schedules compatible channel reductions, static slice-add producers, and structured 2x2 pool epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 80
H = 8
W = 8
HW = H * W
OUT_H = 16
OUT_W = 16
OUT_HW = OUT_H * OUT_W
R = N * HW
SCALE = 0.0001220703125


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
    residual_ptr,
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
    active = rows < 8192
    n = rows // 64
    spatial = rows - n * 64
    base = n * (80 * 64) + c * 64 + spatial

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
    mean_term = _f32_mul(sum_where, 0.0001220703125)
    dot_scaled = _f32_mul(sum_centered, 0.0001220703125)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    after_variance = _f32_sub(where_f32, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    bn_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    residual_offset = n * (96 * 64) + (c + 16) * 64 + spatial
    residual = tl.load(residual_ptr + residual_offset, mask=active, other=0.0)
    pool_parent = _bf16_add(residual, bn_bf16)
    pool_value = _f32_mul(pool_parent.to(tl.float32), 0.25).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    h = spatial // 8
    w = spatial - h * 8
    out_base = n * (80 * 256) + c * 256 + h * 32 + w * 2

    tl.store(sum_out_ptr + c, sum_where)
    tl.store(scale_grad_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(pool_out_ptr + out_base, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 1, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 16, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 17, pool_value, mask=active)


# torchbench_phlippe_densenet_train, one residual slice plus BN backward and 2x2 pool backward.
@oracle_impl(
    hardware="B200",
    point="529a48b9",
    BLOCK_R=8192,
    num_warps=4,
)
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
        _arg8_1,
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
        sum_out,
        scale_grad,
        pool_out,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )

    return sum_out, scale_grad, pool_out
