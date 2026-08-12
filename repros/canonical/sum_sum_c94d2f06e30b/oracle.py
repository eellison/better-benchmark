"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete phlippe DenseNet bf16 BN-backward plus fixed 2x2 avg-pool-backward tail, sharing the masked bf16 `where` producer across the returned `sum(where)` and `sum(where * centered) * invstd` f32 channel vectors, then sinking those finalized channel summaries into the bf16 residual-add epilogue and writing the returned expanded pool-backward tensor directly, whereas Inductor schedules the sibling reductions, broadcast BN-backward epilogue, bf16 slice add, and structured pool-backward expansion as separate generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan whose per-channel summaries feed a layout-changing 2x2 pool-backward consumer while preserving bf16 cast boundaries; the fix is SCHEDULER_FUSION: teach reduction scheduling to co-schedule compatible channel reductions and fuse their dependent static pool-backward epilogue."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 88
INPUT_C = 104
H = 4
W = 4
OUT_H = 8
OUT_W = 8
HW = H * W
OUT_HW = OUT_H * OUT_W
K_TOTAL = N * HW
SLICE_START = 16
SCALE = 0.00048828125


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
def _channel_reduce_pool_kernel(
    residual_ptr,
    mask_ptr,
    fill_ptr,
    rhs_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    mul8_out_ptr,
    pool_out_ptr,
    C_N: tl.constexpr,
    INPUT_C_N: tl.constexpr,
    SLICE_START_N: tl.constexpr,
    W_N: tl.constexpr,
    OUT_W_N: tl.constexpr,
    HW_N: tl.constexpr,
    OUT_HW_N: tl.constexpr,
    K_TOTAL_N: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < K_TOTAL_N
    n = k // HW_N
    hw = k - n * HW_N
    h = hw // W_N
    w = hw - h * W_N

    compact_offsets = n * (C_N * HW_N) + c * HW_N + hw
    slice_offsets = n * (INPUT_C_N * HW_N) + (c + SLICE_START_N) * HW_N + hw

    mask_value = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    rhs = tl.load(rhs_ptr + compact_offsets, mask=active, other=0.0)
    selected = tl.where(mask_value <= 0.0, fill, rhs).to(tl.float32)

    activation = tl.load(activation_ptr + compact_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(activation, mean)
    dot_values = _f32_mul(selected, centered)

    sum_value = tl.sum(tl.where(active, selected, 0.0), axis=0)
    dot_value = tl.sum(tl.where(active, dot_values, 0.0), axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(mul8_out_ptr + c, _f32_mul(dot_value, invstd))

    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    dot_mean = _f32_mul(dot_value, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    after_variance = _f32_sub(selected, _f32_mul(centered, correction_scale))
    after_mean = _f32_sub(after_variance, mean_term)
    grad = _f32_mul(after_mean, output_scale)
    grad_bf16 = grad.to(tl.bfloat16, fp_downcast_rounding="rtne")

    residual = tl.load(residual_ptr + slice_offsets, mask=active, other=0.0)
    add_bf16 = _bf16_add(residual, grad_bf16)
    pool_bf16 = _f32_mul(add_bf16.to(tl.float32), 0.25).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    out_base = n * (C_N * OUT_HW_N) + c * OUT_HW_N + (h * 2) * OUT_W_N + w * 2
    tl.store(pool_out_ptr + out_base, pool_bf16, mask=active)
    tl.store(pool_out_ptr + out_base + 1, pool_bf16, mask=active)
    tl.store(pool_out_ptr + out_base + OUT_W_N, pool_bf16, mask=active)
    tl.store(pool_out_ptr + out_base + OUT_W_N + 1, pool_bf16, mask=active)


@oracle_impl(
    hardware="B200",
    point="9d14ecda",
    BLOCK_K=2048,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    (
        residual,
        mask,
        fill,
        rhs,
        activation,
        mean,
        invstd,
        weight,
        pool_input,
    ) = inputs

    sum_out = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    mul8_out = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    pool_out = torch.empty_strided(
        (N, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=pool_input.device,
        dtype=torch.bfloat16,
    )

    _channel_reduce_pool_kernel[(C,)](
        residual,
        mask,
        fill,
        rhs,
        activation,
        mean,
        invstd,
        weight,
        sum_out,
        mul8_out,
        pool_out,
        C_N=C,
        INPUT_C_N=INPUT_C,
        SLICE_START_N=SLICE_START,
        W_N=W,
        OUT_W_N=OUT_W,
        HW_N=HW,
        OUT_HW_N=OUT_HW,
        K_TOTAL_N=K_TOTAL,
        SCALE_VALUE=SCALE,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=4,
    )

    return sum_out, mul8_out, pool_out
