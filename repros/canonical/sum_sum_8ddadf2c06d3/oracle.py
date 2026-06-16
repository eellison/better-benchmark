"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 BN-backward tail by sharing the masked `where` producer across both returned per-channel f32 reductions, immediately reusing the finalized channel summaries in the bf16 normalization epilogue, and returning the last-32-channel slice as an alias of the full dense output. Inductor currently schedules the sibling reductions, broadcast-dependent epilogue, full tensor materialization, and aliasing slice return as generic regions around materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that keeps compatible channel reductions and their dependent dense/slice consumers together while preserving bf16/f32 cast boundaries. The fix is SCHEDULER_FUSION: add a guarded DenseNet BN-backward reduction/epilogue template that shares the masked producer, finalizes both channel vectors once, and sinks the full bf16 tensor plus aliasing slice output into the same fused plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 1024
H = 14
W = 14
HW = H * W
R = N * HW
SCALE = 0.0012755102040816326
SLICE_START = 992


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
def _dense_bn_backward_kernel(
    mask_input_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    C_SIZE: tl.constexpr,
    HW_SIZE: tl.constexpr,
    R_SIZE: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < R_SIZE
    n = rows // HW_SIZE
    spatial = rows - n * HW_SIZE
    offsets = n * (C_SIZE * HW_SIZE) + c * HW_SIZE + spatial

    mask_value = tl.load(mask_input_ptr + offsets, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    source_value = tl.load(source_ptr + offsets, mask=active, other=0.0)
    where_bf16 = tl.where(mask_value <= 0.0, fill_value, source_value)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    centered_source = tl.load(
        centered_source_ptr + offsets, mask=active, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _f32_sub(centered_source, mean), 0.0)

    product = _f32_mul(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_centered = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_where, SCALE_VALUE)
    dot_scaled = _f32_mul(sum_centered, SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    after_variance = _f32_sub(where_f32, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    dense_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(sum_out_ptr + c, sum_where)
    tl.store(scale_grad_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(dense_out_ptr + offsets, dense_bf16, mask=active)


# (T([4,1024,14,14], bf16), T([], bf16), T([4,1024,14,14], bf16), T([4,1024,14,14], bf16), T([1,1024,1,1], f32), T([1024], f32), T([1024], f32))
@oracle_impl(hardware="B200", point="470762bc", BLOCK_R=1024, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    device = arg0_1.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _dense_bn_backward_kernel[(C,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        sum_out,
        scale_grad,
        dense_out,
        C_SIZE=C,
        HW_SIZE=HW,
        R_SIZE=R,
        SCALE_VALUE=SCALE,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )

    return sum_out, scale_grad, dense_out, dense_out[:, SLICE_START:C, :, :]
