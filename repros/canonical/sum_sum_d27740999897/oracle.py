"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DenseNet BN-backward tail by sharing the masked `where` producer across the two returned per-channel f32 reductions, reusing the finalized channel summaries in the full bf16 normalization epilogue, and returning the last-32-channel slice as an alias of the full gradient output; Inductor currently schedules the sibling reductions, broadcast-dependent epilogue, full tensor store, and slice return as generic regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that keeps compatible channel reductions, exact bf16/f32 cast boundaries, and dependent dense/slice outputs together; the fix is SCHEDULER_FUSION: add a guarded DenseNet BN-backward reduction/epilogue template that shares the masked producer, finalizes both channel vectors once, and sinks the full tensor plus aliasing slice output into the same fused plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 512
H = 28
W = 28
HW = H * W
R = N * HW
SCALE = 0.00031887755102040814
SLICE_START = 480


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
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < 3136
    n = rows // 784
    spatial = rows - n * 784
    offsets = n * (512 * 784) + c * 784 + spatial

    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0).to(tl.bfloat16)
    fill_value = tl.load(fill_ptr)
    source_value = tl.load(source_ptr + offsets, mask=active, other=0.0)
    where_bf16 = tl.where(mask_value <= 0.0, fill_value, source_value)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _f32_sub(centered_source, mean), 0.0)

    product = _f32_mul(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_centered = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_where, 0.00031887755102040814)
    dot_scaled = _f32_mul(sum_centered, 0.00031887755102040814)
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


# (T([4,512,28,28], bf16), T([], bf16), T([4,512,28,28], bf16), T([4,512,28,28], bf16), T([1,512,1,1], f32), T([512], f32), T([512], f32))
@oracle_impl(hardware="B200", point="8cf827b9", BLOCK_R=4096, num_warps=8)
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
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )

    return sum_out, scale_grad, dense_out, dense_out[:, SLICE_START:C, :, :]
