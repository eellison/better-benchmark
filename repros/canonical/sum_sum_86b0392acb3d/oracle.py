"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete LearningToPaint bf16 avg-pool-backward BN-backward fragment returned by Repro.forward, including the scalar bf16 zero, the materialized masked bf16 `where` tensor, both per-channel f32 reductions, and the reduction-dependent bf16 dense epilogue. Inductor currently schedules the structured avg_pool2d_backward producer, mask/where materialization, sibling channel reductions, and broadcast BN-backward epilogue as generic regions around materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that shares the bf16-rounded pool producer across visible side outputs, compatible reductions, and the dependent dense consumer while preserving explicit bf16/f32 cast boundaries. The fix is SCHEDULER_FUSION: add a guarded training-BN backward template that keeps the structured pool producer and masked bf16 output live while finalizing both channel reductions once and sinking the dense epilogue into the same fused plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 96
C = 512
H = 4
W = 4
HW = H * W
R = N * HW
POOL_SCALE = 0.0625
REDUCE_SCALE = 0.0006510416666666666


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
def _pool_bn_backward_kernel(
    grad_pool_ptr,
    mask_source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    full_out_ptr,
    where_out_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    dense_out_ptr,
    C_SIZE: tl.constexpr,
    H_SIZE: tl.constexpr,
    W_SIZE: tl.constexpr,
    HW_SIZE: tl.constexpr,
    R_SIZE: tl.constexpr,
    POOL_SCALE_VALUE: tl.constexpr,
    REDUCE_SCALE_VALUE: tl.constexpr,
    G_SN: tl.constexpr,
    G_SC: tl.constexpr,
    M_SN: tl.constexpr,
    M_SC: tl.constexpr,
    M_SH: tl.constexpr,
    M_SW: tl.constexpr,
    X_SN: tl.constexpr,
    X_SC: tl.constexpr,
    X_SH: tl.constexpr,
    X_SW: tl.constexpr,
    O_SN: tl.constexpr,
    O_SC: tl.constexpr,
    O_SH: tl.constexpr,
    O_SW: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    c = tl.program_id(0)
    rows = tl.arange(0, BLOCK_R)
    active = rows < R_SIZE
    n = rows // HW_SIZE
    spatial = rows - n * HW_SIZE
    h = spatial // W_SIZE
    w = spatial - h * W_SIZE

    grad = tl.load(
        grad_pool_ptr + n * G_SN + c * G_SC,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    pooled_bf16 = _f32_mul(grad, POOL_SCALE_VALUE).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    mask_offsets = n * M_SN + c * M_SC + h * M_SH + w * M_SW
    mask_value = tl.load(mask_source_ptr + mask_offsets, mask=active, other=0.0)
    zero_scalar = tl.full((), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    zero_bf16 = tl.full((BLOCK_R,), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    where_bf16 = tl.where(mask_value <= 0.0, zero_bf16, pooled_bf16)
    where_f32 = tl.where(active, where_bf16.to(tl.float32), 0.0)

    out_offsets = n * O_SN + c * O_SC + h * O_SH + w * O_SW
    tl.store(where_out_ptr + out_offsets, where_bf16, mask=active)

    source_offsets = n * X_SN + c * X_SC + h * X_SH + w * X_SW
    source = tl.load(centered_source_ptr + source_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = tl.where(active, _f32_sub(source, mean), 0.0)

    product = _f32_mul(where_f32, centered)
    sum_where = tl.sum(where_f32, axis=0)
    sum_centered = tl.sum(product, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_where, REDUCE_SCALE_VALUE)
    dot_scaled = _f32_mul(sum_centered, REDUCE_SCALE_VALUE)
    invstd_sq = _f32_mul(invstd, invstd)
    variance_term = _f32_mul(dot_scaled, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    after_variance = _f32_sub(where_f32, _f32_mul(centered, variance_term))
    after_mean = _f32_sub(after_variance, mean_term)
    dense_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    tl.store(full_out_ptr, zero_scalar, mask=c == 0)
    tl.store(sum_out_ptr + c, sum_where)
    tl.store(scale_grad_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(dense_out_ptr + out_offsets, dense_bf16, mask=active)


# (T([96,512], bf16), T([96,512,4,4], bf16), T([96,512,4,4], bf16, stride=(8192,1,2048,512)), T([1,512,1,1], f32), T([512], f32), T([512], f32), S([96,512,1,1]))
@oracle_impl(hardware="B200", point="ab4c6849", BLOCK_R=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0 = inputs
    device = arg0_1.device

    full_out = torch.empty_strided((), (), device=device, dtype=torch.bfloat16)
    where_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    grad_stride = tuple(int(s) for s in arg0_1.stride())
    mask_stride = tuple(int(s) for s in arg1_1.stride())
    source_stride = tuple(int(s) for s in arg2_1.stride())
    out_stride = tuple(int(s) for s in where_out.stride())

    _pool_bn_backward_kernel[(C,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        full_out,
        where_out,
        sum_out,
        scale_grad,
        dense_out,
        C_SIZE=C,
        H_SIZE=H,
        W_SIZE=W,
        HW_SIZE=HW,
        R_SIZE=R,
        POOL_SCALE_VALUE=POOL_SCALE,
        REDUCE_SCALE_VALUE=REDUCE_SCALE,
        G_SN=grad_stride[0],
        G_SC=grad_stride[1],
        M_SN=mask_stride[0],
        M_SC=mask_stride[1],
        M_SH=mask_stride[2],
        M_SW=mask_stride[3],
        X_SN=source_stride[0],
        X_SC=source_stride[1],
        X_SH=source_stride[2],
        X_SW=source_stride[3],
        O_SN=out_stride[0],
        O_SC=out_stride[1],
        O_SH=out_stride[2],
        O_SW=out_stride[3],
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
        num_stages=3,
    )

    return full_out, where_out, sum_out, scale_grad, dense_out
