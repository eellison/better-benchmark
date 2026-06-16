"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete ResNeXt bf16 const-center max-pool-backward scatter, BN/ReLU gate, both returned f32 channel reductions, and the returned bf16 BN-backward tensor by treating the low-memory offsets as a collision-free 2x upsample and fusing the gated values with the reduction and epilogue consumers, whereas Inductor materializes the f32 zero/scatter_add buffer, casts it to bf16, runs the gate, rereads the gated tensor for sibling reductions, and then schedules the dependent BN-backward epilogue separately; Inductor cannot do this today because scheduler/codegen treats `_low_memory_max_pool_offsets_to_indices` plus `scatter_add` as an opaque producer instead of a fixed-window maxpool-backward scatter-reduce with live reduction and side-output consumers; the fix is SCATTER_REDUCE: add a guarded low-memory maxpool-backward lowering that recognizes constant-center offsets, fuses the bf16 gate/reductions, and emits the dependent full-gradient output from the same structured producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _partials_kernel(
    grad0_ptr,
    grad1_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    GRAD0_STRIDE_N: tl.constexpr,
    GRAD0_STRIDE_C: tl.constexpr,
    GRAD0_STRIDE_H: tl.constexpr,
    GRAD0_STRIDE_W: tl.constexpr,
    GRAD1_STRIDE_N: tl.constexpr,
    GRAD1_STRIDE_C: tl.constexpr,
    GRAD1_STRIDE_H: tl.constexpr,
    GRAD1_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    GROUP_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    c_block = tl.program_id(1)
    rows_base = tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C

    mean = tl.load(mean_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.range(0, GROUP_R, BLOCK_R):
        rows = group * GROUP_R + start + rows_base
        row_mask = rows < R
        n = rows // (OUT_H * OUT_W)
        rem = rows - n * (OUT_H * OUT_W)
        oh = rem // OUT_W
        ow = rem - oh * OUT_W
        active = row_mask[:, None] & col_mask[None, :]

        x_offsets = (
            n[:, None] * X_STRIDE_N
            + cols[None, :] * X_STRIDE_C
            + oh[:, None] * X_STRIDE_H
            + ow[:, None] * X_STRIDE_W
        )
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :]
        affine = affine * weight[None, :] + bias[None, :]
        affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
        take_scalar = affine_bf16 <= 0.0

        src_h = oh // 2
        src_w = ow // 2
        has_source = active & ((oh % 2) == 0)[:, None] & ((ow % 2) == 0)[:, None]
        src_offsets0 = (
            n[:, None] * GRAD0_STRIDE_N
            + cols[None, :] * GRAD0_STRIDE_C
            + src_h[:, None] * GRAD0_STRIDE_H
            + src_w[:, None] * GRAD0_STRIDE_W
        )
        src_offsets1 = (
            n[:, None] * GRAD1_STRIDE_N
            + cols[None, :] * GRAD1_STRIDE_C
            + src_h[:, None] * GRAD1_STRIDE_H
            + src_w[:, None] * GRAD1_STRIDE_W
        )
        src0 = tl.load(grad0_ptr + src_offsets0, mask=has_source, other=0.0)
        src1 = tl.load(grad1_ptr + src_offsets1, mask=has_source, other=0.0)
        scatter_bf16 = (src0 + src1).to(tl.bfloat16, fp_downcast_rounding="rtne")
        selected = tl.where(take_scalar, scalar, scatter_bf16).to(tl.float32)
        selected = tl.where(active, selected, 0.0)
        centered = tl.where(active, centered, 0.0)
        acc1 += tl.sum(selected, axis=0)
        acc2 += tl.sum(selected * centered, axis=0)

    partial_offsets = group * C + cols
    tl.store(partial_sum1_ptr + partial_offsets, acc1, mask=col_mask)
    tl.store(partial_sum2_ptr + partial_offsets, acc2, mask=col_mask)


@triton.jit
def _final_sums_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    invstd_ptr,
    sum1_ptr,
    sum2_ptr,
    mul10_ptr,
    NUM_GROUPS: tl.constexpr,
    C: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    groups = tl.arange(0, BLOCK_GROUPS)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < C)
    offsets = groups[:, None] * C + cols[None, :]
    partial1 = tl.load(
        partial_sum1_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    partial2 = tl.load(
        partial_sum2_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    sum1 = tl.sum(partial1, axis=0)
    sum2 = tl.sum(partial2, axis=0)
    col_mask = cols < C
    invstd = tl.load(invstd_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    tl.store(sum1_ptr + cols, sum1, mask=col_mask)
    tl.store(sum2_ptr + cols, sum2, mask=col_mask)
    tl.store(mul10_ptr + cols, sum2 * invstd, mask=col_mask)


@triton.jit
def _output_kernel(
    grad0_ptr,
    grad1_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    GRAD0_STRIDE_N: tl.constexpr,
    GRAD0_STRIDE_C: tl.constexpr,
    GRAD0_STRIDE_H: tl.constexpr,
    GRAD0_STRIDE_W: tl.constexpr,
    GRAD1_STRIDE_N: tl.constexpr,
    GRAD1_STRIDE_C: tl.constexpr,
    GRAD1_STRIDE_H: tl.constexpr,
    GRAD1_STRIDE_W: tl.constexpr,
    X_STRIDE_N: tl.constexpr,
    X_STRIDE_C: tl.constexpr,
    X_STRIDE_H: tl.constexpr,
    X_STRIDE_W: tl.constexpr,
    OUT_STRIDE_N: tl.constexpr,
    OUT_STRIDE_C: tl.constexpr,
    OUT_STRIDE_H: tl.constexpr,
    OUT_STRIDE_W: tl.constexpr,
    REDUCTION_SCALE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r_block = tl.program_id(0)
    c_block = tl.program_id(1)
    rows = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < R
    col_mask = cols < C
    active = row_mask[:, None] & col_mask[None, :]

    n = rows // (OUT_H * OUT_W)
    rem = rows - n * (OUT_H * OUT_W)
    oh = rem // OUT_W
    ow = rem - oh * OUT_W
    x_offsets = (
        n[:, None] * X_STRIDE_N
        + cols[None, :] * X_STRIDE_C
        + oh[:, None] * X_STRIDE_H
        + ow[:, None] * X_STRIDE_W
    )
    x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr)

    centered = x_val - mean[None, :]
    affine = centered * invstd[None, :]
    affine = affine * weight[None, :] + bias[None, :]
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    take_scalar = affine_bf16 <= 0.0

    src_h = oh // 2
    src_w = ow // 2
    has_source = active & ((oh % 2) == 0)[:, None] & ((ow % 2) == 0)[:, None]
    src_offsets0 = (
        n[:, None] * GRAD0_STRIDE_N
        + cols[None, :] * GRAD0_STRIDE_C
        + src_h[:, None] * GRAD0_STRIDE_H
        + src_w[:, None] * GRAD0_STRIDE_W
    )
    src_offsets1 = (
        n[:, None] * GRAD1_STRIDE_N
        + cols[None, :] * GRAD1_STRIDE_C
        + src_h[:, None] * GRAD1_STRIDE_H
        + src_w[:, None] * GRAD1_STRIDE_W
    )
    src0 = tl.load(grad0_ptr + src_offsets0, mask=has_source, other=0.0)
    src1 = tl.load(grad1_ptr + src_offsets1, mask=has_source, other=0.0)
    scatter_bf16 = (src0 + src1).to(tl.bfloat16, fp_downcast_rounding="rtne")
    selected = tl.where(take_scalar, scalar, scatter_bf16).to(tl.float32)

    sum1_scaled = sum1 * REDUCTION_SCALE
    sum2_scaled = (sum2 * REDUCTION_SCALE) * (invstd * invstd)
    channel_weight = invstd * weight
    out = selected - centered * sum2_scaled[None, :]
    out = out - sum1_scaled[None, :]
    out = out * channel_weight[None, :]

    out_offsets = (
        n[:, None] * OUT_STRIDE_N
        + cols[None, :] * OUT_STRIDE_C
        + oh[:, None] * OUT_STRIDE_H
        + ow[:, None] * OUT_STRIDE_W
    )
    tl.store(
        out_ptr + out_offsets,
        out.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _launch(
    inputs,
    *,
    GROUP_R: int,
    REDUCE_BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    OUT_BLOCK_R: int,
    num_warps_reduce: int,
    num_warps_output: int,
):
    grad0, grad1, _offsets, x, mean, invstd, weight, bias, scalar, *_shape_params = inputs
    n = int(x.shape[0])
    c = int(x.shape[1])
    out_h = int(x.shape[2])
    out_w = int(x.shape[3])
    r = n * out_h * out_w
    reduction_scale = 1.0 / float(r)
    num_groups = triton.cdiv(r, GROUP_R)

    partial_sum1 = torch.empty((num_groups, c), device=x.device, dtype=torch.float32)
    partial_sum2 = torch.empty((num_groups, c), device=x.device, dtype=torch.float32)
    sum1 = torch.empty((c,), device=x.device, dtype=torch.float32)
    sum2 = torch.empty((c,), device=x.device, dtype=torch.float32)
    mul10 = torch.empty((c,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(int(s) for s in x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )

    _partials_kernel[(num_groups, triton.cdiv(c, BLOCK_C))](
        grad0,
        grad1,
        x,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        partial_sum1,
        partial_sum2,
        R=r,
        C=c,
        OUT_H=out_h,
        OUT_W=out_w,
        GRAD0_STRIDE_N=int(grad0.stride(0)),
        GRAD0_STRIDE_C=int(grad0.stride(1)),
        GRAD0_STRIDE_H=int(grad0.stride(2)),
        GRAD0_STRIDE_W=int(grad0.stride(3)),
        GRAD1_STRIDE_N=int(grad1.stride(0)),
        GRAD1_STRIDE_C=int(grad1.stride(1)),
        GRAD1_STRIDE_H=int(grad1.stride(2)),
        GRAD1_STRIDE_W=int(grad1.stride(3)),
        X_STRIDE_N=int(x.stride(0)),
        X_STRIDE_C=int(x.stride(1)),
        X_STRIDE_H=int(x.stride(2)),
        X_STRIDE_W=int(x.stride(3)),
        GROUP_R=GROUP_R,
        BLOCK_R=REDUCE_BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_reduce,
        num_stages=3,
    )
    _final_sums_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        partial_sum1,
        partial_sum2,
        invstd,
        sum1,
        sum2,
        mul10,
        NUM_GROUPS=num_groups,
        C=c,
        BLOCK_GROUPS=_next_power_of_2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _output_kernel[(triton.cdiv(r, OUT_BLOCK_R), triton.cdiv(c, BLOCK_C))](
        grad0,
        grad1,
        x,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        sum1,
        sum2,
        out,
        R=r,
        C=c,
        OUT_H=out_h,
        OUT_W=out_w,
        GRAD0_STRIDE_N=int(grad0.stride(0)),
        GRAD0_STRIDE_C=int(grad0.stride(1)),
        GRAD0_STRIDE_H=int(grad0.stride(2)),
        GRAD0_STRIDE_W=int(grad0.stride(3)),
        GRAD1_STRIDE_N=int(grad1.stride(0)),
        GRAD1_STRIDE_C=int(grad1.stride(1)),
        GRAD1_STRIDE_H=int(grad1.stride(2)),
        GRAD1_STRIDE_W=int(grad1.stride(3)),
        X_STRIDE_N=int(x.stride(0)),
        X_STRIDE_C=int(x.stride(1)),
        X_STRIDE_H=int(x.stride(2)),
        X_STRIDE_W=int(x.stride(3)),
        OUT_STRIDE_N=int(out.stride(0)),
        OUT_STRIDE_C=int(out.stride(1)),
        OUT_STRIDE_H=int(out.stride(2)),
        OUT_STRIDE_W=int(out.stride(3)),
        REDUCTION_SCALE=reduction_scale,
        BLOCK_R=OUT_BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_output,
        num_stages=3,
    )
    return sum1, mul10, out


# c3354c52: torchbench resnext50_32x4d train, bf16 const-center maxpool scatter feeding BN/ReLU reductions.
@oracle_impl(hardware="B200", point="c3354c52", GROUP_R=1024, REDUCE_BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=16, OUT_BLOCK_R=128, num_warps_reduce=8, num_warps_output=8)
def oracle_forward(
    inputs,
    *,
    GROUP_R: int,
    REDUCE_BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    OUT_BLOCK_R: int,
    num_warps_reduce: int,
    num_warps_output: int,
):
    return _launch(
        inputs,
        GROUP_R=GROUP_R,
        REDUCE_BLOCK_R=REDUCE_BLOCK_R,
        BLOCK_C=BLOCK_C,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        OUT_BLOCK_R=OUT_BLOCK_R,
        num_warps_reduce=num_warps_reduce,
        num_warps_output=num_warps_output,
    )
