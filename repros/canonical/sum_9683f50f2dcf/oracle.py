"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 bf16 residual add plus live channel-sum scope by materializing the returned add tensor and accumulating the sum partials in the same Triton pass, then applying the captured bf16 reduction boundary before the fp32 cast, whereas Inductor currently materializes the add and schedules the channel reduction as a separate generic reduction over that producer; Inductor cannot do this today because its scheduler does not form a multi-output pointwise-plus-reduction plan when the pointwise producer is also a returned output with a non-default dense stride; the fix is SCHEDULER_FUSION: teach the scheduler to fuse returned pointwise materialization with compatible reductions while preserving output strides and dtype cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_sum_partials_cinner_kernel(
    x_ptr,
    y_ptr,
    add_ptr,
    partials_ptr,
    y_stride_n: tl.constexpr,
    y_stride_c: tl.constexpr,
    y_stride_h: tl.constexpr,
    y_stride_w: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    k_offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)

    hw = H * W
    n = k_offsets // hw
    spatial = k_offsets - n * hw
    h = spatial % H
    w = spatial // H

    mask = (k_offsets[:, None] < N * hw) & (c_offsets[None, :] < C)
    x_offsets = k_offsets[:, None] * C + c_offsets[None, :]
    y_offsets = (
        n[:, None] * y_stride_n
        + c_offsets[None, :] * y_stride_c
        + h[:, None] * y_stride_h
        + w[:, None] * y_stride_w
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0).to(tl.float32)
    add = (x + y).to(tl.bfloat16)
    tl.store(add_ptr + x_offsets, add, mask=mask)

    partial = tl.sum(tl.where(mask, add.to(tl.float32), 0.0), axis=0)
    tl.store(
        partials_ptr + tl.program_id(0) * C + c_offsets,
        partial,
        mask=c_offsets < C,
    )


@triton.jit
def _add_sum_partials_kernel(
    x_ptr,
    y_ptr,
    add_ptr,
    partials_ptr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    y_stride_n: tl.constexpr,
    y_stride_c: tl.constexpr,
    y_stride_h: tl.constexpr,
    y_stride_w: tl.constexpr,
    out_stride_n: tl.constexpr,
    out_stride_c: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_w: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    SPATIAL_H_INNER: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    k_offsets = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)

    hw = H * W
    n = k_offsets // hw
    spatial = k_offsets - n * hw
    if SPATIAL_H_INNER:
        h = spatial % H
        w = spatial // H
    else:
        h = spatial // W
        w = spatial - h * W

    mask = (k_offsets[:, None] < N * hw) & (c_offsets[None, :] < C)
    x_offsets = (
        n[:, None] * x_stride_n
        + c_offsets[None, :] * x_stride_c
        + h[:, None] * x_stride_h
        + w[:, None] * x_stride_w
    )
    y_offsets = (
        n[:, None] * y_stride_n
        + c_offsets[None, :] * y_stride_c
        + h[:, None] * y_stride_h
        + w[:, None] * y_stride_w
    )
    out_offsets = (
        n[:, None] * out_stride_n
        + c_offsets[None, :] * out_stride_c
        + h[:, None] * out_stride_h
        + w[:, None] * out_stride_w
    )

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0).to(tl.float32)
    add = (x + y).to(tl.bfloat16)
    tl.store(add_ptr + out_offsets, add, mask=mask)

    partial = tl.sum(tl.where(mask, add.to(tl.float32), 0.0), axis=0)
    tl.store(
        partials_ptr + tl.program_id(0) * C + c_offsets,
        partial,
        mask=c_offsets < C,
    )


@triton.jit
def _finish_sum_kernel(
    partials_ptr,
    out_ptr,
    C: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial_offsets = tl.arange(0, BLOCK_PARTIALS)
    mask = (partial_offsets[:, None] < NUM_PARTIALS) & (c_offsets[None, :] < C)
    values = tl.load(
        partials_ptr + partial_offsets[:, None] * C + c_offsets[None, :],
        mask=mask,
        other=0.0,
    )
    total = tl.sum(values.to(tl.float32), axis=0)
    tl.store(
        out_ptr + c_offsets,
        total.to(tl.bfloat16).to(tl.float32),
        mask=c_offsets < C,
    )


def _launch(inputs, *, BLOCK_K, BLOCK_C, SPATIAL_H_INNER, num_warps):
    arg0_1, arg1_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    num_partials = triton.cdiv(n * h * w, BLOCK_K)

    add_out = torch.empty_strided(
        (n, c, h, w),
        tuple(int(s) for s in arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty((num_partials, c), device=arg0_1.device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)

    grid = (num_partials, triton.cdiv(c, BLOCK_C))
    if SPATIAL_H_INNER:
        _add_sum_partials_cinner_kernel[grid](
            arg0_1,
            arg1_1,
            add_out,
            partials,
            y_stride_n=arg1_1.stride(0),
            y_stride_c=arg1_1.stride(1),
            y_stride_h=arg1_1.stride(2),
            y_stride_w=arg1_1.stride(3),
            N=n,
            C=c,
            H=h,
            W=w,
            BLOCK_K=BLOCK_K,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=3,
        )
    else:
        _add_sum_partials_kernel[grid](
            arg0_1,
            arg1_1,
            add_out,
            partials,
            x_stride_n=arg0_1.stride(0),
            x_stride_c=arg0_1.stride(1),
            x_stride_h=arg0_1.stride(2),
            x_stride_w=arg0_1.stride(3),
            y_stride_n=arg1_1.stride(0),
            y_stride_c=arg1_1.stride(1),
            y_stride_h=arg1_1.stride(2),
            y_stride_w=arg1_1.stride(3),
            out_stride_n=add_out.stride(0),
            out_stride_c=add_out.stride(1),
            out_stride_h=add_out.stride(2),
            out_stride_w=add_out.stride(3),
            N=n,
            C=c,
            H=h,
            W=w,
            NUM_PARTIALS=num_partials,
            SPATIAL_H_INNER=SPATIAL_H_INNER,
            BLOCK_K=BLOCK_K,
            BLOCK_C=BLOCK_C,
            num_warps=num_warps,
            num_stages=3,
        )
    _finish_sum_kernel[(triton.cdiv(c, 16),)](
        partials,
        sum_out,
        C=c,
        NUM_PARTIALS=num_partials,
        BLOCK_PARTIALS=triton.next_power_of_2(num_partials),
        BLOCK_C=16,
        num_warps=8,
        num_stages=3,
    )
    return add_out, sum_out


# 201d3861: (T([128,160,28,28], bf16, stride=(125440,1,160,4480)), T([128,160,28,28], bf16, stride=(125440,1,4480,160)))
@oracle_impl(hardware="B200", point="201d3861", BLOCK_K=512, BLOCK_C=64, SPATIAL_H_INNER=True, num_warps=16)
# b04a3c92: (T([128,320,14,14], bf16, stride=(62720,1,320,4480)), T([128,320,14,14], bf16, stride=(62720,1,4480,320)))
@oracle_impl(hardware="B200", point="b04a3c92", BLOCK_K=128, BLOCK_C=16, SPATIAL_H_INNER=True, num_warps=8)
# 9ea47abe: (T([128,640,7,7], bf16), T([128,640,7,7], bf16, stride=(31360,1,4480,640)))
@oracle_impl(hardware="B200", point="9ea47abe", BLOCK_K=128, BLOCK_C=16, SPATIAL_H_INNER=False, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K, BLOCK_C, SPATIAL_H_INNER, num_warps):
    return _launch(
        inputs,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        SPATIAL_H_INNER=SPATIAL_H_INNER,
        num_warps=num_warps,
    )
