"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet add/copy/clone high-channel batchnorm-backward return tuple by writing both required bf16 add outputs, cooperatively reducing the high 40 channels across the batch/spatial domain, and reusing the finalized channel summaries for the f32 vector outputs and bf16 channels-last tensor epilogue, whereas Inductor materializes the add/copy/clone producer and then schedules the sibling channel reductions, finalizers, and dependent tensor epilogue as separate generic regions; Inductor cannot do this today because its scheduler/codegen has no full-scope cooperative split-K reduction template that shares a memory-format producer with compatible channel reductions and their dependent materializing epilogue; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible channel reductions over the reduced domain, combine partial summaries once, and fuse the downstream tensor/vector epilogues while preserving returned layout-copy outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_ALL = 80
C = 40
H = 14
W = 14
HW = H * W
SLICE_START = 40
NUMEL = N * C * HW
TOTAL_SPATIAL = N * HW
SCALE = 9.964923469387754e-06


@triton.jit
def _add_copy_kernel(
    arg0_ptr,
    arg1_ptr,
    clone_out_ptr,
    copy_out_ptr,
    C_ALL_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_ROWS_: tl.constexpr,
    X_BLOCK: tl.constexpr,
    Y_BLOCK: tl.constexpr,
):
    x = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[None, :]
    y = tl.program_id(1) * Y_BLOCK + tl.arange(0, Y_BLOCK)[:, None]
    mask = (x < HW_) & (y < TOTAL_ROWS_)

    c = y % C_ALL_
    n = y // C_ALL_
    contig_offset = y * HW_ + x
    cl_offset = n * C_ALL_ * HW_ + x * C_ALL_ + c

    x0 = tl.load(arg0_ptr + contig_offset, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(arg1_ptr + cl_offset, mask=mask, other=0.0).to(tl.float32)
    value = x0 + x1
    tl.store(clone_out_ptr + contig_offset, value, mask=mask)
    tl.store(copy_out_ptr + cl_offset, value, mask=mask)


@triton.jit
def _partial_reduce_kernel(
    copy_ptr,
    rhs_ptr,
    mean_ptr,
    partials_ptr,
    C_ALL_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    TOTAL_PARTIAL_: tl.constexpr,
    R_BLOCK: tl.constexpr,
    X_BLOCK: tl.constexpr,
):
    x_offsets = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)
    r_offsets = tl.arange(0, R_BLOCK)
    x_mask = x_offsets < TOTAL_PARTIAL_

    x = x_offsets[:, None]
    c = x % C_
    partial = x // C_
    spatial = partial * R_BLOCK + r_offsets[None, :]
    mask = x_mask[:, None] & (spatial < TOTAL_SPATIAL_)

    copy_offset = spatial * C_ALL_ + SLICE_START_ + c
    rhs_offset = spatial * C_ + c

    value = tl.load(copy_ptr + copy_offset, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + rhs_offset, mask=mask, other=0.0).to(tl.float32)
    centered = rhs - tl.load(mean_ptr + c, mask=x_mask[:, None], other=0.0).to(tl.float32)

    value = tl.where(mask, value, 0.0)
    product = tl.where(mask, value * centered, 0.0)
    sum0 = tl.sum(value, axis=1)
    sum1 = tl.sum(product, axis=1)

    tl.store(partials_ptr + x_offsets, sum0, mask=x_mask)
    tl.store(partials_ptr + TOTAL_PARTIAL_ + x_offsets, sum1, mask=x_mask)


@triton.jit
def _finalize_kernel(
    partials_ptr,
    stats_ptr,
    sum_out_ptr,
    scaled_sum_out_ptr,
    rsqrt_ptr,
    C_: tl.constexpr,
    NUM_BLOCKS_: tl.constexpr,
    TOTAL_PARTIAL_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    block = tl.arange(0, BLOCK)
    mask = block < NUM_BLOCKS_
    offsets = block * C_ + c

    sum0 = tl.sum(tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
    sum1 = tl.sum(
        tl.load(partials_ptr + TOTAL_PARTIAL_ + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    rsqrt = tl.load(rsqrt_ptr + c).to(tl.float32)
    tl.store(sum_out_ptr + c, sum0)
    tl.store(scaled_sum_out_ptr + c, sum1 * rsqrt)
    tl.store(stats_ptr + c, sum0)
    tl.store(stats_ptr + C_ + c, sum1)


@triton.jit
def _epilogue_kernel(
    copy_ptr,
    rhs_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    stats_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_ALL_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL_

    c = offsets % C_
    spatial = offsets // C_
    copy_offset = spatial * C_ALL_ + SLICE_START_ + c

    value = tl.load(copy_ptr + copy_offset, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum0 = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum1 = tl.load(stats_ptr + C_ + c, mask=mask, other=0.0).to(tl.float32)

    centered = rhs - mean
    centered_term = (sum1 * SCALE_) * (rsqrt * rsqrt)
    mean_term = sum0 * SCALE_
    out = ((value - centered * centered_term) - mean_term) * (rsqrt * weight)
    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="0c5208f9", COPY_X_BLOCK=256, COPY_Y_BLOCK=32, X_BLOCK=32, R_BLOCK=256, FINAL_BLOCK=512, EPILOGUE_BLOCK=1024)
def oracle_forward(inputs, *, COPY_X_BLOCK, COPY_Y_BLOCK, X_BLOCK, R_BLOCK, FINAL_BLOCK, EPILOGUE_BLOCK):
    arg0, arg1, rhs, mean, rsqrt, weight, _shape_param_0, _shape_param_1 = inputs
    clone_out = torch.empty_strided((N, C_ALL, H, W), (C_ALL * HW, HW, W, 1), device=arg0.device, dtype=torch.bfloat16)
    copy_out = torch.empty_strided((N, C_ALL, H, W), (C_ALL * HW, 1, W * C_ALL, C_ALL), device=arg0.device, dtype=torch.bfloat16)
    sum_out = torch.empty((C,), device=arg0.device, dtype=torch.float32)
    scaled_sum_out = torch.empty((C,), device=arg0.device, dtype=torch.float32)
    tensor_out = torch.empty_strided((N, C, H, W), (C * HW, 1, W * C, C), device=arg0.device, dtype=torch.bfloat16)
    num_blocks = triton.cdiv(TOTAL_SPATIAL, R_BLOCK)
    total_partial = C * num_blocks
    partials = torch.empty((2, total_partial), device=arg0.device, dtype=torch.float32)
    stats = torch.empty((2, C), device=arg0.device, dtype=torch.float32)

    _add_copy_kernel[(triton.cdiv(HW, COPY_X_BLOCK), triton.cdiv(N * C_ALL, COPY_Y_BLOCK))](
        arg0,
        arg1,
        clone_out,
        copy_out,
        C_ALL_=C_ALL,
        HW_=HW,
        TOTAL_ROWS_=N * C_ALL,
        X_BLOCK=COPY_X_BLOCK,
        Y_BLOCK=COPY_Y_BLOCK,
        num_warps=4,
    )
    _partial_reduce_kernel[(triton.cdiv(total_partial, X_BLOCK),)](
        copy_out,
        rhs,
        mean,
        partials,
        C_ALL_=C_ALL,
        C_=C,
        SLICE_START_=SLICE_START,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        TOTAL_PARTIAL_=total_partial,
        R_BLOCK=R_BLOCK,
        X_BLOCK=X_BLOCK,
        num_warps=8,
    )
    _finalize_kernel[(C,)](
        partials,
        stats,
        sum_out,
        scaled_sum_out,
        rsqrt,
        C_=C,
        NUM_BLOCKS_=num_blocks,
        TOTAL_PARTIAL_=total_partial,
        BLOCK=FINAL_BLOCK,
        num_warps=4,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK),)](
        copy_out,
        rhs,
        mean,
        rsqrt,
        weight,
        stats,
        tensor_out,
        NUMEL_=NUMEL,
        C_ALL_=C_ALL,
        C_=C,
        SLICE_START_=SLICE_START,
        SCALE_=SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return clone_out, copy_out, sum_out, scaled_sum_out, tensor_out
