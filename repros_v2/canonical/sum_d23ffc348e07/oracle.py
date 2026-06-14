"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Demucs bf16 zero-padded slice_scatter, returned bf16 scalar zero, materialized masked where tensor, and compiled-path f32 channel sum from one source-space traversal plus a small finalizer, whereas Inductor schedules the zero-fill/slice_scatter output, scalar full, where materialization, and sibling reduction as separate generic regions that reread the same source and strided mask; Inductor cannot do this today because its scheduler/codegen has no structured scatter-reduce template that keeps a materialized zero-padded slice_scatter side output and an unpadded masked reduction/where epilogue in one fused producer while preserving the returned bf16 materializations; the fix is SCATTER_REDUCE: add a structured slice_scatter lowering that emits padded side-output stores and accumulates sibling source-space reductions directly from the scattered tile."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 64
IN_T = 92844
PAD = 1426
OUT_T = 95696
MASK_STRIDE_B = 5947392
MASK_STRIDE_C = 92928
COPY_REDUCE_BLOCK_T = 4096
TILES_T = triton.cdiv(IN_T, COPY_REDUCE_BLOCK_T)
PARTIALS_PER_CHANNEL = BATCH * TILES_T
FINAL_BLOCK = 128
PAD_BLOCK = 2048


@triton.jit
def _zero_pad_kernel(
    out_ptr,
    pad: tl.constexpr,
    out_t: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_PAD)
    active = offsets < pad
    base = row * out_t
    zero = tl.full((BLOCK_PAD,), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    tl.store(out_ptr + base + offsets, zero, mask=active)
    tl.store(out_ptr + base + (out_t - pad) + offsets, zero, mask=active)


@triton.jit
def _zero_scalar_kernel(scalar_ptr):
    zero = tl.full((), 0.0, tl.float32).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(scalar_ptr, zero)


@triton.jit
def _copy_where_reduce_kernel(
    src_ptr,
    mask_ptr,
    padded_ptr,
    where_ptr,
    partial_ptr,
    in_t: tl.constexpr,
    out_t: tl.constexpr,
    channels: tl.constexpr,
    pad: tl.constexpr,
    mask_stride_b: tl.constexpr,
    mask_stride_c: tl.constexpr,
    tiles_t: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    channel = tl.program_id(0)
    batch = tl.program_id(1)
    tile = tl.program_id(2)
    offsets_t = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    active = offsets_t < in_t

    src_base = (batch * channels + channel) * in_t
    src_offsets = src_base + offsets_t
    mask_offsets = batch * mask_stride_b + channel * mask_stride_c + offsets_t
    padded_offsets = (batch * channels + channel) * out_t + pad + offsets_t

    src_vals = tl.load(src_ptr + src_offsets, mask=active, other=0.0)
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=active, other=1)
    zero = tl.full((BLOCK_T,), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    where_vals = tl.where(mask_vals, zero, src_vals)

    tl.store(padded_ptr + padded_offsets, src_vals, mask=active)
    tl.store(where_ptr + src_offsets, where_vals, mask=active)

    reduced = tl.sum(
        tl.where(active, where_vals.to(tl.float32), 0.0),
        axis=0,
    )
    partial_offset = channel * (4 * tiles_t) + batch * tiles_t + tile
    tl.store(partial_ptr + partial_offset, reduced)


@triton.jit
def _final_sum_kernel(
    partial_ptr,
    out_ptr,
    partials_per_channel: tl.constexpr,
    BLOCK_PARTIALS: tl.constexpr,
):
    channel = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_PARTIALS)
    active = offsets < partials_per_channel
    values = tl.load(
        partial_ptr + channel * partials_per_channel + offsets,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(values, axis=0)
    tl.store(out_ptr + channel, total)


# 9e7b7398: torchbench demucs train, bf16 [4,64,92844] with strided bool mask.
@oracle_impl(hardware="B200", point="9e7b7398", BLOCK_T=COPY_REDUCE_BLOCK_T, num_warps=8)
def oracle_forward(inputs, *, BLOCK_T: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0 = inputs
    del _shape_param_0

    device = arg0_1.device
    slice_scatter = torch.empty_strided(
        (BATCH, CHANNELS, OUT_T),
        (CHANNELS * OUT_T, OUT_T, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    full_1 = torch.empty((), device=device, dtype=torch.bfloat16)
    where = torch.empty_strided(
        (BATCH, CHANNELS, IN_T),
        (CHANNELS * IN_T, IN_T, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty(
        (CHANNELS, PARTIALS_PER_CHANNEL),
        device=device,
        dtype=torch.float32,
    )
    convert_element_type = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _zero_pad_kernel[(BATCH * CHANNELS,)](
        slice_scatter,
        PAD,
        OUT_T,
        BLOCK_PAD=PAD_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _zero_scalar_kernel[(1,)](full_1, num_warps=1, num_stages=3)
    _copy_where_reduce_kernel[(CHANNELS, BATCH, TILES_T)](
        arg0_1,
        arg1_1,
        slice_scatter,
        where,
        partial,
        IN_T,
        OUT_T,
        CHANNELS,
        PAD,
        MASK_STRIDE_B,
        MASK_STRIDE_C,
        TILES_T,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
        num_stages=3,
    )
    _final_sum_kernel[(CHANNELS,)](
        partial,
        convert_element_type,
        PARTIALS_PER_CHANNEL,
        BLOCK_PARTIALS=FINAL_BLOCK,
        num_warps=4,
        num_stages=3,
    )

    return slice_scatter, full_1, where, convert_element_type
