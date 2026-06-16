"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Demucs bf16 zero-padded slice_scatter, materialized masked where tensor, and compiled-path f32 channel sum from one source-space traversal plus a small finalizer, whereas Inductor schedules the zero-fill/slice_scatter output, the scalar where materialization, and the sibling reduction as separate generic regions that reread the same source and strided mask. Inductor cannot do this today because its scheduler/codegen has no structured scatter-reduce template that keeps a materialized zero-padded slice_scatter side output and an unpadded masked reduction/where epilogue in one fused producer while preserving the returned bf16 materializations and compiled reduction accuracy. The fix is SCATTER_REDUCE: add a structured slice_scatter lowering that emits padded side-output stores and accumulates sibling source-space reductions directly from the scattered tile."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 256
IN_T = 5804
PAD_LEFT = 87
PAD_RIGHT = 88
OUT_T = IN_T + PAD_LEFT + PAD_RIGHT
MASK_STRIDE_B = 1507328
MASK_STRIDE_C = 5888
COPY_REDUCE_BLOCK_T = 4096
TILES_T = triton.cdiv(IN_T, COPY_REDUCE_BLOCK_T)
PARTIALS_PER_CHANNEL = BATCH * TILES_T
FINAL_BLOCK = 8
PAD_BLOCK = 128


@triton.jit
def _zero_pad_kernel(
    out_ptr,
    pad_left: tl.constexpr,
    pad_right: tl.constexpr,
    out_t: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_PAD)
    zero = tl.full((BLOCK_PAD,), 0.0, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    base = row * out_t
    tl.store(out_ptr + base + offsets, zero, mask=offsets < pad_left)
    tl.store(
        out_ptr + base + (out_t - pad_right) + offsets,
        zero,
        mask=offsets < pad_right,
    )


@triton.jit
def _copy_where_reduce_kernel(
    src_ptr,
    mask_ptr,
    scalar_ptr,
    padded_ptr,
    where_ptr,
    partial_ptr,
    in_t: tl.constexpr,
    out_t: tl.constexpr,
    channels: tl.constexpr,
    pad_left: tl.constexpr,
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

    src_offsets = (batch * channels + channel) * in_t + offsets_t
    mask_offsets = batch * mask_stride_b + channel * mask_stride_c + offsets_t
    padded_offsets = (batch * channels + channel) * out_t + pad_left + offsets_t

    src_vals = tl.load(src_ptr + src_offsets, mask=active, other=0.0)
    mask_vals = tl.load(mask_ptr + mask_offsets, mask=active, other=0)
    scalar = tl.load(scalar_ptr)
    where_vals = tl.where(mask_vals, scalar, src_vals)

    tl.store(padded_ptr + padded_offsets, src_vals, mask=active)
    tl.store(where_ptr + src_offsets, where_vals, mask=active)

    partial = tl.sum(tl.where(active, where_vals.to(tl.float32), 0.0), axis=0)
    partial_offset = channel * (4 * tiles_t) + batch * tiles_t + tile
    tl.store(partial_ptr + partial_offset, partial)


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


# torchbench_demucs train, bf16 [4,256,5804] with strided bool mask.
@oracle_impl(hardware="B200", point="1bed4207", BLOCK_T=COPY_REDUCE_BLOCK_T, num_warps=8)
def oracle_forward(inputs, *, BLOCK_T: int, num_warps: int):
    src, mask, scalar, shape_param = inputs

    slice_scatter = torch.empty_strided(
        tuple(int(dim) for dim in shape_param),
        (CHANNELS * OUT_T, OUT_T, 1),
        device=src.device,
        dtype=torch.bfloat16,
    )
    where = torch.empty_strided(
        (BATCH, CHANNELS, IN_T),
        (CHANNELS * IN_T, IN_T, 1),
        device=src.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided(
        (CHANNELS, PARTIALS_PER_CHANNEL),
        (PARTIALS_PER_CHANNEL, 1),
        device=src.device,
        dtype=torch.float32,
    )
    reduced = torch.empty_strided((CHANNELS,), (1,), device=src.device, dtype=torch.float32)

    _zero_pad_kernel[(BATCH * CHANNELS,)](
        slice_scatter,
        PAD_LEFT,
        PAD_RIGHT,
        OUT_T,
        BLOCK_PAD=PAD_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _copy_where_reduce_kernel[(CHANNELS, BATCH, TILES_T)](
        src,
        mask,
        scalar,
        slice_scatter,
        where,
        partial,
        IN_T,
        OUT_T,
        CHANNELS,
        PAD_LEFT,
        MASK_STRIDE_B,
        MASK_STRIDE_C,
        TILES_T,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
        num_stages=3,
    )
    _final_sum_kernel[(CHANNELS,)](
        partial,
        reduced,
        PARTIALS_PER_CHANNEL,
        BLOCK_PARTIALS=FINAL_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return slice_scatter, where, reduced
