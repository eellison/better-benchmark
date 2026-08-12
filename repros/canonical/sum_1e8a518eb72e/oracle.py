"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete demucs bf16 zero-padded `slice_scatter` plus masked channel-reduction scope, including the returned `[4,128,23923]` padded side output, the returned contiguous `[4,128,23212]` `where(mask, scalar, arg0)` materialization from the padded-stride bool mask, and the `[128]` f32 channel-sum output. Inductor currently schedules the `full -> slice_scatter` side output, the masked `where` side output, and the sibling `sum([0,2])` reduction as separate generic producers/consumers over the same source-space tile; it cannot do this today because scheduler/codegen does not model a zero-fill slice-scatter side output plus a source-space masked materialization and reduction epilogue as one structured scatter-reduce plan. The fix is SCATTER_REDUCE: add a structured `slice_scatter` lowering that emits the padded side-output stores while accumulating sibling masked reductions directly from the scattered source tile."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 128
W = 23212
PAD_LEFT = 355
PADDED_W = 23923
MASK_C_STRIDE = 23296
MASK_N_STRIDE = C * MASK_C_STRIDE
PADDED_SHAPE = (N, C, PADDED_W)
SOURCE_SHAPE = (N, C, W)


@triton.jit
def _scatter_where_partial_kernel(
    src_ptr,
    mask_ptr,
    scalar_ptr,
    padded_out_ptr,
    where_out_ptr,
    partial_ptr,
    N_N: tl.constexpr,
    C_N: tl.constexpr,
    W_N: tl.constexpr,
    PAD_LEFT_N: tl.constexpr,
    PADDED_W_N: tl.constexpr,
    MASK_N_STRIDE_N: tl.constexpr,
    MASK_C_STRIDE_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    N_TILES: tl.constexpr,
):
    channel_block = tl.program_id(0)
    tile = tl.program_id(1)

    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    linear = tile * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_linear = linear < (N_N * PADDED_W_N)
    batch = linear // PADDED_W_N
    padded_w = linear - batch * PADDED_W_N
    in_source = (
        valid_linear
        & (padded_w >= PAD_LEFT_N)
        & (padded_w < (PAD_LEFT_N + W_N))
    )
    source_w = padded_w - PAD_LEFT_N

    c = channels[:, None]
    b = batch[None, :]
    sw = source_w[None, :]
    pw = padded_w[None, :]
    channel_valid = channels[:, None] < C_N
    valid = channel_valid & valid_linear[None, :]
    source_mask = channel_valid & in_source[None, :]

    source_offsets = b * (C_N * W_N) + c * W_N + sw
    padded_offsets = b * (C_N * PADDED_W_N) + c * PADDED_W_N + pw
    mask_offsets = b * MASK_N_STRIDE_N + c * MASK_C_STRIDE_N + sw

    src = tl.load(src_ptr + source_offsets, mask=source_mask, other=0.0)
    scalar = tl.load(scalar_ptr)
    keep_scalar = tl.load(mask_ptr + mask_offsets, mask=source_mask, other=0)
    where_value = tl.where(keep_scalar, scalar, src)
    padded_value = tl.where(source_mask, src, 0.0)

    tl.store(padded_out_ptr + padded_offsets, padded_value, mask=valid)
    tl.store(where_out_ptr + source_offsets, where_value, mask=source_mask)

    reduce_value = tl.where(source_mask, where_value.to(tl.float32), 0.0)
    partial = tl.sum(reduce_value, axis=1)
    tl.store(
        partial_ptr + channels * N_TILES + tile,
        partial,
        mask=channels < C_N,
    )


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    sum_out_ptr,
    C_N: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FINAL_BLOCK: tl.constexpr,
):
    channel_block = tl.program_id(0)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, FINAL_BLOCK)
    mask = (channels[:, None] < C_N) & (tiles[None, :] < N_TILES)
    values = tl.load(
        partial_ptr + channels[:, None] * N_TILES + tiles[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    total = tl.sum(values, axis=1)
    tl.store(sum_out_ptr + channels, total, mask=channels < C_N)


@oracle_impl(
    hardware="B200",
    point="aa121e37",
    BLOCK_C=4,
    BLOCK_M=1024,
    FINAL_BLOCK=128,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_C: int,
    BLOCK_M: int,
    FINAL_BLOCK: int,
    num_warps: int,
):
    src, mask, scalar, shape0 = inputs
    del shape0

    padded = torch.empty_strided(
        PADDED_SHAPE,
        (C * PADDED_W, PADDED_W, 1),
        device=src.device,
        dtype=torch.bfloat16,
    )
    where = torch.empty_strided(
        SOURCE_SHAPE,
        (C * W, W, 1),
        device=src.device,
        dtype=torch.bfloat16,
    )
    n_tiles = triton.cdiv(N * PADDED_W, BLOCK_M)
    partial = torch.empty_strided(
        (C, n_tiles),
        (n_tiles, 1),
        device=src.device,
        dtype=torch.float32,
    )
    sums = torch.empty_strided((C,), (1,), device=src.device, dtype=torch.float32)

    grid = (triton.cdiv(C, BLOCK_C), n_tiles)
    _scatter_where_partial_kernel[grid](
        src,
        mask,
        scalar,
        padded,
        where,
        partial,
        N_N=N,
        C_N=C,
        W_N=W,
        PAD_LEFT_N=PAD_LEFT,
        PADDED_W_N=PADDED_W,
        MASK_N_STRIDE_N=MASK_N_STRIDE,
        MASK_C_STRIDE_N=MASK_C_STRIDE,
        BLOCK_C=BLOCK_C,
        BLOCK_M=BLOCK_M,
        N_TILES=n_tiles,
        num_warps=num_warps,
        num_stages=4,
    )
    _finalize_sum_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial,
        sums,
        C_N=C,
        N_TILES=n_tiles,
        BLOCK_C=BLOCK_C,
        FINAL_BLOCK=FINAL_BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return padded, where, sums
