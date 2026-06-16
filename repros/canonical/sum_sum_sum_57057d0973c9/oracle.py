"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete BEiT relative-position scatter-reduce return tuple by reducing all twelve bf16 `[128,12,197,197]` sources over batch, applying the captured f32-sum-to-bf16-to-f32 boundary, folding the zero-fill/slice_scatter/negative-pad crop into the direct `[197,197,12]` stream, and atomically accumulating duplicate indices into twelve independent `[732,12]` outputs, whereas Inductor lowers each branch's batch reduction, bf16 slice/pad materialization, permute/view, and `index_put(accumulate=True)` as separate generic regions; Inductor cannot do this today because its scheduler/codegen does not model this zero-fill slice/pad chain as a structured scatter-reduce producer and cannot fuse the repeated relative-position index accumulation across branches; the fix is SCATTER_REDUCE: add a relative-position scatter-reduce lowering that reduces source tiles, preserves the bf16 rounding boundary, and accumulates duplicate buckets directly for every branch."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
CHANNELS = 12
HEIGHT = 197
WIDTH = 197
SPATIAL = HEIGHT * WIDTH
BUCKETS = 732


@triton.jit
def _zero_outputs_kernel(
    out0,
    out1,
    out2,
    out3,
    out4,
    out5,
    out6,
    out7,
    out8,
    out9,
    out10,
    out11,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    zero = tl.zeros((BLOCK,), dtype=tl.float32)
    tl.store(out0 + offsets, zero, mask=mask)
    tl.store(out1 + offsets, zero, mask=mask)
    tl.store(out2 + offsets, zero, mask=mask)
    tl.store(out3 + offsets, zero, mask=mask)
    tl.store(out4 + offsets, zero, mask=mask)
    tl.store(out5 + offsets, zero, mask=mask)
    tl.store(out6 + offsets, zero, mask=mask)
    tl.store(out7 + offsets, zero, mask=mask)
    tl.store(out8 + offsets, zero, mask=mask)
    tl.store(out9 + offsets, zero, mask=mask)
    tl.store(out10 + offsets, zero, mask=mask)
    tl.store(out11 + offsets, zero, mask=mask)


@triton.jit
def _scatter_one(
    x_ptr,
    index_ptr,
    out_ptr,
    channel,
    offsets_s,
    h,
    w,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    index_stride_h: tl.constexpr,
    index_stride_w: tl.constexpr,
    n_items: tl.constexpr,
    channels: tl.constexpr,
    spatial: tl.constexpr,
    buckets: tl.constexpr,
    block_n: tl.constexpr,
    block_s: tl.constexpr,
):
    partial = tl.zeros((block_s,), tl.float32)
    for n_base in tl.range(0, n_items, block_n):
        n_offsets = n_base + tl.arange(0, block_n)
        x_offsets = (
            n_offsets[:, None] * x_stride_n
            + channel * x_stride_c
            + h[None, :] * x_stride_h
            + w[None, :] * x_stride_w
        )
        load_mask = (n_offsets[:, None] < n_items) & (offsets_s[None, :] < spatial)
        values = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0).to(tl.float32)
        partial += tl.sum(values, axis=0)

    rounded = partial.to(tl.bfloat16).to(tl.float32)
    bucket = tl.load(
        index_ptr + h * index_stride_h + w * index_stride_w,
        mask=offsets_s < spatial,
        other=0,
    ).to(tl.int64)
    store_mask = (offsets_s < spatial) & (bucket >= 0) & (bucket < buckets)
    tl.atomic_add(
        out_ptr + bucket * channels + channel,
        rounded,
        sem="relaxed",
        mask=store_mask,
    )


@triton.jit
def _scatter_all12_kernel(
    x0,
    index0,
    x1,
    index1,
    x2,
    index2,
    x3,
    index3,
    x4,
    index4,
    x5,
    index5,
    x6,
    index6,
    x7,
    index7,
    x8,
    index8,
    x9,
    index9,
    x10,
    index10,
    x11,
    index11,
    out0,
    out1,
    out2,
    out3,
    out4,
    out5,
    out6,
    out7,
    out8,
    out9,
    out10,
    out11,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    index_stride_h: tl.constexpr,
    index_stride_w: tl.constexpr,
    n_items: tl.constexpr,
    channels: tl.constexpr,
    width: tl.constexpr,
    spatial: tl.constexpr,
    buckets: tl.constexpr,
    block_n: tl.constexpr,
    block_s: tl.constexpr,
):
    channel = tl.program_id(0)
    spatial_block = tl.program_id(1)
    offsets_s = spatial_block * block_s + tl.arange(0, block_s)
    h = offsets_s // width
    w = offsets_s - h * width

    _scatter_one(x0, index0, out0, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x1, index1, out1, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x2, index2, out2, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x3, index3, out3, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x4, index4, out4, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x5, index5, out5, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x6, index6, out6, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x7, index7, out7, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x8, index8, out8, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x9, index9, out9, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x10, index10, out10, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)
    _scatter_one(x11, index11, out11, channel, offsets_s, h, w, x_stride_n, x_stride_c, x_stride_h, x_stride_w, index_stride_h, index_stride_w, n_items, channels, spatial, buckets, block_n, block_s)


@oracle_impl(
    hardware="B200",
    point="019a4c87",
    BLOCK_N=4,
    BLOCK_S=256,
    ZERO_BLOCK=1024,
    scatter_warps=4,
    zero_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_N: int,
    BLOCK_S: int,
    ZERO_BLOCK: int,
    scatter_warps: int,
    zero_warps: int,
):
    (
        x0,
        index0,
        x1,
        index1,
        x2,
        index2,
        x3,
        index3,
        x4,
        index4,
        x5,
        index5,
        x6,
        index6,
        x7,
        index7,
        x8,
        index8,
        x9,
        index9,
        x10,
        index10,
        x11,
        index11,
        *_shape_params,
    ) = inputs
    del _shape_params

    device = x0.device
    outs = tuple(
        torch.empty_strided((BUCKETS, CHANNELS), (CHANNELS, 1), device=device, dtype=torch.float32)
        for _ in range(12)
    )
    total = BUCKETS * CHANNELS
    _zero_outputs_kernel[(triton.cdiv(total, ZERO_BLOCK),)](
        *outs,
        TOTAL=total,
        BLOCK=ZERO_BLOCK,
        num_warps=zero_warps,
    )

    _scatter_all12_kernel[(CHANNELS, triton.cdiv(SPATIAL, BLOCK_S))](
        x0,
        index0,
        x1,
        index1,
        x2,
        index2,
        x3,
        index3,
        x4,
        index4,
        x5,
        index5,
        x6,
        index6,
        x7,
        index7,
        x8,
        index8,
        x9,
        index9,
        x10,
        index10,
        x11,
        index11,
        *outs,
        x_stride_n=x0.stride(0),
        x_stride_c=x0.stride(1),
        x_stride_h=x0.stride(2),
        x_stride_w=x0.stride(3),
        index_stride_h=index0.stride(0),
        index_stride_w=index0.stride(1),
        n_items=N,
        channels=CHANNELS,
        width=WIDTH,
        spatial=SPATIAL,
        buckets=BUCKETS,
        block_n=BLOCK_N,
        block_s=BLOCK_S,
        num_warps=scatter_warps,
    )
    return outs
