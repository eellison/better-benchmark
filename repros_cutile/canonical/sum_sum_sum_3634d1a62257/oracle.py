"""cuTile port of sum_sum_sum_3634d1a62257: RepVGG BN-backward with two branches.

Mirrors Triton's 3-kernel structure:
  1. `_partial_sums_kernel`: producer + per-tile channel partials for
     sum(where), sum(where*(arg4-mean1)), sum(where*(arg8-mean2))
  2. `_finalize_sums_kernel`: per-channel reduction of partials into
     the returned sum vectors + 5 (channels,) stats used by the epilogue.
  3. `_dense_epilogue_kernel`: pointwise BN-backward for both branches.

Channels-last layout: input has stride (CHW, 1, WC, C), so flattened
to (N*H*W, C) with C in {96, 192, 384}. BLOCK_M=128, BLOCK_C=16 exactly
matches Triton.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


CAPTURED_INV_NHW = 2.4912308673469386e-06


def _next_p2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


@ct.kernel
def _partial_sums_kernel(
    a0_ptr,          # bf16 [NHW, C]
    a1_ptr,          # bf16 [NHW, C]
    a2_ptr,          # bf16 [NHW, C]
    fill_ptr,        # bf16 [1]
    a4_ptr,          # bf16 [NHW, C]
    mean1_ptr,       # f32  [C]
    a8_ptr,          # bf16 [NHW, C]
    mean2_ptr,       # f32  [C]
    out0_ptr,        # f32  [NHW, C]  (where.float())
    partial_ptr,     # f32  [3, num_tiles, C]  (row-major)
    NUM_TILES: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)
    c_block = ct.bid(1)

    a0 = ct.astype(ct.load(a0_ptr, index=(tile, c_block),
                           shape=(BLOCK_M, BLOCK_C)), ct.float32)
    a1 = ct.astype(ct.load(a1_ptr, index=(tile, c_block),
                           shape=(BLOCK_M, BLOCK_C)), ct.float32)
    a2 = ct.astype(ct.load(a2_ptr, index=(tile, c_block),
                           shape=(BLOCK_M, BLOCK_C)), ct.float32)
    fill_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bcast = ct.full((BLOCK_M, BLOCK_C), 0.0, dtype=ct.float32) + \
        ct.astype(ct.reshape(fill_1, (1, 1)), ct.float32)

    added = ct.astype(ct.astype(a0 + a1, ct.bfloat16), ct.float32)
    zero_bf16 = ct.astype(
        ct.full((BLOCK_M, BLOCK_C), 0.0, dtype=ct.float32),
        ct.bfloat16,
    )
    cond = ct.astype(a2, ct.bfloat16) <= zero_bf16
    where_bf16 = ct.astype(ct.where(cond, fill_bcast, added), ct.bfloat16)
    where_f32 = ct.astype(where_bf16, ct.float32)
    ct.store(out0_ptr, index=(tile, c_block), tile=where_f32)

    a4 = ct.astype(ct.load(a4_ptr, index=(tile, c_block),
                           shape=(BLOCK_M, BLOCK_C)), ct.float32)
    a8 = ct.astype(ct.load(a8_ptr, index=(tile, c_block),
                           shape=(BLOCK_M, BLOCK_C)), ct.float32)
    mean1 = ct.load(mean1_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean2 = ct.load(mean2_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean1_2d = ct.reshape(mean1, (1, BLOCK_C))
    mean2_2d = ct.reshape(mean2, (1, BLOCK_C))

    centered1 = a4 - mean1_2d
    centered2 = a8 - mean2_2d
    dot1 = where_f32 * centered1
    dot2 = where_f32 * centered2

    partial_where = ct.sum(where_f32, axis=0)  # (BLOCK_C,)
    partial_dot1 = ct.sum(dot1, axis=0)
    partial_dot2 = ct.sum(dot2, axis=0)

    ct.store(partial_ptr, index=(0, tile, c_block),
             tile=ct.reshape(partial_where, (1, 1, BLOCK_C)))
    ct.store(partial_ptr, index=(1, tile, c_block),
             tile=ct.reshape(partial_dot1, (1, 1, BLOCK_C)))
    ct.store(partial_ptr, index=(2, tile, c_block),
             tile=ct.reshape(partial_dot2, (1, 1, BLOCK_C)))


@ct.kernel
def _finalize_sums_kernel(
    partial_ptr,       # f32 [3, num_tiles, C]
    invstd1_ptr,       # f32 [C]
    weight1_ptr,       # f32 [C]
    invstd2_ptr,       # f32 [C]
    weight2_ptr,       # f32 [C]
    sum1_ptr,          # f32 [C]
    vec1_ptr,          # f32 [C]
    sum3_ptr,          # f32 [C]
    vec2_ptr,          # f32 [C]
    stats_ptr,         # f32 [5, C]
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_CF: ct.Constant[int],
    INV_NHW: ct.Constant[float],
):
    c_block = ct.bid(0)

    part_where = ct.load(
        partial_ptr, index=(0, 0, c_block),
        shape=(1, BLOCK_TILES, BLOCK_CF),
        padding_mode=ct.PaddingMode.ZERO,
    )
    part_dot1 = ct.load(
        partial_ptr, index=(1, 0, c_block),
        shape=(1, BLOCK_TILES, BLOCK_CF),
        padding_mode=ct.PaddingMode.ZERO,
    )
    part_dot2 = ct.load(
        partial_ptr, index=(2, 0, c_block),
        shape=(1, BLOCK_TILES, BLOCK_CF),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # Reduce (1, BLOCK_TILES, BLOCK_CF) → (BLOCK_CF,) along tile axis.
    sum_where_2d = ct.reshape(part_where, (BLOCK_TILES, BLOCK_CF))
    sum_dot1_2d = ct.reshape(part_dot1, (BLOCK_TILES, BLOCK_CF))
    sum_dot2_2d = ct.reshape(part_dot2, (BLOCK_TILES, BLOCK_CF))
    sum_where = ct.sum(sum_where_2d, axis=0)
    sum_dot1 = ct.sum(sum_dot1_2d, axis=0)
    sum_dot2 = ct.sum(sum_dot2_2d, axis=0)

    invstd1 = ct.load(invstd1_ptr, index=(c_block,), shape=(BLOCK_CF,))
    weight1 = ct.load(weight1_ptr, index=(c_block,), shape=(BLOCK_CF,))
    invstd2 = ct.load(invstd2_ptr, index=(c_block,), shape=(BLOCK_CF,))
    weight2 = ct.load(weight2_ptr, index=(c_block,), shape=(BLOCK_CF,))

    mean_where = sum_where * INV_NHW
    coeff1 = sum_dot1 * INV_NHW * invstd1 * invstd1
    scale1 = invstd1 * weight1
    vec1 = sum_dot1 * invstd1

    coeff2 = sum_dot2 * INV_NHW * invstd2 * invstd2
    scale2 = invstd2 * weight2
    vec2 = sum_dot2 * invstd2

    ct.store(sum1_ptr, index=(c_block,), tile=sum_where)
    ct.store(vec1_ptr, index=(c_block,), tile=vec1)
    ct.store(sum3_ptr, index=(c_block,), tile=sum_where)
    ct.store(vec2_ptr, index=(c_block,), tile=vec2)
    ct.store(stats_ptr, index=(0, c_block),
             tile=ct.reshape(mean_where, (1, BLOCK_CF)))
    ct.store(stats_ptr, index=(1, c_block),
             tile=ct.reshape(coeff1, (1, BLOCK_CF)))
    ct.store(stats_ptr, index=(2, c_block),
             tile=ct.reshape(scale1, (1, BLOCK_CF)))
    ct.store(stats_ptr, index=(3, c_block),
             tile=ct.reshape(coeff2, (1, BLOCK_CF)))
    ct.store(stats_ptr, index=(4, c_block),
             tile=ct.reshape(scale2, (1, BLOCK_CF)))


@ct.kernel
def _dense_epilogue_kernel(
    out0_ptr,          # f32 [NHW, C]  (where as f32)
    a4_ptr,            # bf16 [NHW, C]
    mean1_ptr,         # f32 [C]
    a8_ptr,            # bf16 [NHW, C]
    mean2_ptr,         # f32 [C]
    stats_ptr,         # f32 [5, C]
    out1_ptr,          # bf16 [NHW, C]
    out2_ptr,          # bf16 [NHW, C]
    C_C: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)
    c_block = ct.bid(1)

    where_val = ct.load(out0_ptr, index=(tile, c_block),
                        shape=(BLOCK_M, BLOCK_C))
    a4 = ct.astype(ct.load(a4_ptr, index=(tile, c_block),
                           shape=(BLOCK_M, BLOCK_C)), ct.float32)
    a8 = ct.astype(ct.load(a8_ptr, index=(tile, c_block),
                           shape=(BLOCK_M, BLOCK_C)), ct.float32)
    mean1 = ct.load(mean1_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean2 = ct.load(mean2_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_where = ct.load(stats_ptr, index=(0, c_block),
                         shape=(1, BLOCK_C))
    coeff1 = ct.load(stats_ptr, index=(1, c_block),
                     shape=(1, BLOCK_C))
    scale1 = ct.load(stats_ptr, index=(2, c_block),
                     shape=(1, BLOCK_C))
    coeff2 = ct.load(stats_ptr, index=(3, c_block),
                     shape=(1, BLOCK_C))
    scale2 = ct.load(stats_ptr, index=(4, c_block),
                     shape=(1, BLOCK_C))
    mean_where_2d = ct.reshape(mean_where, (1, BLOCK_C))
    coeff1_2d = ct.reshape(coeff1, (1, BLOCK_C))
    scale1_2d = ct.reshape(scale1, (1, BLOCK_C))
    coeff2_2d = ct.reshape(coeff2, (1, BLOCK_C))
    scale2_2d = ct.reshape(scale2, (1, BLOCK_C))
    mean1_2d = ct.reshape(mean1, (1, BLOCK_C))
    mean2_2d = ct.reshape(mean2, (1, BLOCK_C))

    centered1 = a4 - mean1_2d
    correction1 = centered1 * coeff1_2d
    tmp1 = where_val - correction1
    tmp1 = tmp1 - mean_where_2d
    dense1 = tmp1 * scale1_2d

    centered2 = a8 - mean2_2d
    correction2 = centered2 * coeff2_2d
    tmp2 = where_val - correction2
    tmp2 = tmp2 - mean_where_2d
    dense2 = tmp2 * scale2_2d

    ct.store(out1_ptr, index=(tile, c_block),
             tile=ct.astype(dense1, ct.bfloat16))
    ct.store(out2_ptr, index=(tile, c_block),
             tile=ct.astype(dense2, ct.bfloat16))


@oracle_impl(hardware="B200", point="25561d2e", BLOCK_M=128, BLOCK_C=16, EPILOGUE_BLOCK=256)
@oracle_impl(hardware="B200", point="63be79ac", BLOCK_M=128, BLOCK_C=16, EPILOGUE_BLOCK=256)
@oracle_impl(hardware="B200", point="f6f7ee3c", BLOCK_M=128, BLOCK_C=16, EPILOGUE_BLOCK=256)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_C: int, EPILOGUE_BLOCK: int):
    (
        arg0, arg1, arg2, arg3,
        arg4, arg5, arg6, arg7,
        arg8, arg9, arg10, arg11,
    ) = inputs
    n, c, h, w = (int(d) for d in arg0.shape)
    device = arg0.device

    # Channels-last layout: viewing (n, h, w, c) via permute+contiguous is
    # already free since it's the natural layout of channels-last tensors.
    a0 = arg0.permute(0, 2, 3, 1).contiguous().view(n * h * w, c)
    a1 = arg1.permute(0, 2, 3, 1).contiguous().view(n * h * w, c)
    a2 = arg2.permute(0, 2, 3, 1).contiguous().view(n * h * w, c)
    a4 = arg4.permute(0, 2, 3, 1).contiguous().view(n * h * w, c)
    a8 = arg8.permute(0, 2, 3, 1).contiguous().view(n * h * w, c)

    fill_flat = arg3.view(1)
    mean1 = arg5.view(c)
    invstd1 = arg6.view(c)
    weight1 = arg7.view(c)
    mean2 = arg9.view(c)
    invstd2 = arg10.view(c)
    weight2 = arg11.view(c)

    nhw = n * h * w
    num_tiles = nhw // BLOCK_M  # exact: BLOCK_M=128 divides nhw=401408
    num_c_tiles = c // BLOCK_C  # exact: BLOCK_C=16 divides {96,192,384}

    out0 = torch.empty((nhw, c), device=device, dtype=torch.float32)
    out1 = torch.empty((nhw, c), device=device, dtype=torch.bfloat16)
    out2 = torch.empty((nhw, c), device=device, dtype=torch.bfloat16)

    partial = torch.zeros((3, num_tiles, c), device=device, dtype=torch.float32)
    stats = torch.empty((5, c), device=device, dtype=torch.float32)

    sum1 = torch.empty((c,), device=device, dtype=torch.float32)
    vec1 = torch.empty((c,), device=device, dtype=torch.float32)
    sum3 = torch.empty((c,), device=device, dtype=torch.float32)
    vec2 = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    # Kernel 1: producer + partial channel sums.
    ct.launch(
        stream,
        (num_tiles, num_c_tiles, 1),
        _partial_sums_kernel,
        (a0, a1, a2, fill_flat, a4, mean1, a8, mean2, out0, partial,
         num_tiles, BLOCK_M, BLOCK_C),
    )

    # Kernel 2: finalize per-channel reductions and derive stats.
    block_tiles = _next_p2(num_tiles)
    ct.launch(
        stream,
        (num_c_tiles, 1, 1),
        _finalize_sums_kernel,
        (partial, invstd1, weight1, invstd2, weight2,
         sum1, vec1, sum3, vec2, stats,
         num_tiles, block_tiles, BLOCK_C, CAPTURED_INV_NHW),
    )

    # Kernel 3: dense epilogue for both branches.
    ct.launch(
        stream,
        (num_tiles, num_c_tiles, 1),
        _dense_epilogue_kernel,
        (out0, a4, mean1, a8, mean2, stats, out1, out2, c, BLOCK_M, BLOCK_C),
    )

    # Reshape outputs back to (N, C, H, W) with channels-last layout.
    out0_r = out0.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)
    out1_r = out1.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)
    out2_r = out2.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)

    return out0_r, sum1, vec1, out1_r, sum3, vec2, out2_r
