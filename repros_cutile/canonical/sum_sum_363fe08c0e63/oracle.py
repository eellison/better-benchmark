"""cuTile port of sum_sum_363fe08c0e63: ghostnet BN-backward tail (12ch, 56x56).

Matches the Triton 3-kernel structure:
  1. `_partial_reduce_kernel` — per (channel, spatial-tile) partial sum_1
     and sum_2 = sum(add_bf * (arg2 - mean)).
  2. `_finalize_kernel` — reduce partials along the tile axis and compute all
     per-channel derived scalars in-kernel.
  3. `_epilogue_kernel` — per-element BN-backward output.

All reductions and per-channel scalar math live in cuTile kernels; the torch
side only prepares layout buffers.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_IN = 24
C = 12
H = 56
W = 56
NHW = N * H * W  # 1,605,632
REDUCE_SCALE = 6.228077168367346e-07


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    add_ptr,          # bf16 (NHW, C)  channels-last flatten
    arg2_ptr,         # bf16 (NHW, C)  channels-last flatten of arg2
    mean_ptr,         # f32  (C,)
    partial_sum_ptr,  # f32  (NUM_TILES, C)  per-tile sum(add_f)
    partial_dot_ptr,  # f32  (NUM_TILES, C)  per-tile sum(add_f * (arg2 - mean))
    NHW_C: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    """One program per (tile, c_block). BLOCK_R rows, BLOCK_C cols."""
    tile = ct.bid(0)
    c_block = ct.bid(1)

    add_bf = ct.load(add_ptr, index=(tile, c_block), shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    arg2_bf = ct.load(arg2_ptr, index=(tile, c_block), shape=(BLOCK_R, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    add_f = ct.astype(add_bf, ct.float32)
    arg2_f = ct.astype(arg2_bf, ct.float32)

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))

    centered = arg2_f - mean_2d
    dot = add_f * centered

    # Row-mask to guard the tail tile that may extend past NHW.
    row_idx = ct.arange(BLOCK_R, dtype=ct.int32) + tile * BLOCK_R
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    row_valid = ct.reshape(row_idx < NHW_C, (BLOCK_R, 1))
    col_valid = ct.reshape(col_idx < C_C, (1, BLOCK_C))
    valid = row_valid & col_valid
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    add_masked = ct.where(valid, add_f, zero_f)
    dot_masked = ct.where(valid, dot, zero_f)

    p_sum = ct.sum(add_masked, axis=0, keepdims=True)  # (1, BLOCK_C)
    p_dot = ct.sum(dot_masked, axis=0, keepdims=True)  # (1, BLOCK_C)
    ct.store(partial_sum_ptr, index=(tile, c_block), tile=p_sum)
    ct.store(partial_dot_ptr, index=(tile, c_block), tile=p_dot)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,      # f32 (NUM_TILES, C)
    partial_dot_ptr,      # f32 (NUM_TILES, C)
    invstd_ptr,           # f32 (C,)
    weight_ptr,           # f32 (C,)
    sum_out_ptr,          # f32 (C,)
    scale_grad_ptr,       # f32 (C,) — sum_2 * invstd
    channel_mean_ptr,     # f32 (C,) — sum_1 * SCALE
    weight2_ptr,          # f32 (C,) — sum_2 * SCALE * invstd^2
    scale_ptr,            # f32 (C,) — invstd * weight
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c_block = ct.bid(0)
    tile = ct.load(partial_sum_ptr, index=(0, c_block),
                   shape=(BLOCK_TILES, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    tile2 = ct.load(partial_dot_ptr, index=(0, c_block),
                    shape=(BLOCK_TILES, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)

    # Mask rows past NUM_TILES.
    t_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    t_valid = ct.reshape(t_idx < NUM_TILES, (BLOCK_TILES, 1))
    c_valid = ct.reshape(c_idx < C_C, (1, BLOCK_C))
    valid = t_valid & c_valid
    zero_f = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)
    tile = ct.where(valid, tile, zero_f)
    tile2 = ct.where(valid, tile2, zero_f)

    sum_value = ct.sum(tile, axis=0)   # (BLOCK_C,)
    dot_value = ct.sum(tile2, axis=0)  # (BLOCK_C,)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    channel_mean = sum_value * SCALE_
    dot_scaled = dot_value * SCALE_
    invstd_sq = invstd * invstd
    weight2 = dot_scaled * invstd_sq
    scale = invstd * weight
    scale_grad = dot_value * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(scale_grad_ptr, index=(c_block,), tile=scale_grad)
    ct.store(channel_mean_ptr, index=(c_block,), tile=channel_mean)
    ct.store(weight2_ptr, index=(c_block,), tile=weight2)
    ct.store(scale_ptr, index=(c_block,), tile=scale)


@ct.kernel
def _bn_backward_kernel(
    add_ptr,          # bf16 (NHW, C) — channels-last flatten
    arg2_ptr,         # bf16 (NHW, C) — channels-last flatten
    saved_mean_ptr,   # f32  (C,)
    scale_ptr,        # f32  (C,) — invstd*weight
    weight2_ptr,      # f32  (C,) — sum_2 * SCALE * invstd^2
    channel_mean_ptr, # f32  (C,) — sum_1 * SCALE
    out_ptr,          # bf16 (NHW, C)
    BLOCK_C: ct.Constant[int],
):
    pid = ct.bid(0)
    add_bf = ct.load(add_ptr, index=(pid, 0), shape=(1, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    arg2_bf = ct.load(arg2_ptr, index=(pid, 0), shape=(1, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    add_f = ct.astype(add_bf, ct.float32)
    arg2_f = ct.astype(arg2_bf, ct.float32)

    saved_mean = ct.load(saved_mean_ptr, index=(0,), shape=(BLOCK_C,),
                         padding_mode=ct.PaddingMode.ZERO)
    scale = ct.load(scale_ptr, index=(0,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    weight2 = ct.load(weight2_ptr, index=(0,), shape=(BLOCK_C,),
                      padding_mode=ct.PaddingMode.ZERO)
    channel_mean = ct.load(channel_mean_ptr, index=(0,), shape=(BLOCK_C,),
                           padding_mode=ct.PaddingMode.ZERO)
    sm_2d = ct.reshape(saved_mean, (1, BLOCK_C))
    scale_2d = ct.reshape(scale, (1, BLOCK_C))
    w2_2d = ct.reshape(weight2, (1, BLOCK_C))
    cm_2d = ct.reshape(channel_mean, (1, BLOCK_C))

    sub = arg2_f - sm_2d
    mul_6 = sub * w2_2d
    sub_1 = add_f - mul_6
    sub_2 = sub_1 - cm_2d
    mul_7 = sub_2 * scale_2d
    ct.store(out_ptr, index=(pid, 0), tile=ct.astype(mul_7, ct.bfloat16))


@oracle_impl(hardware="B200", point="34d0cb10", BLOCK_R=1024, BLOCK_C=16)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    # arg0 is bf16[N, 24, H, W] channels-last (stride 75264,1,1344,24) — slice
    # channels [0:C] means the first 12 channels of the 24-channel layout.
    # arg1 is bf16[N, 12, H, W] channels-last (37632,1,672,12).
    # arg2 is bf16[N, 12, H, W] channels-last (37632,1,672,12).
    arg0_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()
    arg0_slice = arg0_nhwc[..., :C].contiguous()  # (N, H, W, 12)
    arg1_nhwc = arg1_1.permute(0, 2, 3, 1).contiguous()
    arg2_nhwc = arg2_1.permute(0, 2, 3, 1).contiguous()

    # bf16 add — match eager `arg0[:, 0:12] + arg1` (bf16 add).
    add_bf = (arg0_slice + arg1_nhwc)  # bf16 [N, H, W, 12] contiguous
    add_2d = add_bf.view(NHW, C)
    arg2_2d = arg2_nhwc.view(NHW, C)

    saved_mean_flat = arg3_1.view(C)
    invstd = arg4_1.to(torch.float32).contiguous()
    weight = arg5_1.to(torch.float32).contiguous()

    stream = torch.cuda.current_stream()

    num_tiles = (NHW + BLOCK_R - 1) // BLOCK_R
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    block_tiles = _next_power_of_2(num_tiles)
    partial_sum = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, C), device=device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_tiles, num_c_blocks, 1),
        _partial_reduce_kernel,
        (add_2d, arg2_2d, saved_mean_flat, partial_sum, partial_dot,
         NHW, C, BLOCK_R, BLOCK_C),
    )

    # Finalize per-channel scalars in-kernel.
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    channel_mean = torch.empty((C,), device=device, dtype=torch.float32)
    weight2 = torch.empty((C,), device=device, dtype=torch.float32)
    scale = torch.empty((C,), device=device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd, weight,
         sum_out, scale_grad, channel_mean, weight2, scale,
         num_tiles, block_tiles, C, BLOCK_C, REDUCE_SCALE),
    )

    # Epilogue.
    out_flat = torch.empty((NHW, C), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (NHW, 1, 1),
        _bn_backward_kernel,
        (add_2d, arg2_2d, saved_mean_flat, scale, weight2, channel_mean,
         out_flat, BLOCK_C),
    )

    # Reshape to (N, C, H, W) with channels-last stride to match repro.
    out_final = torch.empty_strided(
        (N, C, H, W),
        (H * W * C, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out_final.copy_(out_flat.view(N, H, W, C).permute(0, 3, 1, 2))

    return sum_out, scale_grad, out_final
