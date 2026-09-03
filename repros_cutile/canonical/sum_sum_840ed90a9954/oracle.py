"""cuTile port of sum_sum_840ed90a9954: ghostnet BN-backward (160->80, 7x7).

Similar structure to sum_sum_bf3fda8e124c but with 80 channels, 7x7 spatial.
The 160-channel add produces `clone` (contiguous) and `copy` (channels-last),
we slice the last 80 channels for BN backward, reduce over N*H*W per channel
with cuTile, then run an epilogue kernel to write the bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_IN = 160
C = 80
H = 7
W = 7
NHW = N * H * W  # 25088
NCHW_IN = N * C_IN * H * W  # 4_014_080
REDUCE_SCALE = 3.985969387755102e-05


@ct.kernel
def _add_bf16_kernel(
    a_ptr,          # bf16 (N*C_IN*H*W,)
    b_ptr,          # bf16 (N*C_IN*H*W,)
    out_ptr,        # bf16 (N*C_IN*H*W,)
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    a = ct.load(a_ptr, index=(pid,), shape=(BLOCK,))
    b = ct.load(b_ptr, index=(pid,), shape=(BLOCK,))
    y = ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32), ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=y)


@ct.kernel
def _partial_reduce_kernel(
    slice_ptr,        # bf16 (NHW, C) — sliced high 80 channels flattened
    arg2_ptr,         # bf16 (NHW, C)
    mean_ptr,         # f32  (C,)
    partial_sum_ptr,  # f32  (NUM_TILES, C)
    partial_dot_ptr,  # f32  (NUM_TILES, C)
    NHW_C: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    tile = ct.bid(0)
    c_block = ct.bid(1)

    slice_bf = ct.load(slice_ptr, index=(tile, c_block), shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    arg2_bf = ct.load(arg2_ptr, index=(tile, c_block), shape=(BLOCK_R, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    slice_f = ct.astype(slice_bf, ct.float32)
    arg2_f = ct.astype(arg2_bf, ct.float32)

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    centered = arg2_f - mean_2d
    dot = slice_f * centered

    row_idx = ct.arange(BLOCK_R, dtype=ct.int32) + tile * BLOCK_R
    col_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    row_valid = ct.reshape(row_idx < NHW_C, (BLOCK_R, 1))
    col_valid = ct.reshape(col_idx < C_C, (1, BLOCK_C))
    valid = row_valid & col_valid
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    slice_m = ct.where(valid, slice_f, zero_f)
    dot_m = ct.where(valid, dot, zero_f)

    p_sum = ct.sum(slice_m, axis=0, keepdims=True)
    p_dot = ct.sum(dot_m, axis=0, keepdims=True)
    ct.store(partial_sum_ptr, index=(tile, c_block), tile=p_sum)
    ct.store(partial_dot_ptr, index=(tile, c_block), tile=p_dot)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32  (NUM_TILES, C)
    partial_dot_ptr,   # f32  (NUM_TILES, C)
    invstd_ptr,        # f32  (C,)
    weight_ptr,        # f32  (C,)
    sum_out_ptr,       # f32  (C,)
    mul8_out_ptr,      # f32  (C,)
    channel_mean_ptr,  # f32  (C,)
    weight2_ptr,       # f32  (C,)
    scale_ptr,         # f32  (C,)
    NUM_TILES: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    p_sum = ct.load(partial_sum_ptr, index=(0, c_block),
                    shape=(BLOCK_TILES, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    p_dot = ct.load(partial_dot_ptr, index=(0, c_block),
                    shape=(BLOCK_TILES, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    tile_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tile_valid = ct.reshape(tile_idx < NUM_TILES, (BLOCK_TILES, 1))
    zero_f = ct.zeros((BLOCK_TILES, BLOCK_C), dtype=ct.float32)
    p_sum_v = ct.where(tile_valid, p_sum, zero_f)
    p_dot_v = ct.where(tile_valid, p_dot, zero_f)
    sum_1 = ct.sum(p_sum_v, axis=0)
    sum_2 = ct.sum(p_dot_v, axis=0)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    channel_mean = sum_1 * REDUCE_SCALE
    weight2 = sum_2 * REDUCE_SCALE * invstd * invstd
    scale = invstd * weight
    mul_8 = sum_2 * invstd
    ct.store(sum_out_ptr, index=(c_block,), tile=sum_1)
    ct.store(mul8_out_ptr, index=(c_block,), tile=mul_8)
    ct.store(channel_mean_ptr, index=(c_block,), tile=channel_mean)
    ct.store(weight2_ptr, index=(c_block,), tile=weight2)
    ct.store(scale_ptr, index=(c_block,), tile=scale)


@ct.kernel
def _bn_backward_kernel(
    slice_ptr,        # bf16 (NHW, C)
    arg2_ptr,         # bf16 (NHW, C)
    saved_mean_ptr,   # f32  (C,)
    scale_ptr,        # f32  (C,)
    weight2_ptr,      # f32  (C,)
    channel_mean_ptr, # f32  (C,)
    out_ptr,          # bf16 (NHW, C)
    BLOCK_C: ct.Constant[int],
):
    pid = ct.bid(0)
    c_block = ct.bid(1)
    slice_bf = ct.load(slice_ptr, index=(pid, c_block), shape=(1, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    arg2_bf = ct.load(arg2_ptr, index=(pid, c_block), shape=(1, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    slice_f = ct.astype(slice_bf, ct.float32)
    arg2_f = ct.astype(arg2_bf, ct.float32)

    saved_mean = ct.load(saved_mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                         padding_mode=ct.PaddingMode.ZERO)
    scale = ct.load(scale_ptr, index=(c_block,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    weight2 = ct.load(weight2_ptr, index=(c_block,), shape=(BLOCK_C,),
                      padding_mode=ct.PaddingMode.ZERO)
    channel_mean = ct.load(channel_mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                           padding_mode=ct.PaddingMode.ZERO)
    sm_2d = ct.reshape(saved_mean, (1, BLOCK_C))
    scale_2d = ct.reshape(scale, (1, BLOCK_C))
    w2_2d = ct.reshape(weight2, (1, BLOCK_C))
    cm_2d = ct.reshape(channel_mean, (1, BLOCK_C))

    sub = arg2_f - sm_2d
    mul_6 = sub * w2_2d
    sub_1 = slice_f - mul_6
    sub_2 = sub_1 - cm_2d
    mul_7 = sub_2 * scale_2d
    ct.store(out_ptr, index=(pid, c_block), tile=ct.astype(mul_7, ct.bfloat16))


@oracle_impl(hardware="B200", point="e2841909", BLOCK_R=512, BLOCK_C=16, ADD_BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, ADD_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _s0, _s1 = inputs
    device = arg0_1.device

    # arg0 is bf16[N, 160, H, W] contiguous (default NCHW).
    # arg1 is bf16[N, 160, H, W] channels-last (7840, 1, 1120, 160).
    # For BN backward we need the values, so materialize add as NHWC-flat.
    arg0_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous().view(-1)
    arg1_nhwc = arg1_1.permute(0, 2, 3, 1).contiguous().view(-1)
    add_flat = torch.empty(NCHW_IN, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NCHW_IN // ADD_BLOCK, 1, 1),
        _add_bf16_kernel,
        (arg0_nhwc, arg1_nhwc, add_flat, ADD_BLOCK),
    )

    # add_flat is (N, H, W, C_IN) memory-contiguous.
    add_nhwc = add_flat.view(N, H, W, C_IN)
    # clone: memory-format contiguous_format (NCHW contiguous).
    clone = add_nhwc.permute(0, 3, 1, 2).contiguous()
    # copy_1: same values but with NHWC memory layout (channels-last stride).
    # The repro's new_empty_strided uses (7840, 1, 1120, 160).
    copy_1 = torch.empty_strided(
        (N, C_IN, H, W),
        (C_IN * H * W, 1, W * C_IN, C_IN),
        device=device, dtype=torch.bfloat16,
    )
    copy_1.copy_(clone)

    # slice_1 = copy_1[:, 80:160]  → high 80 channels.
    slice_nhwc = add_nhwc[..., C:C_IN].contiguous()  # (N, H, W, 80)
    slice_2d = slice_nhwc.view(NHW, C)

    arg2_nhwc = arg2_1.permute(0, 2, 3, 1).contiguous()
    arg2_2d = arg2_nhwc.view(NHW, C)

    saved_mean_flat = arg3_1.view(C)
    invstd = arg4_1.to(torch.float32)
    weight = arg5_1.to(torch.float32)

    num_tiles = (NHW + BLOCK_R - 1) // BLOCK_R
    num_c_blocks = (C + BLOCK_C - 1) // BLOCK_C
    partial_sum = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, C), device=device, dtype=torch.float32)

    ct.launch(
        stream,
        (num_tiles, num_c_blocks, 1),
        _partial_reduce_kernel,
        (slice_2d, arg2_2d, saved_mean_flat, partial_sum, partial_dot,
         NHW, C, BLOCK_R, BLOCK_C),
    )

    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    mul_8 = torch.empty((C,), device=device, dtype=torch.float32)
    channel_mean = torch.empty((C,), device=device, dtype=torch.float32)
    weight2 = torch.empty((C,), device=device, dtype=torch.float32)
    scale = torch.empty((C,), device=device, dtype=torch.float32)

    block_tiles = 1 << (int(num_tiles) - 1).bit_length()
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd, weight,
         sum_1, mul_8, channel_mean, weight2, scale,
         num_tiles, C, block_tiles, BLOCK_C),
    )

    out_flat = torch.empty((NHW, C), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (NHW, num_c_blocks, 1),
        _bn_backward_kernel,
        (slice_2d, arg2_2d, saved_mean_flat, scale, weight2, channel_mean,
         out_flat, BLOCK_C),
    )

    out_final = torch.empty_strided(
        (N, C, H, W),
        (H * W * C, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out_final.copy_(out_flat.view(N, H, W, C).permute(0, 3, 1, 2))

    return clone, copy_1, sum_1, mul_8, out_final
