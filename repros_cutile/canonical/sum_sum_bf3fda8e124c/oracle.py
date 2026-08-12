"""cuTile port of sum_sum_bf3fda8e124c: ghostnet BN-backward with dual channel-cat.

Structure (matches the Triton oracle at a high level):
  1. `_add_kernel` computes `clone = bf16(f32(arg0) + f32(arg1))` (contiguous
     NCHW), producing the value later re-emitted with the channels-last stride
     as `copy`.
  2. `_partial_reduce_kernel` walks (channel, spatial-tile) and produces per
     tile sum_1[c, tile] and sum_2[c, tile] = sum(slice_bf * (arg2 - mean)),
     where slice_bf = clone[:, 12:24].
  3. `_bn_backward_kernel` produces the final bf16 output element-wise from
     the per-channel scalars (finalized in torch).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_IN = 24
C = 12
H = 56
W = 56
NHW = N * H * W
NCHW = N * C * H * W
NCHW_IN = N * C_IN * H * W
REDUCE_SCALE = 6.228077168367346e-07


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
def _finalize_partials_kernel(
    partial_sum_ptr,   # f32 (NUM_TILES, C)
    partial_dot_ptr,   # f32 (NUM_TILES, C)
    sum1_out_ptr,      # f32 (C,)
    sum2_out_ptr,      # f32 (C,)
    NUM_TILES_: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    # partial_{sum,dot} is stored as (NUM_TILES, C). Use a 2D tile load with
    # ZERO padding for the out-of-range tiles at the tail.
    sum_partials = ct.load(
        partial_sum_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    dot_partials = ct.load(
        partial_dot_ptr, index=(0, c_block), shape=(BLOCK_TILES, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sum_final = ct.sum(sum_partials, axis=0)   # (BLOCK_C,)
    dot_final = ct.sum(dot_partials, axis=0)
    cols = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    col_mask = cols < C_C
    ct.scatter(sum1_out_ptr, cols, sum_final, mask=col_mask)
    ct.scatter(sum2_out_ptr, cols, dot_final, mask=col_mask)


@ct.kernel
def _partial_reduce_kernel(
    slice_ptr,        # bf16 (NHW, C)  — arg0+arg1 slice[..., C:C_IN] flat
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
    slice_bf = ct.load(slice_ptr, index=(pid, 0), shape=(1, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    arg2_bf = ct.load(arg2_ptr, index=(pid, 0), shape=(1, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    slice_f = ct.astype(slice_bf, ct.float32)
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
    sub_1 = slice_f - mul_6
    sub_2 = sub_1 - cm_2d
    mul_7 = sub_2 * scale_2d
    ct.store(out_ptr, index=(pid, 0), tile=ct.astype(mul_7, ct.bfloat16))


@oracle_impl(hardware="B200", point="a63c41b1", BLOCK_R=1024, BLOCK_C=16, ADD_BLOCK=1024)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, ADD_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    # arg0, arg1 are both bf16[N, 24, H, W] channels-last strided
    # (75264, 1, 1344, 24). Permute to (N, H, W, C_IN) contiguous, then add.
    # arg2 is bf16[N, 12, H, W] channels-last (37632, 1, 672, 12).
    arg0_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous().view(-1)  # (N*H*W*C_IN,)
    arg1_nhwc = arg1_1.permute(0, 2, 3, 1).contiguous().view(-1)
    add_flat = torch.empty(NCHW_IN, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NCHW_IN // ADD_BLOCK, 1, 1),
        _add_bf16_kernel,
        (arg0_nhwc, arg1_nhwc, add_flat, ADD_BLOCK),
    )

    # add_flat is (N, H, W, C_IN) memory-contiguous. This is exactly the same
    # values as `clone` from the repro but laid out as NHWC. Re-materialize
    # for clone (NCHW contiguous) and copy (NCHW channels-last strided).
    add_nhwc = add_flat.view(N, H, W, C_IN)
    clone = add_nhwc.permute(0, 3, 1, 2).contiguous()  # NCHW contiguous
    copy = torch.empty_strided(
        (N, C_IN, H, W),
        (C_IN * H * W, 1, W * C_IN, C_IN),
        device=device, dtype=torch.bfloat16,
    )
    copy.copy_(clone)

    # slice_1 = copy[:, 12:24] equals add_nhwc[..., 12:24]. Take channels [C:C_IN].
    slice_nhwc = add_nhwc[..., C:C_IN].contiguous()  # (N, H, W, C)
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

    def _next_pow2(v):
        r = 1
        while r < v:
            r <<= 1
        return r

    BLOCK_TILES = _next_pow2(num_tiles)
    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_partials_kernel,
        (partial_sum, partial_dot, sum_1, sum_2,
         num_tiles, C, BLOCK_TILES, BLOCK_C),
    )
    channel_mean = sum_1 * REDUCE_SCALE
    weight2 = sum_2 * REDUCE_SCALE * invstd * invstd
    scale = invstd * weight
    mul_8 = sum_2 * invstd

    out_flat = torch.empty((NHW, C), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (NHW, 1, 1),
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

    return clone, copy, sum_1, mul_8, out_final
