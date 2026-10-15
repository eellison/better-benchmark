"""cuTile port of sum_sum_sum_9a5b3965d355: MobileViT patch LN-backward.

Torch handles patch layout (multiple views + permutes + clones), then cuTile
row kernel computes row-wise LN backward (sum1, sum2, epilogue), and torch
computes feature-wise reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


# The eager repro hardcodes 144 as the LN divisor regardless of channel count.
LN_DIVISOR = 144.0


@ct.kernel
def _ln_bwd_row_kernel(
    view2_ptr,    # f32 [rows, BLOCK_C]  (patch-gathered fp32 x)
    weight_ptr,   # f32 [BLOCK_C]
    arg2_ptr,     # bf16 [rows, BLOCK_C]
    mean_ptr,     # f32 [rows]  (per-row mean)
    invstd_ptr,   # f32 [rows]  (per-row invstd)
    mul_out_ptr,  # f32 [rows, BLOCK_C]  (view2 * weight)
    mul2_out_ptr, # f32 [rows, BLOCK_C]  ((arg2_f - mean) * invstd)
    out_ptr,      # bf16 [rows, BLOCK_C]  (epilogue result)
    CHANNELS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row = ct.bid(0)
    view2 = ct.load(view2_ptr, index=(row, 0), shape=(1, BLOCK_C))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    arg2_bf = ct.load(arg2_ptr, index=(row, 0), shape=(1, BLOCK_C))
    mean_1d = ct.load(mean_ptr, index=(row,), shape=(1,))
    invstd_1d = ct.load(invstd_ptr, index=(row,), shape=(1,))
    mean_2d = ct.reshape(mean_1d, (1, 1))
    invstd_2d = ct.reshape(invstd_1d, (1, 1))

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_mask = ct.reshape(cols < CHANNELS, (1, BLOCK_C))

    mul = view2 * weight_2d
    arg2_f = ct.astype(arg2_bf, ct.float32)
    mul2 = (arg2_f - mean_2d) * invstd_2d

    ct.store(mul_out_ptr, index=(row, 0), tile=mul)
    ct.store(mul2_out_ptr, index=(row, 0), tile=mul2)

    mul1 = mul * LN_DIVISOR
    mul_masked = ct.where(col_mask, mul, 0.0)
    sum1 = ct.sum(mul_masked, axis=1, keepdims=True)
    prod = mul * mul2
    prod_masked = ct.where(col_mask, prod, 0.0)
    sum2 = ct.sum(prod_masked, axis=1, keepdims=True)
    mul4 = mul2 * sum2
    sub1 = mul1 - sum1
    sub2 = sub1 - mul4
    div_val = invstd_2d * (1.0 / LN_DIVISOR)
    mul5 = div_val * sub2
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(mul5, ct.bfloat16))


def _pad_block(v, block):
    """Pad last dim of a 2D tensor v to `block`."""
    if v.shape[-1] == block:
        return v.contiguous()
    padded = torch.zeros(v.shape[:-1] + (block,), device=v.device, dtype=v.dtype)
    padded[..., :v.shape[-1]].copy_(v)
    return padded


def _run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
         channels, k_tiles, height, width, block_c):
    device = arg0_1.device

    # Materialize the patch layout via torch.
    # arg0_1 is bf16 [128,144,32,32] channels-last strides [147456,1,4608,144]
    conv_f = arg0_1.to(torch.float32)
    clone = conv_f.contiguous()  # [128, 144, 32, 32]
    n = int(arg0_1.shape[0])
    # view [n*channels*height/2, 2, width/2, 2]
    view = clone.view(n * channels * height // 2, 2, width // 2, 2)
    permute = view.permute(0, 2, 1, 3).contiguous()
    # view [n, channels, (h/2)*(w/2), 4]
    view1 = permute.view(n, channels, (height // 2) * (width // 2), 4)
    permute1 = view1.permute(0, 3, 2, 1).contiguous()
    # view [n*4, k_tiles, channels] = [512, k_tiles, channels]
    view2 = permute1.view(n * 4, k_tiles, channels)

    # mul = view2 * arg1_1  (broadcast arg1_1 over first two dims)
    # These are all fp32 (arg1_1 is fp32[144]).
    # Compute row-wise LN backward for each row, each of size `channels`.
    total_rows = 512 * k_tiles  # 131072 for 144-channel
    rows = total_rows
    view2_flat = view2.reshape(rows, channels)  # f32
    arg2_flat = arg2_1.view(rows, channels)  # bf16
    arg3_flat = arg3_1.view(rows)  # f32 - mean (already computed)
    arg4_flat = arg4_1.view(rows)  # f32 - invstd (already computed)

    # Pad if needed.
    view2_pad = _pad_block(view2_flat, block_c)
    arg2_pad = _pad_block(arg2_flat, block_c)
    weight_pad = _pad_block(arg1_1.view(1, channels), block_c).view(block_c)

    mul_pad = torch.empty((rows, block_c), device=device, dtype=torch.float32)
    mul2_pad = torch.empty((rows, block_c), device=device, dtype=torch.float32)
    out_pad = torch.empty((rows, block_c), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _ln_bwd_row_kernel,
        (view2_pad, weight_pad, arg2_pad, arg3_flat, arg4_flat,
         mul_pad, mul2_pad, out_pad, channels, block_c),
    )

    # Unpad
    if block_c == channels:
        mul = mul_pad
        mul2 = mul2_pad
        out = out_pad
    else:
        mul = mul_pad.narrow(1, 0, channels).contiguous()
        mul2 = mul2_pad.narrow(1, 0, channels).contiguous()
        out = out_pad.narrow(1, 0, channels).contiguous()

    # Feature-wise reductions.
    # sum_3 = sum(view2 * mul2, dim=[0,1]) — [144]
    # sum_4 = sum(view2, dim=[0,1]) — [144]
    view2_2d = view2.view(rows, channels)
    sum_3 = (view2_2d * mul2).sum(dim=0)
    sum_4 = view2_2d.sum(dim=0)

    # out is [rows, channels]. Reshape back to [512, k_tiles, channels] and to view_3.
    out_reshaped = out.view(512, k_tiles, channels)
    view_3 = out_reshaped.view(rows, channels)
    permute_2 = view_3.permute(1, 0)
    # sum_5 = sum(view_3, dim=[0], keepdim, dtype=f32) - [1, 144]
    sum_5 = view_3.sum(dim=[0], keepdim=True, dtype=torch.float32)
    view_4 = sum_5.view(channels)
    ce3 = view_4.to(torch.bfloat16)
    ce4 = ce3.to(torch.float32)
    return sum_3, sum_4, out_reshaped, view_3, permute_2, ce4


@oracle_impl(hardware="B200", point="5afc2635",
             CHANNELS=144, K_TILES=256, HEIGHT=32, WIDTH=32, BLOCK_C=256)
@oracle_impl(hardware="B200", point="052eef63",
             CHANNELS=192, K_TILES=64, HEIGHT=16, WIDTH=16, BLOCK_C=256)
@oracle_impl(hardware="B200", point="9ff73aef",
             CHANNELS=240, K_TILES=16, HEIGHT=8, WIDTH=8, BLOCK_C=256)
def oracle_forward(inputs, *, CHANNELS, K_TILES, HEIGHT, WIDTH, BLOCK_C):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_ = inputs
    return _run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
                CHANNELS, K_TILES, HEIGHT, WIDTH, BLOCK_C)
