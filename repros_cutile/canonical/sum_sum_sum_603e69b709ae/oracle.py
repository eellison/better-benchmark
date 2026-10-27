"""cuTile port of sum_sum_sum_603e69b709ae: NFNet residual-block backward.

Uses torch for the avg_pool2d_backward, sigmoid, and torch.special.erf (as
cuTile lacks direct primitives). Uses cuTile for the GELU-derivative +
bf16 rounding chain, and for the sum_3/sum_4 channel reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
CHANNELS = 256
H_LARGE = 48
W_LARGE = 48
HW_LARGE = H_LARGE * W_LARGE  # 2304


@ct.kernel
def _gelu_deriv_kernel(
    z_ptr,          # f32 [N] (post-add convert to f32)
    upstream_ptr,   # f32 [N] (convert_element_type = mul_1 as f32)
    erf_ptr,        # f32 [N] pre-computed erf(z * 0.7071...)
    out_ptr,        # bf16 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    z = ct.load(z_ptr, index=(pid,), shape=(BLOCK,))
    upstream = ct.load(upstream_ptr, index=(pid,), shape=(BLOCK,))
    erf = ct.load(erf_ptr, index=(pid,), shape=(BLOCK,))

    cdf = (erf + 1.0) * 0.5
    z_sq = z * z
    pdf = ct.exp(z_sq * -0.5) * 0.3989422804014327
    gelu_deriv = cdf + z * pdf
    upstream_deriv = upstream * gelu_deriv
    ct.store(out_ptr, index=(pid,), tile=ct.astype(upstream_deriv, ct.bfloat16))


@ct.kernel
def _reduce_conv_only_kernel(
    conv_bf_ptr,        # bf16 [BATCH*CHANNELS, HW]  (contig)
    partial_conv_ptr,   # f32  [BATCH, CHANNELS]  per-(n,c) sum(conv_bf)
    CHANNELS_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    NUM_HW_BLOCKS: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    row = n * CHANNELS_ + c
    acc = ct.full(shape=(1, BLOCK_HW), fill_value=0.0, dtype=ct.float32)
    for hb in range(NUM_HW_BLOCKS):
        conv_bf = ct.load(conv_bf_ptr, index=(row, hb),
                          shape=(1, BLOCK_HW),
                          padding_mode=ct.PaddingMode.ZERO)
        acc = acc + ct.astype(conv_bf, ct.float32)
    ct.store(partial_conv_ptr, index=(n, c),
             tile=ct.sum(acc, axis=1, keepdims=True))


@ct.kernel
def _finalize_channel_kernel(
    partial_conv_ptr,   # f32 [BATCH, CHANNELS]
    partial_conv6_ptr,  # f32 [BATCH, CHANNELS]
    conv_out_ptr,       # f32 [CHANNELS]
    conv6_out_ptr,      # f32 [CHANNELS]
    BATCH_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_blk = ct.bid(0)
    tile_conv = ct.load(partial_conv_ptr, index=(0, c_blk),
                        shape=(BATCH_, BLOCK_C))
    tile_conv6 = ct.load(partial_conv6_ptr, index=(0, c_blk),
                         shape=(BATCH_, BLOCK_C))
    ct.store(conv_out_ptr, index=(c_blk,), tile=ct.sum(tile_conv, axis=0))
    ct.store(conv6_out_ptr, index=(c_blk,), tile=ct.sum(tile_conv6, axis=0))


@oracle_impl(hardware="B200", point="9c141705", BLOCK=2048)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = inputs
    device = arg0_1.device

    # avg_pool2d_backward (torch)
    avg_pool_back = torch.ops.aten.avg_pool2d_backward.default(
        arg0_1, arg1_1, [2, 2], [2, 2], [0, 0], True, False, None)
    add_val = arg2_1 + avg_pool_back
    mul = add_val * 0.9805806756909201
    mul_1 = mul * 1.7015043497085571  # bf16
    mul_1_f = mul_1.to(torch.float32)

    sigmoid_val = torch.sigmoid(arg3_1)  # bf16
    mul_2 = arg4_1 * sigmoid_val
    mul_3 = mul_2 * 2.0  # bf16
    mul_4 = mul_3 * arg5_1
    mul_5 = mul_4 * 0.2
    add_1 = mul_5 + arg6_1
    z_f = add_1.to(torch.float32)

    erf_arg = z_f * 0.7071067811865476
    erf_val = torch.special.erf(erf_arg)

    total = z_f.numel()
    z_flat = z_f.contiguous().view(total)
    upstream_flat = mul_1_f.contiguous().view(total)
    erf_flat = erf_val.contiguous().view(total)
    out_flat = torch.empty(total, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (ct.cdiv(total, BLOCK), 1, 1), _gelu_deriv_kernel,
        (z_flat, upstream_flat, erf_flat, out_flat, BLOCK),
    )
    conv_bf = out_flat.view(BATCH, CHANNELS, H_LARGE, W_LARGE)

    # Torch does sum_1, sum_2 in fp32:
    mul_13 = conv_bf * 0.2
    mul_14 = mul_13 * mul_3
    mul_15 = mul_13 * arg5_1
    sum_1 = mul_14.sum(dtype=torch.float32)
    mul_16 = mul_15 * 2.0
    mul_17 = mul_16 * arg4_1
    sum_2 = mul_17.sum(dim=(2, 3), keepdim=True, dtype=torch.float32)

    conv3_bf = sum_2.to(torch.bfloat16)
    conv4_f = conv3_bf.to(torch.float32)
    sig_f = sigmoid_val.to(torch.float32)
    sub = 1 - sig_f
    mul_18 = sig_f * sub
    mul_19 = conv4_f * mul_18
    conv6_bf = mul_19.to(torch.bfloat16)

    # sum_3 and sum_4 via cuTile:
    # conv_bf is (N, C, 48, 48) channels-last; conv6_bf is (N, C, 1, 1).
    # We compute:
    #   sum_4 = conv_bf.sum(dim=(0,2,3)) -> (C,) via per-(n,c) HW sum then over N.
    #   sum_3 = conv6_bf.sum(dim=(0,2,3)) -> just sum over N since HW == 1.
    conv_bf_nchw = conv_bf.contiguous().view(BATCH * CHANNELS, HW_LARGE)
    # conv6_bf is (N, C, 1, 1) -> flatten to (BATCH, CHANNELS)
    conv6_bf_nc = conv6_bf.contiguous().view(BATCH, CHANNELS)
    # We need conv6_bf as (BATCH*CHANNELS, HW=1) for _reduce_channel_kernel but
    # HW=1 fails power-of-2 tile requirement. Instead: use a separate reduce path.
    # Easier: treat conv6_bf_nc as the "partial" directly (it IS per-(n,c) f32-ish
    # values). Actually the values are bf16 so we still need to convert.
    partial_conv = torch.empty((BATCH, CHANNELS), device=device, dtype=torch.float32)
    partial_conv6 = conv6_bf_nc.to(torch.float32)
    # Use _reduce_channel_kernel to compute partial_conv only. But it also
    # writes partial_conv6; give it a dummy input for conv6.
    # To keep kernel count reasonable, use a single-input reduction kernel.
    BLOCK_HW = 256
    NUM_HW_BLOCKS = HW_LARGE // BLOCK_HW  # 2304 / 256 = 9
    ct.launch(
        stream, (BATCH, CHANNELS, 1),
        _reduce_conv_only_kernel,
        (conv_bf_nchw, partial_conv, CHANNELS, HW_LARGE, BLOCK_HW, NUM_HW_BLOCKS),
    )
    conv_8 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    conv_7 = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    BLOCK_C = 32
    ct.launch(
        stream, (CHANNELS // BLOCK_C, 1, 1),
        _finalize_channel_kernel,
        (partial_conv, partial_conv6, conv_8, conv_7, BATCH, BLOCK_C),
    )

    return sigmoid_val, conv_bf, sum_1, mul_16, conv6_bf, conv_7, conv_8
