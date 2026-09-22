"""cuTile port of sum_sum_f7157664c88b: MobileViT BN-backward slice-add tail.

Uses cuTile per-chunk 2D tile reductions. Channel dim is C=160 (non-pow2),
so we pad to 256 with PaddingMode.ZERO and rely on out-of-range channels
contributing zero to the reduction.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
IN_CHANNELS = 320
CHANNELS = 160
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW  # 128 * 64 = 8192
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 0.0001220703125
OUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (CHANNELS * HW, 1, CHANNELS * WIDTH, CHANNELS)
BLOCK_C_PAD = 256   # next pow2 >= 160


@ct.kernel
def _fused_bn_backward_kernel(
    wide_ptr,      # bf16 [ELEMENTS_PER_CHANNEL, IN_CHANNELS]
    residual_ptr,  # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    x_ptr,         # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    mean_ptr,      # f32 [CHANNELS]
    partial_sum_ptr,   # f32 [NUM_CHUNKS, CHANNELS_PAD]
    partial_prod_ptr,  # f32 [NUM_CHUNKS, CHANNELS_PAD]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    # Load with padding-zero for channels beyond CHANNELS.
    wide = ct.load(wide_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)

    wide_f = ct.astype(wide, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    x_f = ct.astype(x, ct.float32)

    add_bf = ct.astype(wide_f + residual_f, ct.bfloat16)
    add_f = ct.astype(add_bf, ct.float32)

    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    centered = x_f - mean_2d
    prod = add_f * centered

    partial_sum = ct.sum(add_f, axis=0)
    partial_prod = ct.sum(prod, axis=0)
    ct.store(partial_sum_ptr, index=(chunk, 0), tile=ct.reshape(partial_sum, (1, BLOCK_C)))
    ct.store(partial_prod_ptr, index=(chunk, 0), tile=ct.reshape(partial_prod, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,     # f32 [NUM_CHUNKS, CHANNELS_PAD]
    partial_prod_ptr,    # f32 [NUM_CHUNKS, CHANNELS_PAD]
    invstd_ptr,          # f32 [CHANNELS_PAD]
    weight_ptr,          # f32 [CHANNELS_PAD]
    sum_out_ptr,         # f32 [CHANNELS_PAD]
    prod_scaled_out_ptr, # f32 [CHANNELS_PAD]
    mean_term_ptr,       # f32 [CHANNELS_PAD]
    prod_coeff_ptr,      # f32 [CHANNELS_PAD]
    output_scale_ptr,    # f32 [CHANNELS_PAD]
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    REDUCE_SCALE_: ct.Constant[float],
):
    total_sum = ct.zeros((BLOCK_C,), dtype=ct.float32)
    total_prod = ct.zeros((BLOCK_C,), dtype=ct.float32)
    for chunk in range(NUM_CHUNKS):
        s = ct.load(partial_sum_ptr, index=(chunk, 0), shape=(1, BLOCK_C))
        p = ct.load(partial_prod_ptr, index=(chunk, 0), shape=(1, BLOCK_C))
        total_sum = total_sum + ct.reshape(s, (BLOCK_C,))
        total_prod = total_prod + ct.reshape(p, (BLOCK_C,))

    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)

    ct.store(sum_out_ptr, index=(0,), tile=total_sum)
    ct.store(prod_scaled_out_ptr, index=(0,), tile=total_prod * invstd)
    ct.store(mean_term_ptr, index=(0,), tile=total_sum * REDUCE_SCALE_)
    ct.store(prod_coeff_ptr, index=(0,), tile=total_prod * REDUCE_SCALE_ * invstd * invstd)
    ct.store(output_scale_ptr, index=(0,), tile=invstd * weight)


@ct.kernel
def _epilogue_kernel(
    wide_ptr,           # bf16 [ELEMENTS_PER_CHANNEL, IN_CHANNELS]
    residual_ptr,       # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    x_ptr,              # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    mean_ptr,           # f32 [CHANNELS]
    mean_term_ptr,      # f32 [CHANNELS]
    prod_coeff_ptr,     # f32 [CHANNELS]
    output_scale_ptr,   # f32 [CHANNELS]
    out_ptr,            # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    wide = ct.load(wide_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    prod_coeff = ct.load(prod_coeff_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)

    wide_f = ct.astype(wide, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    prod_coeff_2d = ct.reshape(prod_coeff, (1, BLOCK_C))
    output_scale_2d = ct.reshape(output_scale, (1, BLOCK_C))

    add_bf = ct.astype(wide_f + residual_f, ct.bfloat16)
    add_f = ct.astype(add_bf, ct.float32)
    centered = x_f - mean_2d
    correction = centered * prod_coeff_2d
    residual2 = add_f - correction - mean_term_2d
    out_f = residual2 * output_scale_2d

    # Store back to out_ptr [ELEMENTS_PER_CHANNEL, CHANNELS]; use scatter to mask
    # channels >= CHANNELS.
    ct.store(out_ptr, index=(chunk, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="5d467cd6", BLOCK_R=64)
def oracle_forward(inputs, *, BLOCK_R: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    # Reorganize to NHWC-like layout: [ELEMENTS_PER_CHANNEL, CHANNELS_or_INCHANNELS]
    wide_full = arg0_1.permute(0, 2, 3, 1).contiguous().view(ELEMENTS_PER_CHANNEL, IN_CHANNELS)
    residual = arg1_1.permute(0, 2, 3, 1).contiguous().view(ELEMENTS_PER_CHANNEL, CHANNELS)
    x = arg2_1.permute(0, 2, 3, 1).contiguous().view(ELEMENTS_PER_CHANNEL, CHANNELS)
    mean = arg3_1.view(CHANNELS)
    invstd = arg4_1
    weight = arg5_1

    # For the kernel we want a [ELEMENTS_PER_CHANNEL, CHANNELS] view of wide (slice).
    # cuTile can only load a tile from a tensor whose last-dim stride matches tile.
    # Take a contiguous copy of the slice arg0[:, :CHANNELS].
    wide_slice = wide_full[:, :CHANNELS].contiguous()

    num_chunks = (ELEMENTS_PER_CHANNEL + BLOCK_R - 1) // BLOCK_R
    partial_sum = torch.zeros((num_chunks, BLOCK_C_PAD), device=device, dtype=torch.float32)
    partial_prod = torch.zeros((num_chunks, BLOCK_C_PAD), device=device, dtype=torch.float32)

    # Pad wide_slice/residual/x to CHANNELS_PAD along last dim so cuTile can load pow2 tile.
    # Easier: pad the mean/invstd/weight vectors too and pad the compute buffers.
    def _pad_c(t):
        if t.shape[-1] == BLOCK_C_PAD:
            return t
        out = torch.zeros(t.shape[:-1] + (BLOCK_C_PAD,), device=t.device, dtype=t.dtype)
        out[..., :t.shape[-1]] = t
        return out

    wide_pad = _pad_c(wide_slice)
    residual_pad = _pad_c(residual)
    x_pad = _pad_c(x)
    mean_pad = _pad_c(mean)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_chunks, 1, 1),
        _fused_bn_backward_kernel,
        (wide_pad, residual_pad, x_pad, mean_pad, partial_sum, partial_prod,
         BLOCK_R, BLOCK_C_PAD),
    )

    # Finalize on-device in cuTile.
    invstd_pad = _pad_c(invstd)
    weight_pad = _pad_c(weight)
    sum_out_pad = torch.zeros((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    prod_scaled_out_pad = torch.zeros((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    mean_term_pad = torch.zeros((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    prod_coeff_pad = torch.zeros((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    output_scale_pad = torch.zeros((BLOCK_C_PAD,), device=device, dtype=torch.float32)

    ct.launch(
        stream, (1, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, invstd_pad, weight_pad,
         sum_out_pad, prod_scaled_out_pad, mean_term_pad, prod_coeff_pad, output_scale_pad,
         num_chunks, BLOCK_C_PAD, REDUCE_SCALE),
    )
    sum_1 = sum_out_pad[:CHANNELS].contiguous()
    prod_scaled_out = prod_scaled_out_pad[:CHANNELS].contiguous()

    out_flat = torch.empty((ELEMENTS_PER_CHANNEL, BLOCK_C_PAD), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (num_chunks, 1, 1),
        _epilogue_kernel,
        (wide_pad, residual_pad, x_pad, mean_pad, mean_term_pad, prod_coeff_pad, output_scale_pad,
         out_flat, BLOCK_R, BLOCK_C_PAD),
    )
    out_valid = out_flat[:, :CHANNELS].contiguous()  # bf16 [8192, 160]
    out_nhwc = out_valid.view(BATCH, HEIGHT, WIDTH, CHANNELS)
    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.bfloat16)
    out.copy_(out_nhwc.permute(0, 3, 1, 2))
    return sum_1, prod_scaled_out, out
