"""cuTile port of sum_sum_563676e3726c: ShuffleNet channel-shuffle + BN-backward.

Matches Triton's 3-kernel structure:
1. `_partial_kernel` — inline BN-forward + per-channel partial sums via `ct.sum`.
2. `_finalize_kernel` — reduces per-channel partials via `ct.sum` and emits BN
   scalars (sum_1, scaled_dot, mean_term, prod_coeff, output_scale).
3. `_epilogue_kernel` — BN-backward epilogue with bf16 output.

The channel-shuffle producer is kept in torch (matches previous port) since
Triton fuses it into the partial_reduce kernel, but the important fairness fix
is moving the sum reductions in-kernel.

C=116 non-pow2, padded to 128.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
IN_CHANNELS = 232
CHANNELS = 116
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW  # 128 * 196 = 25088
REDUCE_SCALE = 3.985969387755102e-05
BLOCK_C_PAD = 128  # next pow2 >= 116


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _partial_kernel(
    slice_ptr,     # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS] pad — slice_1 from view_1
    x_ptr,         # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS] pad — arg2_1
    mean_ptr,      # f32 [CHANNELS] pad
    invstd_ptr,    # f32 [CHANNELS] pad
    weight_ptr,    # f32 [CHANNELS] pad
    bias_ptr,      # f32 [CHANNELS] pad
    fill_ptr,      # bf16 [1]
    where_out_ptr, # f32 [ELEMENTS_PER_CHANNEL, CHANNELS_PAD]
    partial_sum_ptr, # f32 [NUM_CHUNKS, CHANNELS_PAD]
    partial_prod_ptr,# f32 [NUM_CHUNKS, CHANNELS_PAD]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    slice_1 = ct.load(slice_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C), padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C), padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))

    sub_f = x_f - mean_2d
    mul_f = sub_f * invstd_2d * weight_2d + bias_2d
    add1_bf = ct.astype(mul_f, ct.bfloat16)
    zero_bf = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.bfloat16)
    relu_bf = ct.where(add1_bf > zero_bf, add1_bf, zero_bf)
    le = relu_bf <= zero_bf

    fill_2d = ct.reshape(fill, (1, 1))
    where_bf = ct.where(le, fill_2d, slice_1)
    where_f = ct.astype(where_bf, ct.float32)
    ct.store(where_out_ptr, index=(chunk, 0), tile=where_f)

    prod = where_f * sub_f
    partial_sum = ct.sum(where_f, axis=0)
    partial_prod = ct.sum(prod, axis=0)
    ct.store(partial_sum_ptr, index=(chunk, 0), tile=ct.reshape(partial_sum, (1, BLOCK_C)))
    ct.store(partial_prod_ptr, index=(chunk, 0), tile=ct.reshape(partial_prod, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,    # f32 [NUM_CHUNKS, BLOCK_C_PAD]
    partial_prod_ptr,   # f32 [NUM_CHUNKS, BLOCK_C_PAD]
    invstd_ptr,         # f32 [BLOCK_C_PAD] pad
    weight_ptr,         # f32 [BLOCK_C_PAD] pad
    sum_out_ptr,        # f32 [BLOCK_C_PAD]
    scaled_dot_ptr,     # f32 [BLOCK_C_PAD]  (prod_sum * invstd)
    mean_term_ptr,      # f32 [BLOCK_C_PAD]
    prod_coeff_ptr,     # f32 [BLOCK_C_PAD]
    output_scale_ptr,   # f32 [BLOCK_C_PAD]
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
):
    """Per-channel column reduction with in-kernel `ct.sum`."""
    c = ct.bid(0)

    tile_col_sum = ct.load(
        partial_sum_ptr, index=(0, c), shape=(BLOCK_CHUNKS, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_col_dot = ct.load(
        partial_prod_ptr, index=(0, c), shape=(BLOCK_CHUNKS, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rows = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    active = ct.reshape(rows < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    tile_col_sum = ct.where(active, tile_col_sum, 0.0)
    tile_col_dot = ct.where(active, tile_col_dot, 0.0)

    sum_value = ct.reshape(ct.sum(tile_col_sum), ())
    dot_value = ct.reshape(ct.sum(tile_col_dot), ())

    invstd = ct.reshape(ct.load(invstd_ptr, index=(c,), shape=(1,)), ())
    weight = ct.reshape(ct.load(weight_ptr, index=(c,), shape=(1,)), ())
    invstd_sq = invstd * invstd

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(scaled_dot_ptr, index=(c,), tile=ct.reshape(dot_value * invstd, (1,)))
    ct.store(mean_term_ptr, index=(c,), tile=ct.reshape(sum_value * REDUCE_SCALE, (1,)))
    ct.store(prod_coeff_ptr, index=(c,), tile=ct.reshape(dot_value * REDUCE_SCALE * invstd_sq, (1,)))
    ct.store(output_scale_ptr, index=(c,), tile=ct.reshape(invstd * weight, (1,)))


@ct.kernel
def _epilogue_kernel(
    where_ptr,       # f32 [ELEMENTS_PER_CHANNEL, CHANNELS_PAD]
    x_ptr,           # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS_PAD]
    mean_ptr,        # f32 [CHANNELS_PAD]
    mean_term_ptr,   # f32 [CHANNELS_PAD]
    prod_coeff_ptr,  # f32 [CHANNELS_PAD]
    output_scale_ptr,# f32 [CHANNELS_PAD]
    out_ptr,         # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS_PAD]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    where_f = ct.load(where_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C), padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(chunk, 0), shape=(BLOCK_R, BLOCK_C), padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    prod_coeff = ct.load(prod_coeff_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    prod_coeff_2d = ct.reshape(prod_coeff, (1, BLOCK_C))
    output_scale_2d = ct.reshape(output_scale, (1, BLOCK_C))
    sub_f = x_f - mean_2d
    correction = sub_f * prod_coeff_2d
    res = where_f - correction - mean_term_2d
    out_f = res * output_scale_2d
    ct.store(out_ptr, index=(chunk, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="f0fa1db9", BLOCK_R=64)
def oracle_forward(inputs, *, BLOCK_R: int):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
     _shape0, _shape1) = inputs
    device = arg0_1.device

    # Channel shuffle: view_1 = permute(0, 2, 1, 3, 4).reshape(128, 232, 14, 14)
    add_bf = arg0_1 + arg1_1
    view = add_bf.view(BATCH, CHANNELS, 2, HEIGHT, WIDTH)
    permuted = view.permute(0, 2, 1, 3, 4).contiguous()
    view_1 = permuted.view(BATCH, 2 * CHANNELS, HEIGHT, WIDTH)
    slice_1 = view_1[:, CHANNELS:, :, :]  # last 116 channels

    def _nhwc_flat(t):
        return t.permute(0, 2, 3, 1).contiguous().view(ELEMENTS_PER_CHANNEL, CHANNELS)

    slice_flat = _nhwc_flat(slice_1)
    x_flat = _nhwc_flat(arg2_1)
    mean_c = arg3_1.view(CHANNELS)
    invstd_c = arg4_1.view(CHANNELS)
    weight_c = arg5_1
    bias_c = arg6_1

    def _pad_c(t):
        if t.shape[-1] == BLOCK_C_PAD:
            return t
        out = torch.zeros(t.shape[:-1] + (BLOCK_C_PAD,), device=t.device, dtype=t.dtype)
        out[..., :t.shape[-1]] = t
        return out

    slice_pad = _pad_c(slice_flat)
    x_pad = _pad_c(x_flat)
    mean_pad = _pad_c(mean_c)
    invstd_pad = _pad_c(invstd_c)
    weight_pad = _pad_c(weight_c)
    bias_pad = _pad_c(bias_c)
    fill_1d = arg7_1.reshape(1)

    num_chunks = (ELEMENTS_PER_CHANNEL + BLOCK_R - 1) // BLOCK_R
    where_out = torch.zeros((ELEMENTS_PER_CHANNEL, BLOCK_C_PAD), device=device, dtype=torch.float32)
    partial_sum = torch.zeros((num_chunks, BLOCK_C_PAD), device=device, dtype=torch.float32)
    partial_prod = torch.zeros((num_chunks, BLOCK_C_PAD), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_chunks, 1, 1),
        _partial_kernel,
        (slice_pad, x_pad, mean_pad, invstd_pad, weight_pad, bias_pad, fill_1d,
         where_out, partial_sum, partial_prod,
         BLOCK_R, BLOCK_C_PAD),
    )

    # In-kernel finalize (matches Triton's `_finalize_kernel` with `tl.sum`).
    sum_out = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    scaled_dot = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    mean_term_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    prod_coeff_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    output_scale_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    block_chunks = _next_pow2(num_chunks)
    ct.launch(
        stream,
        (BLOCK_C_PAD, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, invstd_pad, weight_pad,
         sum_out, scaled_dot, mean_term_pad, prod_coeff_pad, output_scale_pad,
         num_chunks, block_chunks),
    )

    out_flat = torch.empty((ELEMENTS_PER_CHANNEL, BLOCK_C_PAD), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (num_chunks, 1, 1),
        _epilogue_kernel,
        (where_out, x_pad, mean_pad, mean_term_pad, prod_coeff_pad, output_scale_pad, out_flat,
         BLOCK_R, BLOCK_C_PAD),
    )

    sum_1 = sum_out[:CHANNELS].contiguous()
    mul_10 = scaled_dot[:CHANNELS].contiguous()

    out_valid = out_flat[:, :CHANNELS].contiguous()
    out_nhwc = out_valid.view(BATCH, HEIGHT, WIDTH, CHANNELS)
    out_shape = (BATCH, CHANNELS, HEIGHT, WIDTH)
    out_stride = (CHANNELS * HW, 1, CHANNELS * WIDTH, CHANNELS)
    out = torch.empty_strided(out_shape, out_stride, device=device, dtype=torch.bfloat16)
    out.copy_(out_nhwc.permute(0, 3, 1, 2))
    return view_1, sum_1, mul_10, out
