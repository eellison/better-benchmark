"""cuTile port of sum_sum_1e07e3ba8c68: GhostNet BN-backward slice-add-relu-fill tail.

Three-kernel structure mirrors the Triton reference:
1. `_partial_reduce_kernel` — split-K partial sums of `where` and
   `where * centered`.
2. `_finalize_reduce_kernel` — reduces per-chunk partials; computes
   sum_out / dot_out / vector_out = dot * invstd in-kernel.
3. `_epilogue_kernel` — per-element BN-backward writeback.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
IN_CHANNELS = 960
CHANNELS = 480
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW  # 25088
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 3.985969387755102e-05
OUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (CHANNELS * HW, 1, CHANNELS * WIDTH, CHANNELS)
BLOCK_C_PAD = 512   # next pow2 >= 480
BLOCK_R = 64
NUM_CHUNKS = (ELEMENTS_PER_CHANNEL + BLOCK_R - 1) // BLOCK_R


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


NUM_CHUNKS_PAD = _next_pow2(NUM_CHUNKS)  # 512


@ct.kernel
def _bn_backward_partial_kernel(
    wide_ptr,      # bf16 [ELEMENTS_PER_CHANNEL, IN_CHANNELS] pad
    residual_ptr,  # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS_PAD]
    relu_input_ptr,# bf16 [ELEMENTS_PER_CHANNEL, CHANNELS_PAD]
    fill_ptr,      # bf16 [1]
    x_ptr,         # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS_PAD]
    mean_ptr,      # f32 [CHANNELS_PAD]
    partial_sum_ptr,  # f32 [NUM_CHUNKS, CHANNELS_PAD]
    partial_prod_ptr, # f32 [NUM_CHUNKS, CHANNELS_PAD]
    BLOCK_R_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    chunk = ct.bid(0)
    wide = ct.load(wide_ptr, index=(chunk, 0), shape=(BLOCK_R_, BLOCK_C_),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(chunk, 0), shape=(BLOCK_R_, BLOCK_C_),
                       padding_mode=ct.PaddingMode.ZERO)
    relu_in = ct.load(relu_input_ptr, index=(chunk, 0), shape=(BLOCK_R_, BLOCK_C_),
                      padding_mode=ct.PaddingMode.ZERO)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    x = ct.load(x_ptr, index=(chunk, 0), shape=(BLOCK_R_, BLOCK_C_),
                padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C_,),
                   padding_mode=ct.PaddingMode.ZERO)

    add_bf = ct.astype(ct.astype(wide, ct.float32) + ct.astype(residual, ct.float32), ct.bfloat16)
    zero_bf = ct.zeros((BLOCK_R_, BLOCK_C_), dtype=ct.bfloat16)
    le = relu_in <= zero_bf
    fill_2d = ct.reshape(fill, (1, 1))
    where_bf = ct.where(le, fill_2d, add_bf)
    where_f = ct.astype(where_bf, ct.float32)

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C_))
    centered = x_f - mean_2d
    prod = where_f * centered

    partial_sum = ct.sum(where_f, axis=0)
    partial_prod = ct.sum(prod, axis=0)
    ct.store(partial_sum_ptr, index=(chunk, 0),
             tile=ct.reshape(partial_sum, (1, BLOCK_C_)))
    ct.store(partial_prod_ptr, index=(chunk, 0),
             tile=ct.reshape(partial_prod, (1, BLOCK_C_)))


@ct.kernel
def _finalize_reduce_kernel(
    partial_sum_ptr,   # f32 [NUM_CHUNKS, C_PAD]
    partial_prod_ptr,  # f32 [NUM_CHUNKS, C_PAD]
    invstd_ptr,        # f32 [C_PAD]
    weight_ptr,        # f32 [C_PAD]
    sum_out_ptr,       # f32 [C_PAD]
    dot_out_ptr,       # f32 [C_PAD]
    vector_out_ptr,    # f32 [C_PAD]  (dot * invstd)
    mean_term_ptr,     # f32 [C_PAD]  (sum * SCALE)
    prod_coeff_ptr,    # f32 [C_PAD]  (dot * SCALE * invstd^2)
    output_scale_ptr,  # f32 [C_PAD]  (invstd * weight)
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    p_sum = ct.load(
        partial_sum_ptr, index=(0, 0), shape=(BLOCK_TILES, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    p_prod = ct.load(
        partial_prod_ptr, index=(0, 0), shape=(BLOCK_TILES, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sum_val = ct.sum(p_sum, axis=0)
    dot_val = ct.sum(p_prod, axis=0)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C_,))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C_,))

    vector = dot_val * invstd
    mean_term = sum_val * SCALE_
    dot_scaled = dot_val * SCALE_
    invstd_sq = invstd * invstd
    prod_coeff = dot_scaled * invstd_sq
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(0,), tile=sum_val)
    ct.store(dot_out_ptr, index=(0,), tile=dot_val)
    ct.store(vector_out_ptr, index=(0,), tile=vector)
    ct.store(mean_term_ptr, index=(0,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(0,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(0,), tile=output_scale)


@ct.kernel
def _bn_backward_epilogue_kernel(
    wide_ptr,
    residual_ptr,
    relu_input_ptr,
    fill_ptr,
    x_ptr,
    mean_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    BLOCK_R_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    chunk = ct.bid(0)
    wide = ct.load(wide_ptr, index=(chunk, 0), shape=(BLOCK_R_, BLOCK_C_), padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(chunk, 0), shape=(BLOCK_R_, BLOCK_C_), padding_mode=ct.PaddingMode.ZERO)
    relu_in = ct.load(relu_input_ptr, index=(chunk, 0), shape=(BLOCK_R_, BLOCK_C_), padding_mode=ct.PaddingMode.ZERO)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    x = ct.load(x_ptr, index=(chunk, 0), shape=(BLOCK_R_, BLOCK_C_), padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C_,), padding_mode=ct.PaddingMode.ZERO)
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(BLOCK_C_,), padding_mode=ct.PaddingMode.ZERO)
    prod_coeff = ct.load(prod_coeff_ptr, index=(0,), shape=(BLOCK_C_,), padding_mode=ct.PaddingMode.ZERO)
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(BLOCK_C_,), padding_mode=ct.PaddingMode.ZERO)

    add_bf = ct.astype(ct.astype(wide, ct.float32) + ct.astype(residual, ct.float32), ct.bfloat16)
    zero_bf = ct.zeros((BLOCK_R_, BLOCK_C_), dtype=ct.bfloat16)
    le = relu_in <= zero_bf
    fill_2d = ct.reshape(fill, (1, 1))
    where_bf = ct.where(le, fill_2d, add_bf)
    where_f = ct.astype(where_bf, ct.float32)

    x_f = ct.astype(x, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C_))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C_))
    prod_coeff_2d = ct.reshape(prod_coeff, (1, BLOCK_C_))
    output_scale_2d = ct.reshape(output_scale, (1, BLOCK_C_))

    centered = x_f - mean_2d
    correction = centered * prod_coeff_2d
    res = where_f - correction - mean_term_2d
    out_f = res * output_scale_2d
    ct.store(out_ptr, index=(chunk, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="8399096d")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device

    def _nhwc_flat(t, c):
        return t.permute(0, 2, 3, 1).contiguous().view(ELEMENTS_PER_CHANNEL, c)

    wide_slice = _nhwc_flat(arg0_1, IN_CHANNELS)[:, :CHANNELS].contiguous()
    residual = _nhwc_flat(arg1_1, CHANNELS)
    relu_input = _nhwc_flat(arg2_1, CHANNELS)
    x = _nhwc_flat(arg4_1, CHANNELS)
    mean = arg5_1.view(CHANNELS)
    invstd = arg6_1
    weight = arg7_1

    def _pad_c(t):
        if t.shape[-1] == BLOCK_C_PAD:
            return t
        out = torch.zeros(t.shape[:-1] + (BLOCK_C_PAD,), device=t.device, dtype=t.dtype)
        out[..., :t.shape[-1]] = t
        return out

    wide_pad = _pad_c(wide_slice)
    residual_pad = _pad_c(residual)
    relu_pad = _pad_c(relu_input)
    x_pad = _pad_c(x)
    mean_pad = _pad_c(mean)
    invstd_pad = _pad_c(invstd)
    weight_pad = _pad_c(weight)
    fill_1d = arg3_1.reshape(1)

    partial_sum = torch.zeros((NUM_CHUNKS, BLOCK_C_PAD), device=device, dtype=torch.float32)
    partial_prod = torch.zeros((NUM_CHUNKS, BLOCK_C_PAD), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (NUM_CHUNKS, 1, 1),
        _bn_backward_partial_kernel,
        (wide_pad, residual_pad, relu_pad, fill_1d, x_pad, mean_pad,
         partial_sum, partial_prod,
         BLOCK_R, BLOCK_C_PAD),
    )

    sum_out_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    dot_out_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    vector_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    mean_term_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    prod_coeff_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)
    output_scale_pad = torch.empty((BLOCK_C_PAD,), device=device, dtype=torch.float32)

    ct.launch(
        stream, (1, 1, 1), _finalize_reduce_kernel,
        (partial_sum, partial_prod, invstd_pad, weight_pad,
         sum_out_pad, dot_out_pad, vector_pad,
         mean_term_pad, prod_coeff_pad, output_scale_pad,
         NUM_CHUNKS_PAD, BLOCK_C_PAD, REDUCE_SCALE),
    )

    out_flat = torch.empty((ELEMENTS_PER_CHANNEL, BLOCK_C_PAD), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (NUM_CHUNKS, 1, 1),
        _bn_backward_epilogue_kernel,
        (wide_pad, residual_pad, relu_pad, fill_1d, x_pad, mean_pad,
         mean_term_pad, prod_coeff_pad, output_scale_pad, out_flat,
         BLOCK_R, BLOCK_C_PAD),
    )

    sum_1 = sum_out_pad[:CHANNELS].contiguous()
    mul_8 = vector_pad[:CHANNELS].contiguous()
    out_valid = out_flat[:, :CHANNELS].contiguous()
    out_nhwc = out_valid.view(BATCH, HEIGHT, WIDTH, CHANNELS)
    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.bfloat16)
    out.copy_(out_nhwc.permute(0, 3, 1, 2))
    return sum_1, mul_8, out
