"""cuTile port of sum_sum_3104a9ce2ada: MobileViT BN-backward.

Matches Triton's 3-kernel structure:
  1. `_partial_reduce_kernel` — per-chunk partial reductions of `slice+add`
     and `(slice+add) * (arg2 - mean)`.
  2. `_finalize_kernel` — reduce partials and compute derived channel scalars.
  3. `_epilogue_kernel` — per-element BN-backward output.

The producer (slice arg0 + arg1, bf16-rounded) and reductions are all
performed inside the cuTile kernels.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
IN_CHANNELS = 192
CHANNELS = 96
HEIGHT = 32
WIDTH = 32
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW  # 128 * 1024 = 131072
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 7.62939453125e-06


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    wide_ptr,          # bf16 (ELEMENTS_PER_CHANNEL, IN_CHANNELS) channels-last flatten
    residual_ptr,      # bf16 (ELEMENTS_PER_CHANNEL, CHANNELS) channels-last flatten (arg1)
    x_ptr,             # bf16 (ELEMENTS_PER_CHANNEL, CHANNELS) channels-last flatten (arg2)
    mean_ptr,          # f32  (CHANNELS,)
    partial_sum_ptr,   # f32  (NUM_CHUNKS, CHANNELS)
    partial_prod_ptr,  # f32  (NUM_CHUNKS, CHANNELS)
    ELEMENTS_PER_CHANNEL_: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    IN_CHANNELS_: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    # Load the wide input (arg0) with IN_CHANNELS stride, then take first CHANNELS.
    wide = ct.load(wide_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)

    wide_f = ct.astype(wide, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    # bf16 add: cast to f32, add, round to bf16, back to f32
    add_bf = ct.astype(wide_f + residual_f, ct.bfloat16)
    add_value = ct.astype(add_bf, ct.float32)
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    centered = x_f - mean_2d
    prod = add_value * centered

    # Row-mask for tail chunk.
    r_idx = ct.arange(BLOCK_R, dtype=ct.int32) + r_block * BLOCK_R
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    r_valid = ct.reshape(r_idx < ELEMENTS_PER_CHANNEL_, (BLOCK_R, 1))
    c_valid = ct.reshape(c_idx < CHANNELS_, (1, BLOCK_C))
    valid = r_valid & c_valid
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    add_m = ct.where(valid, add_value, zero_f)
    prod_m = ct.where(valid, prod, zero_f)

    p_sum = ct.sum(add_m, axis=0, keepdims=True)
    p_dot = ct.sum(prod_m, axis=0, keepdims=True)
    ct.store(partial_sum_ptr, index=(r_block, c_block), tile=p_sum)
    ct.store(partial_prod_ptr, index=(r_block, c_block), tile=p_dot)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,     # f32 (NUM_CHUNKS, CHANNELS)
    partial_prod_ptr,    # f32 (NUM_CHUNKS, CHANNELS)
    invstd_ptr,          # f32 (CHANNELS,)
    weight_ptr,          # f32 (CHANNELS,)
    sum_out_ptr,         # f32 (CHANNELS,)
    prod_scaled_out_ptr, # f32 (CHANNELS,) — dot_value * invstd
    mean_term_ptr,       # f32 (CHANNELS,) — sum * SCALE
    prod_coeff_ptr,      # f32 (CHANNELS,) — dot * SCALE * invstd^2
    output_scale_ptr,    # f32 (CHANNELS,) — invstd * weight
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    CHANNELS_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c_block = ct.bid(0)
    p_sum = ct.load(partial_sum_ptr, index=(0, c_block),
                    shape=(BLOCK_CHUNKS, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    p_dot = ct.load(partial_prod_ptr, index=(0, c_block),
                    shape=(BLOCK_CHUNKS, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    t_idx = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    t_valid = ct.reshape(t_idx < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    c_valid = ct.reshape(c_idx < CHANNELS_, (1, BLOCK_C))
    valid = t_valid & c_valid
    zero_f = ct.zeros((BLOCK_CHUNKS, BLOCK_C), dtype=ct.float32)
    p_sum = ct.where(valid, p_sum, zero_f)
    p_dot = ct.where(valid, p_dot, zero_f)

    sum_value = ct.sum(p_sum, axis=0)  # (BLOCK_C,)
    dot_value = ct.sum(p_dot, axis=0)

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    mean_term = sum_value * SCALE_
    prod_scaled = dot_value * SCALE_
    invstd_sq = invstd * invstd
    prod_coeff = prod_scaled * invstd_sq
    output_scale = invstd * weight
    prod_scaled_out = dot_value * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(prod_scaled_out_ptr, index=(c_block,), tile=prod_scaled_out)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(c_block,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    wide_ptr,           # bf16 (ELEMENTS_PER_CHANNEL, IN_CHANNELS)
    residual_ptr,       # bf16 (ELEMENTS_PER_CHANNEL, CHANNELS)
    x_ptr,              # bf16 (ELEMENTS_PER_CHANNEL, CHANNELS)
    mean_ptr,           # f32  (CHANNELS,)
    mean_term_ptr,      # f32  (CHANNELS,)
    prod_coeff_ptr,     # f32  (CHANNELS,)
    output_scale_ptr,   # f32  (CHANNELS,)
    out_ptr,            # bf16 (ELEMENTS_PER_CHANNEL, CHANNELS)
    BLOCK_C: ct.Constant[int],
):
    pid = ct.bid(0)
    wide = ct.load(wide_ptr, index=(pid, 0), shape=(1, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(pid, 0), shape=(1, BLOCK_C),
                       padding_mode=ct.PaddingMode.ZERO)
    x = ct.load(x_ptr, index=(pid, 0), shape=(1, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    wide_f = ct.astype(wide, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    add_bf = ct.astype(wide_f + residual_f, ct.bfloat16)
    add_value = ct.astype(add_bf, ct.float32)
    x_f = ct.astype(x, ct.float32)

    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    prod_coeff = ct.load(prod_coeff_ptr, index=(0,), shape=(BLOCK_C,),
                         padding_mode=ct.PaddingMode.ZERO)
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(BLOCK_C,),
                           padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    prod_coeff_2d = ct.reshape(prod_coeff, (1, BLOCK_C))
    output_scale_2d = ct.reshape(output_scale, (1, BLOCK_C))

    centered = x_f - mean_2d
    correction = centered * prod_coeff_2d
    residual_val = add_value - correction - mean_term_2d
    out = residual_val * output_scale_2d
    ct.store(out_ptr, index=(pid, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="5096a4b4", BLOCK_CH=128)
def oracle_forward(inputs, *, BLOCK_CH: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    # arg0: bf16 [N, IN_CHANNELS, H, W] channels-last (stride 196608, 1, 6144, 192)
    # arg1: bf16 [N, CHANNELS, H, W]    channels-last (98304, 1, 3072, 96)
    # arg2: bf16 [N, CHANNELS, H, W]    channels-last (98304, 1, 3072, 96)
    # arg3: f32  [1, CHANNELS, 1, 1] mean
    # arg4: f32  [CHANNELS,] invstd
    # arg5: f32  [CHANNELS,] weight
    # Channels-last permutation: (N, H, W, C).
    arg0_nhwc = arg0_1.permute(0, 2, 3, 1).contiguous()  # [N, H, W, IN_CHANNELS]
    arg1_nhwc = arg1_1.permute(0, 2, 3, 1).contiguous()  # [N, H, W, CHANNELS]
    arg2_nhwc = arg2_1.permute(0, 2, 3, 1).contiguous()  # [N, H, W, CHANNELS]

    wide_2d = arg0_nhwc.view(ELEMENTS_PER_CHANNEL, IN_CHANNELS)
    residual_2d = arg1_nhwc.view(ELEMENTS_PER_CHANNEL, CHANNELS)
    x_2d = arg2_nhwc.view(ELEMENTS_PER_CHANNEL, CHANNELS)
    mean_1d = arg3_1.view(CHANNELS).contiguous()
    invstd_1d = arg4_1.contiguous()
    weight_1d = arg5_1.contiguous()

    BLOCK_R = 1024
    BLOCK_C = 16
    num_chunks = (ELEMENTS_PER_CHANNEL + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_power_of_2(num_chunks)
    num_c_blocks = (CHANNELS + BLOCK_C - 1) // BLOCK_C

    partial_sum = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    partial_prod = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_scaled_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    mean_term = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    output_scale = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_c_blocks, num_chunks, 1),
        _partial_reduce_kernel,
        (wide_2d, residual_2d, x_2d, mean_1d, partial_sum, partial_prod,
         ELEMENTS_PER_CHANNEL, CHANNELS, IN_CHANNELS, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream, (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, invstd_1d, weight_1d,
         sum_out, prod_scaled_out, mean_term, prod_coeff, output_scale,
         num_chunks, block_chunks, CHANNELS, BLOCK_C, REDUCE_SCALE),
    )

    out_nhwc = torch.empty((BATCH, HEIGHT, WIDTH, CHANNELS), device=device, dtype=torch.bfloat16)
    out_2d = out_nhwc.view(ELEMENTS_PER_CHANNEL, CHANNELS)
    # Use BLOCK_CH for the epilogue tile width. CHANNELS=96 doesn't divide
    # BLOCK_CH=128, so we launch one row per program with a padded load.
    ct.launch(
        stream, (ELEMENTS_PER_CHANNEL, 1, 1),
        _epilogue_kernel,
        (wide_2d, residual_2d, x_2d, mean_1d,
         mean_term, prod_coeff, output_scale, out_2d, BLOCK_CH),
    )

    out_cl = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, 1, CHANNELS * WIDTH, CHANNELS),
        device=device, dtype=torch.bfloat16,
    )
    out_cl.copy_(out_nhwc.permute(0, 3, 1, 2))
    return sum_out, prod_scaled_out, out_cl
