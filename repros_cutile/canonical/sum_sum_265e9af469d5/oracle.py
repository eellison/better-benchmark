"""cuTile port of sum_sum_265e9af469d5: GhostNet BN-backward cooperative split-K.

Matches Triton's three-kernel plan:
1. `_partial_reduce_kernel`: 2D grid (cdiv(CHANNELS, BLOCK_C), num_chunks).
   For each (c_block, chunk), loads a (BLOCK_R, BLOCK_C) tile of the
   channels-last flat inputs, applies the mask + where + bf16 rounding
   producer, and stores partial_sum and partial_dot rows.
2. `_finalize_kernel`: grid (cdiv(CHANNELS, BLOCK_C),). Reduces partials
   across chunks and computes stats + scale_grad.
3. `_epilogue_kernel`: 1D grid over all NUMEL elements. Recomputes the same
   producer, then applies BN backward using the finalized stats and writes
   the bf16 output.

The channels-last physical layout is treated as flat 2D
(ELEMENTS_PER_CHANNEL, CHANNELS_or_IN_CHANNELS).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
IN_CHANNELS = 672
CHANNELS = 336
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW  # 100352
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 9.964923469387754e-06
OUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (CHANNELS * HW, 1, CHANNELS * WIDTH, CHANNELS)


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    arg0_ptr,           # bf16 [ELEMENTS_PER_CHANNEL, IN_CHANNELS]
    arg1_ptr,           # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    arg2_ptr,           # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    arg3_ptr,           # bf16 [1]
    arg4_ptr,           # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    arg5_ptr,           # f32 [CHANNELS]
    partial_sum_ptr,    # f32 [num_chunks, CHANNELS]
    partial_prod_ptr,   # f32 [num_chunks, CHANNELS]
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    lhs = ct.load(arg0_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C))
    rhs = ct.load(arg1_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C))
    lhs_f = ct.astype(lhs, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    add_bf = ct.astype(lhs_f + rhs_f, ct.bfloat16)
    add_value = ct.astype(add_bf, ct.float32)

    mask_source = ct.astype(
        ct.load(arg2_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    fill_bf = ct.load(arg3_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_bf, ct.float32)
    fill_broadcast = ct.reshape(fill_f, (1, 1)) + ct.zeros(
        (BLOCK_R, BLOCK_C), dtype=ct.float32
    )
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    where_value = ct.where(mask_source <= zero_f, fill_broadcast, add_value)

    arg4_val = ct.astype(
        ct.load(arg4_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    mean_1d = ct.load(arg5_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_2d = ct.reshape(mean_1d, (1, BLOCK_C)) + ct.zeros(
        (BLOCK_R, BLOCK_C), dtype=ct.float32
    )
    centered = arg4_val - mean_2d
    prod = where_value * centered

    partial_sum = ct.sum(where_value, axis=0)
    partial_prod = ct.sum(prod, axis=0)
    ct.store(
        partial_sum_ptr, index=(r_block, c_block),
        tile=ct.reshape(partial_sum, (1, BLOCK_C)),
    )
    ct.store(
        partial_prod_ptr, index=(r_block, c_block),
        tile=ct.reshape(partial_prod, (1, BLOCK_C)),
    )


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,     # f32 [num_chunks, CHANNELS]
    partial_prod_ptr,    # f32 [num_chunks, CHANNELS]
    arg6_ptr,            # f32 [CHANNELS]
    arg7_ptr,            # f32 [CHANNELS]
    sum_out_ptr,         # f32 [CHANNELS]
    mean_term_ptr,       # f32 [CHANNELS]
    prod_coeff_ptr,      # f32 [CHANNELS]
    output_scale_ptr,    # f32 [CHANNELS]
    scale_grad_ptr,      # f32 [CHANNELS]
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    c = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    chunks_2d = ct.reshape(chunks, (BLOCK_CHUNKS, 1))
    c_2d = ct.reshape(c, (1, BLOCK_C))
    NC_full = ct.full((BLOCK_CHUNKS, 1), NUM_CHUNKS, dtype=ct.int32)
    C_full = ct.full((1, BLOCK_C), CHANNELS, dtype=ct.int32)
    mask = (chunks_2d < NC_full) & (c_2d < C_full)
    zero_i2d = ct.zeros((BLOCK_CHUNKS, BLOCK_C), dtype=ct.int32)
    zero_f2d = ct.zeros((BLOCK_CHUNKS, BLOCK_C), dtype=ct.float32)

    chunks_full = chunks_2d + zero_i2d
    c_full = c_2d + zero_i2d
    chunks_safe = ct.where(mask, chunks_full, zero_i2d)
    c_safe = ct.where(mask, c_full, zero_i2d)

    partial_sum = ct.gather(partial_sum_ptr, (chunks_safe, c_safe), mask=mask)
    partial_prod = ct.gather(partial_prod_ptr, (chunks_safe, c_safe), mask=mask)
    partial_sum = ct.where(mask, partial_sum, zero_f2d)
    partial_prod = ct.where(mask, partial_prod, zero_f2d)
    sum_value = ct.sum(partial_sum, axis=0)
    prod_value = ct.sum(partial_prod, axis=0)

    invstd = ct.load(arg6_ptr, index=(c_block,), shape=(BLOCK_C,))
    affine = ct.load(arg7_ptr, index=(c_block,), shape=(BLOCK_C,))

    mean_term = sum_value * REDUCE_SCALE
    prod_scaled = prod_value * REDUCE_SCALE
    invstd_sq = invstd * invstd
    prod_coeff = prod_scaled * invstd_sq
    output_scale = invstd * affine
    scale_grad = prod_value * invstd

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(c_block,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)
    ct.store(scale_grad_ptr, index=(c_block,), tile=scale_grad)


@ct.kernel
def _epilogue_kernel(
    arg0_ptr,           # bf16 flat storage
    arg1_ptr,           # bf16 flat storage
    arg2_ptr,           # bf16 flat storage
    arg3_ptr,           # bf16 [1]
    arg4_ptr,           # bf16 flat storage
    arg5_ptr,           # f32 [CHANNELS]
    mean_term_ptr,      # f32 [CHANNELS]
    prod_coeff_ptr,     # f32 [CHANNELS]
    output_scale_ptr,   # f32 [CHANNELS]
    out_ptr,            # bf16 flat storage
    BLOCK_E: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK_E + ct.arange(BLOCK_E, dtype=ct.int64)
    NUMEL_full = ct.full((BLOCK_E,), NUMEL, dtype=ct.int64)
    mask = offsets < NUMEL_full
    zero_i = ct.zeros((BLOCK_E,), dtype=ct.int64)
    safe_off = ct.where(mask, offsets, zero_i)

    C_full = ct.full((BLOCK_E,), CHANNELS, dtype=ct.int64)
    IN_C_full = ct.full((BLOCK_E,), IN_CHANNELS, dtype=ct.int64)
    channel = safe_off - (safe_off // C_full) * C_full
    reduce_index = safe_off // C_full
    wide_offsets = reduce_index * IN_C_full + channel

    lhs = ct.astype(ct.gather(arg0_ptr, wide_offsets), ct.float32)
    rhs = ct.astype(ct.gather(arg1_ptr, safe_off), ct.float32)
    add_bf = ct.astype(lhs + rhs, ct.bfloat16)
    add_value = ct.astype(add_bf, ct.float32)

    mask_source = ct.astype(ct.gather(arg2_ptr, safe_off), ct.float32)
    fill_bf = ct.load(arg3_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_bf, ct.float32)
    fill_broadcast = ct.reshape(fill_f, (1,)) + ct.zeros(
        (BLOCK_E,), dtype=ct.float32
    )
    zero_f = ct.zeros((BLOCK_E,), dtype=ct.float32)
    where_value = ct.where(mask_source <= zero_f, fill_broadcast, add_value)

    arg4_value = ct.astype(ct.gather(arg4_ptr, safe_off), ct.float32)
    channel_i32 = ct.astype(channel, ct.int32)
    mean = ct.gather(arg5_ptr, channel_i32)
    centered = arg4_value - mean

    prod_coeff = ct.gather(prod_coeff_ptr, channel_i32)
    correction = centered * prod_coeff
    residual = where_value - correction

    mean_term = ct.gather(mean_term_ptr, channel_i32)
    residual = residual - mean_term

    output_scale = ct.gather(output_scale_ptr, channel_i32)
    out_value = residual * output_scale
    out_bf = ct.astype(out_value, ct.bfloat16)
    ct.scatter(out_ptr, offsets, out_bf, mask=mask)


@oracle_impl(hardware="B200", point="dc9eccc3", BLOCK_R=512, BLOCK_C=16, BLOCK_E=512)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, BLOCK_E: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    device = arg0_1.device

    # 2D flat views (channels-last physical layout — element order (n,h,w,c)).
    arg0_2d = torch.as_strided(arg0_1, (ELEMENTS_PER_CHANNEL, IN_CHANNELS), (IN_CHANNELS, 1))
    arg1_2d = torch.as_strided(arg1_1, (ELEMENTS_PER_CHANNEL, CHANNELS), (CHANNELS, 1))
    arg2_2d = torch.as_strided(arg2_1, (ELEMENTS_PER_CHANNEL, CHANNELS), (CHANNELS, 1))
    arg4_2d = torch.as_strided(arg4_1, (ELEMENTS_PER_CHANNEL, CHANNELS), (CHANNELS, 1))

    num_chunks = ELEMENTS_PER_CHANNEL // BLOCK_R  # 100352 // 512 = 196
    block_chunks = _next_pow2(num_chunks)
    partial_sum = torch.empty_strided(
        (num_chunks, CHANNELS), (CHANNELS, 1), device=device, dtype=torch.float32,
    )
    partial_prod = torch.empty_like(partial_sum)
    sum_out = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    output_scale = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.bfloat16)

    arg5_1d = arg5_1.view(CHANNELS)
    arg6_1d = arg6_1.view(CHANNELS)
    arg7_1d = arg7_1.view(CHANNELS)
    arg3_1d = arg3_1.view(1)

    # 1D flat views (physical storage).
    a0_flat = torch.as_strided(arg0_1, (arg0_1.untyped_storage().size() // 2,), (1,))
    a1_flat = torch.as_strided(arg1_1, (arg1_1.untyped_storage().size() // 2,), (1,))
    a2_flat = torch.as_strided(arg2_1, (arg2_1.untyped_storage().size() // 2,), (1,))
    a4_flat = torch.as_strided(arg4_1, (arg4_1.untyped_storage().size() // 2,), (1,))
    out_flat = torch.as_strided(out, (out.untyped_storage().size() // 2,), (1,))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (CHANNELS // BLOCK_C, num_chunks, 1), _partial_reduce_kernel,
        (arg0_2d, arg1_2d, arg2_2d, arg3_1d, arg4_2d, arg5_1d,
         partial_sum, partial_prod, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream, (CHANNELS // BLOCK_C, 1, 1), _finalize_kernel,
        (partial_sum, partial_prod, arg6_1d, arg7_1d,
         sum_out, mean_term, prod_coeff, output_scale, scale_grad,
         num_chunks, block_chunks, BLOCK_C),
    )
    ct.launch(
        stream, (ct.cdiv(NUMEL, BLOCK_E), 1, 1), _epilogue_kernel,
        (a0_flat, a1_flat, a2_flat, arg3_1d, a4_flat, arg5_1d,
         mean_term, prod_coeff, output_scale, out_flat, BLOCK_E),
    )

    return sum_out, scale_grad, out
