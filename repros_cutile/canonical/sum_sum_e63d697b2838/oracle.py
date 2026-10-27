"""cuTile port of sum_sum_e63d697b2838: MobileViT BN-backward with slice+add producer.

Three kernels:
  1. partial_reduce: per (row_block, channel) compute partial sums of add_value
     and add_value*(x-mean); write to partials.
  2. finalize: sum partials across chunks, compute channel-wise coefficients.
  3. epilogue: elementwise BN-backward with output written in channels-last bf16.

Inputs (channels-last):
  arg0: bf16[128, 256, 16, 16] stride [65536,1,4096,256] (wide, sliced [:,:128])
  arg1: bf16[128, 128, 16, 16] stride [32768,1,2048,128] (residual)
  arg2: bf16[128, 128, 16, 16] stride [32768,1,2048,128] (x)
  arg3: f32[1, 128, 1, 1]                                (mean)
  arg4: f32[128]                                         (invstd)
  arg5: f32[128]                                         (weight)

All shapes are powers of 2, so no masking is needed.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
IN_CHANNELS = 256
CHANNELS = 128
HEIGHT = 16
WIDTH = 16
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW  # 32768
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 3.0517578125e-05


@ct.kernel
def _partial_reduce_kernel(
    wide_ptr,       # bf16 (32768, 256) - flattened NHWC
    residual_ptr,   # bf16 (32768, 128) - flattened NHWC
    x_ptr,          # bf16 (32768, 128) - flattened NHWC
    mean_ptr,       # f32 (128,)
    partial_sum_ptr,   # f32 (num_chunks, 128)
    partial_prod_ptr,  # f32 (num_chunks, 128)
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    # Wide is (32768, 256): first 128 channels are the slice.
    lhs = ct.astype(
        ct.load(wide_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    rhs = ct.astype(
        ct.load(residual_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    add_value = ct.astype(ct.astype(lhs + rhs, ct.bfloat16), ct.float32)

    x = ct.astype(
        ct.load(x_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    centered = x - mean_2d
    prod = add_value * centered

    sum_v = ct.sum(add_value, axis=0)   # (BLOCK_C,)
    prod_v = ct.sum(prod, axis=0)       # (BLOCK_C,)

    ct.store(partial_sum_ptr, index=(r_block, c_block), tile=ct.reshape(sum_v, (1, BLOCK_C)))
    ct.store(partial_prod_ptr, index=(r_block, c_block), tile=ct.reshape(prod_v, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 (num_chunks, 128)
    partial_prod_ptr,  # f32 (num_chunks, 128)
    invstd_ptr,        # f32 (128,)
    weight_ptr,        # f32 (128,)
    sum_out_ptr,       # f32 (128,)
    prod_scaled_out_ptr,  # f32 (128,)
    mean_term_ptr,     # f32 (128,)
    prod_coeff_ptr,    # f32 (128,)
    output_scale_ptr,  # f32 (128,)
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    REDUCE_SCALE_: ct.Constant[float],
):
    c_block = ct.bid(0)

    # Sum partials across chunks.
    total_sum = ct.zeros((BLOCK_C,), dtype=ct.float32)
    total_prod = ct.zeros((BLOCK_C,), dtype=ct.float32)
    for chunk in range(NUM_CHUNKS):
        s = ct.load(partial_sum_ptr, index=(chunk, c_block), shape=(1, BLOCK_C))
        p = ct.load(partial_prod_ptr, index=(chunk, c_block), shape=(1, BLOCK_C))
        total_sum = total_sum + ct.reshape(s, (BLOCK_C,))
        total_prod = total_prod + ct.reshape(p, (BLOCK_C,))

    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))

    mean_term = total_sum * REDUCE_SCALE_
    prod_scaled = total_prod * REDUCE_SCALE_
    prod_coeff = prod_scaled * (invstd * invstd)
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=total_sum)
    ct.store(prod_scaled_out_ptr, index=(c_block,), tile=total_prod * invstd)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(c_block,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    wide_ptr,       # bf16 (32768, 256)
    residual_ptr,   # bf16 (32768, 128)
    x_ptr,          # bf16 (32768, 128)
    mean_ptr,       # f32 (128,)
    mean_term_ptr,  # f32 (128,)
    prod_coeff_ptr, # f32 (128,)
    output_scale_ptr, # f32 (128,)
    out_ptr,        # bf16 (32768, 128) - channels-last output
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    lhs = ct.astype(
        ct.load(wide_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    rhs = ct.astype(
        ct.load(residual_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    add_value = ct.astype(ct.astype(lhs + rhs, ct.bfloat16), ct.float32)

    x = ct.astype(
        ct.load(x_ptr, index=(r_block, c_block), shape=(BLOCK_R, BLOCK_C)),
        ct.float32,
    )
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    prod_coeff = ct.load(prod_coeff_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_term = ct.load(mean_term_ptr, index=(c_block,), shape=(BLOCK_C,))
    output_scale = ct.load(output_scale_ptr, index=(c_block,), shape=(BLOCK_C,))
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    prod_coeff_2d = ct.reshape(prod_coeff, (1, BLOCK_C))
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    output_scale_2d = ct.reshape(output_scale, (1, BLOCK_C))

    centered = x - mean_2d
    correction = centered * prod_coeff_2d
    residual = add_value - correction - mean_term_2d
    out_value = residual * output_scale_2d
    ct.store(out_ptr, index=(r_block, c_block), tile=ct.astype(out_value, ct.bfloat16))


@oracle_impl(hardware="B200", point="809435bf", BLOCK_R=1024, BLOCK_C=128)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int):
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs
    device = arg0.device

    # arg0..arg2 are channels-last NCHW.  Their "flattened NHWC" view is
    # (N*H*W, C) with contiguous rows over C. We can reshape via as_strided.
    # For (128, 256, 16, 16) with stride (65536, 1, 4096, 256): the natural
    # NHWC ordering is (N, H, W, C). The linear layout is:
    #   offset(n, c, h, w) = n*65536 + c + h*4096 + w*256
    # Which means over (n, h, w, c) traversal (c varies fastest, then w, then h, then n):
    # linear index = n*(H*W*C) + h*(W*C) + w*C + c
    # so the flat NHWC view is contiguous. As_strided to (N*H*W, C):
    wide_flat = arg0.as_strided((BATCH * HW, IN_CHANNELS), (IN_CHANNELS, 1))
    residual_flat = arg1.as_strided((BATCH * HW, CHANNELS), (CHANNELS, 1))
    x_flat = arg2.as_strided((BATCH * HW, CHANNELS), (CHANNELS, 1))

    # arg3 is f32[1,128,1,1] view as (128,)
    mean_1d = arg3.view(CHANNELS)

    num_chunks = ELEMENTS_PER_CHANNEL // BLOCK_R  # exact, no cdiv since pow-2
    num_c_blocks = CHANNELS // BLOCK_C            # exact

    partial_sum = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    partial_prod = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_scaled_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    mean_term = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    output_scale = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    out = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, 1, CHANNELS * WIDTH, CHANNELS),
        device=device,
        dtype=torch.bfloat16,
    )
    out_flat = out.as_strided((BATCH * HW, CHANNELS), (CHANNELS, 1))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_c_blocks, num_chunks, 1),
        _partial_reduce_kernel,
        (wide_flat, residual_flat, x_flat, mean_1d,
         partial_sum, partial_prod, BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, arg4, arg5,
         sum_out, prod_scaled_out, mean_term, prod_coeff, output_scale,
         num_chunks, BLOCK_C, REDUCE_SCALE),
    )
    ct.launch(
        stream,
        (num_c_blocks, num_chunks, 1),
        _epilogue_kernel,
        (wide_flat, residual_flat, x_flat, mean_1d,
         mean_term, prod_coeff, output_scale, out_flat,
         BLOCK_R, BLOCK_C),
    )
    return sum_out, prod_scaled_out, out
