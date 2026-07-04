"""cuTile port of sum_sum_7f5745d75e84: GhostNet BN-backward.

Mirrors Triton's 3-kernel split-K structure:
  1. partial_reduce: adds bf16 inputs, materializes bf16 residual add, emits
     per-channel-block partials of `sum` and `sum(centered)`.
  2. finalize: sums partials into per-channel scalars, precomputes BN
     epilogue coefficients.
  3. epilogue: applies coefficients to write bf16 gradient tensor.

Reductions live inside `@ct.kernel` via `ct.sum(...)` — no torch `.sum(...)`
in oracle_forward. BLOCK_R=1024, BLOCK_C=16, BLOCK_E=512 match Triton.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
IN_CHANNELS = 160
CHANNELS = 80
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH  # 49
ELEMENTS_PER_CHANNEL = BATCH * HW  # 25088
NUMEL = ELEMENTS_PER_CHANNEL * CHANNELS
REDUCE_SCALE = 3.985969387755102e-05


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    arg0_ptr,               # bf16 [ELEMENTS_PER_CHANNEL, IN_CHANNELS]
    arg1_ptr,               # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    arg2_ptr,               # bf16 [ELEMENTS_PER_CHANNEL, CHANNELS]
    arg3_ptr,               # f32  [CHANNELS]
    partial_sum_ptr,        # f32  [num_chunks, CHANNELS]
    partial_prod_ptr,       # f32  [num_chunks, CHANNELS]
    ELEMENTS_PER_CHANNEL_C: ct.Constant[int],
    CHANNELS_C: ct.Constant[int],
    IN_CHANNELS_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    r_block = ct.bid(1)

    lhs_bf = ct.load(arg0_ptr, index=(r_block, c_block),
                     shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    rhs_bf = ct.load(arg1_ptr, index=(r_block, c_block),
                     shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    lhs = ct.astype(lhs_bf, ct.float32)
    rhs = ct.astype(rhs_bf, ct.float32)
    add_bf16 = ct.astype(lhs + rhs, ct.bfloat16)
    add_value = ct.astype(add_bf16, ct.float32)

    arg2_bf = ct.load(arg2_ptr, index=(r_block, c_block),
                      shape=(BLOCK_R, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    arg2_value = ct.astype(arg2_bf, ct.float32)
    mean = ct.load(arg3_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    centered = arg2_value - mean_2d
    prod = add_value * centered

    r_idx = ct.arange(BLOCK_R, dtype=ct.int32) + r_block * BLOCK_R
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    r_valid = ct.reshape(r_idx < ELEMENTS_PER_CHANNEL_C, (BLOCK_R, 1))
    c_valid = ct.reshape(c_idx < CHANNELS_C, (1, BLOCK_C))
    valid = r_valid & c_valid
    zero_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    add_masked = ct.where(valid, add_value, zero_f)
    prod_masked = ct.where(valid, prod, zero_f)

    p_sum = ct.sum(add_masked, axis=0, keepdims=True)   # (1, BLOCK_C)
    p_prod = ct.sum(prod_masked, axis=0, keepdims=True) # (1, BLOCK_C)
    ct.store(partial_sum_ptr, index=(r_block, c_block), tile=p_sum)
    ct.store(partial_prod_ptr, index=(r_block, c_block), tile=p_prod)


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,    # f32 [num_chunks, CHANNELS]
    partial_prod_ptr,   # f32 [num_chunks, CHANNELS]
    arg4_ptr,           # f32 [CHANNELS]
    arg5_ptr,           # f32 [CHANNELS]
    sum_out_ptr,        # f32 [CHANNELS]
    mean_term_ptr,      # f32 [CHANNELS]
    prod_coeff_ptr,     # f32 [CHANNELS]
    output_scale_ptr,   # f32 [CHANNELS]
    scale_grad_ptr,     # f32 [CHANNELS]
    CHANNELS_C: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    p_sum = ct.load(partial_sum_ptr, index=(0, c_block),
                    shape=(BLOCK_CHUNKS, BLOCK_C),
                    padding_mode=ct.PaddingMode.ZERO)
    p_prod = ct.load(partial_prod_ptr, index=(0, c_block),
                     shape=(BLOCK_CHUNKS, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    chunk_idx = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_valid = ct.reshape(chunk_idx < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    zero_f = ct.zeros((BLOCK_CHUNKS, BLOCK_C), dtype=ct.float32)
    p_sum_v = ct.where(chunk_valid, p_sum, zero_f)
    p_prod_v = ct.where(chunk_valid, p_prod, zero_f)
    sum_value = ct.sum(p_sum_v, axis=0)
    prod_value = ct.sum(p_prod_v, axis=0)

    invstd = ct.load(arg4_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    affine = ct.load(arg5_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
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
    arg0_ptr,           # bf16 [numel]  (channels-last flat NHWC)
    arg1_ptr,           # bf16 [numel]
    arg2_ptr,           # bf16 [numel]
    arg3_ptr,           # f32  [CHANNELS]
    mean_term_ptr,      # f32  [CHANNELS]
    prod_coeff_ptr,     # f32  [CHANNELS]
    output_scale_ptr,   # f32  [CHANNELS]
    out_ptr,            # bf16 [numel]
    CHANNELS_C: ct.Constant[int],
    IN_CHANNELS_C: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
):
    pid = ct.bid(0)
    idxs = ct.arange(BLOCK_E, dtype=ct.int32) + pid * BLOCK_E
    channel = idxs - (idxs // CHANNELS_C) * CHANNELS_C
    reduce_index = idxs // CHANNELS_C
    wide_offsets = reduce_index * IN_CHANNELS_C + channel

    lhs = ct.astype(ct.gather(arg0_ptr, wide_offsets), ct.float32)
    rhs = ct.astype(ct.load(arg1_ptr, index=(pid,), shape=(BLOCK_E,)),
                    ct.float32)
    add_value = ct.astype(ct.astype(lhs + rhs, ct.bfloat16), ct.float32)

    arg2_value = ct.astype(ct.load(arg2_ptr, index=(pid,), shape=(BLOCK_E,)),
                           ct.float32)
    mean = ct.gather(arg3_ptr, channel)
    centered = arg2_value - mean

    prod_coeff = ct.gather(prod_coeff_ptr, channel)
    correction = centered * prod_coeff
    residual = add_value - correction

    mean_term = ct.gather(mean_term_ptr, channel)
    residual = residual - mean_term

    output_scale = ct.gather(output_scale_ptr, channel)
    out_value = residual * output_scale
    ct.store(out_ptr, index=(pid,),
             tile=ct.astype(out_value, ct.bfloat16))


@oracle_impl(hardware="B200", point="ffcd6c5b", BLOCK_R=1024, BLOCK_C=16, BLOCK_E=512)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, BLOCK_E: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device

    # arg0: bf16[512,160,7,7] channels-last (wide)
    # arg1, arg2: bf16[512,80,7,7] channels-last (narrow)
    # arg3: f32[1,80,1,1]; arg4, arg5: f32[80]
    # Layout: reduce index is (batch, h, w) walking in NHWC order.
    # For a channels-last tensor, permute(0,2,3,1).contiguous() gives NHWC-flat.
    arg0_flat = arg0_1.permute(0, 2, 3, 1).contiguous().view(-1)
    arg1_flat = arg1_1.permute(0, 2, 3, 1).contiguous().view(-1)
    arg2_flat = arg2_1.permute(0, 2, 3, 1).contiguous().view(-1)
    arg3_flat = arg3_1.contiguous().view(CHANNELS)
    arg4_flat = arg4_1.contiguous().view(CHANNELS)
    arg5_flat = arg5_1.contiguous().view(CHANNELS)

    # Reshape into 2D [ELEMENTS_PER_CHANNEL, CHANNELS/IN_CHANNELS]
    arg0_2d = arg0_flat.view(ELEMENTS_PER_CHANNEL, IN_CHANNELS)
    arg1_2d = arg1_flat.view(ELEMENTS_PER_CHANNEL, CHANNELS)
    arg2_2d = arg2_flat.view(ELEMENTS_PER_CHANNEL, CHANNELS)

    num_chunks = (ELEMENTS_PER_CHANNEL + BLOCK_R - 1) // BLOCK_R
    block_chunks = _next_power_of_2(num_chunks)

    partial_sum = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    partial_prod = torch.empty((num_chunks, CHANNELS), device=device, dtype=torch.float32)
    sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    mean_term = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    prod_coeff = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    output_scale = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    num_c_blocks = (CHANNELS + BLOCK_C - 1) // BLOCK_C
    ct.launch(
        stream,
        (num_c_blocks, num_chunks, 1),
        _partial_reduce_kernel,
        (arg0_2d, arg1_2d, arg2_2d, arg3_flat,
         partial_sum, partial_prod,
         ELEMENTS_PER_CHANNEL, CHANNELS, IN_CHANNELS,
         BLOCK_R, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_prod, arg4_flat, arg5_flat,
         sum_out, mean_term, prod_coeff, output_scale, scale_grad,
         CHANNELS, num_chunks, block_chunks, BLOCK_C),
    )
    # Output shape: (N, CHANNELS, H, W) channels-last stride.
    # For the epilogue we lay out out_flat as NHWC-flat (which is the
    # channels-last physical layout) with total elements = NUMEL.
    out_flat = torch.empty(NUMEL, device=device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        ((NUMEL + BLOCK_E - 1) // BLOCK_E, 1, 1),
        _epilogue_kernel,
        (arg0_flat, arg1_flat, arg2_flat, arg3_flat,
         mean_term, prod_coeff, output_scale, out_flat,
         CHANNELS, IN_CHANNELS, BLOCK_E),
    )
    out_bf = out_flat.view(BATCH, HEIGHT, WIDTH, CHANNELS).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last)

    return sum_out, scale_grad, out_bf
