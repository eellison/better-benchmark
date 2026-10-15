"""cuTile port of sum_sum_f4b82dbfbec5: EfficientNet BN-backward tail with
SiLU-derivative masked producer.

Matches Triton's 3-kernel structure:
  1. _producer_partials_kernel: crop + BN forward + SiLU-derivative producer;
     per (row_block, c_block) computes partial per-channel sums.
  2. _finalize_kernel: reduce partials across row_blocks per channel.
  3. _bn_backward_epilogue_kernel: recompute producer, apply BN backward.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BN_SCALE = 6.228077168367346e-07


@ct.function
def _sigmoid(x):
    return 1.0 / (1.0 + ct.exp(-x))


@ct.kernel
def _producer_partials_kernel(
    crop_ptr,          # bf16 (n_rows, C)
    act_ptr,           # bf16 (n_rows, C)
    mean_ptr,          # f32 (C,)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)
    bias_ptr,          # f32 (C,)
    partial_sum_ptr,   # f32 (num_row_blocks, C)
    partial_dot_ptr,   # f32 (num_row_blocks, C)
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    r = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(act_ptr, index=(r, c), shape=(BLOCK_HW, BLOCK_C))
    crop = ct.load(crop_ptr, index=(r, c), shape=(BLOCK_HW, BLOCK_C))
    mean = ct.load(mean_ptr, index=(c,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c,), shape=(BLOCK_C,))

    x_f = ct.astype(x, ct.float32)
    crop_f = ct.astype(crop, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))

    centered = x_f - mean_2d
    affine = centered * invstd_2d * weight_2d + bias_2d
    affine_bf = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    sig = _sigmoid(affine_bf)
    producer = crop_f * sig * (affine_bf * (1.0 - sig) + 1.0)
    producer_bf = ct.astype(ct.astype(producer, ct.bfloat16), ct.float32)

    sum_v = ct.sum(producer_bf, axis=0)  # (BLOCK_C,)
    dot_v = ct.sum(producer_bf * centered, axis=0)  # (BLOCK_C,)
    ct.store(partial_sum_ptr, index=(r, c), tile=ct.reshape(sum_v, (1, BLOCK_C)))
    ct.store(partial_dot_ptr, index=(r, c), tile=ct.reshape(dot_v, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,     # f32 (num_row_blocks, C)
    partial_dot_ptr,     # f32 (num_row_blocks, C)
    invstd_ptr,          # f32 (C,)
    sum_out_ptr,         # f32 (C,) raw sum
    dot_out_ptr,         # f32 (C,) raw dot sum
    weight_grad_out_ptr, # f32 (C,) dot * invstd
    NUM_ROW_BLOCKS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c = ct.bid(0)
    total_sum = ct.zeros((BLOCK_C,), dtype=ct.float32)
    total_dot = ct.zeros((BLOCK_C,), dtype=ct.float32)
    for t in range(NUM_ROW_BLOCKS):
        s = ct.load(partial_sum_ptr, index=(t, c), shape=(1, BLOCK_C))
        d = ct.load(partial_dot_ptr, index=(t, c), shape=(1, BLOCK_C))
        total_sum = total_sum + ct.reshape(s, (BLOCK_C,))
        total_dot = total_dot + ct.reshape(d, (BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(BLOCK_C,))
    ct.store(sum_out_ptr, index=(c,), tile=total_sum)
    ct.store(dot_out_ptr, index=(c,), tile=total_dot)
    ct.store(weight_grad_out_ptr, index=(c,), tile=total_dot * invstd)


@ct.kernel
def _bn_backward_epilogue_kernel(
    crop_ptr,       # bf16 (n_rows, C)
    act_ptr,        # bf16 (n_rows, C)
    mean_ptr,       # f32 (C,)
    invstd_ptr,     # f32 (C,)
    weight_ptr,     # f32 (C,)
    bias_ptr,       # f32 (C,)
    sum_ptr,        # f32 (C,)
    dot_ptr,        # f32 (C,)
    out_ptr,        # bf16 (n_rows, C)
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BN_SCALE_C: ct.Constant[float],
):
    r = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(act_ptr, index=(r, c), shape=(BLOCK_HW, BLOCK_C))
    crop = ct.load(crop_ptr, index=(r, c), shape=(BLOCK_HW, BLOCK_C))
    mean = ct.load(mean_ptr, index=(c,), shape=(BLOCK_C,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c,), shape=(BLOCK_C,))
    sum0 = ct.load(sum_ptr, index=(c,), shape=(BLOCK_C,))
    sum1 = ct.load(dot_ptr, index=(c,), shape=(BLOCK_C,))

    x_f = ct.astype(x, ct.float32)
    crop_f = ct.astype(crop, ct.float32)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    invstd_2d = ct.reshape(invstd, (1, BLOCK_C))
    weight_2d = ct.reshape(weight, (1, BLOCK_C))
    bias_2d = ct.reshape(bias, (1, BLOCK_C))
    sum0_2d = ct.reshape(sum0, (1, BLOCK_C))
    sum1_2d = ct.reshape(sum1, (1, BLOCK_C))

    centered = x_f - mean_2d
    affine = centered * invstd_2d * weight_2d + bias_2d
    affine_bf = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    sig = _sigmoid(affine_bf)
    producer = crop_f * sig * (affine_bf * (1.0 - sig) + 1.0)
    producer_bf = ct.astype(ct.astype(producer, ct.bfloat16), ct.float32)

    centered_term = sum1_2d * BN_SCALE_C * invstd_2d * invstd_2d
    mean_term = sum0_2d * BN_SCALE_C
    scale = invstd_2d * weight_2d
    out = (producer_bf - centered * centered_term - mean_term) * scale
    ct.store(out_ptr, index=(r, c), tile=ct.astype(out, ct.bfloat16))


# BLOCK_HW=1024, BLOCK_C=16 divides both shapes' n_rows and C.
@oracle_impl(hardware="B200", point="afb3f0db",
             BLOCK_HW=1024, BLOCK_C=16, NUM_ROW_BLOCKS=1568)
@oracle_impl(hardware="B200", point="9eed2ad9",
             BLOCK_HW=1024, BLOCK_C=16, NUM_ROW_BLOCKS=98)
def oracle_forward(inputs, *, BLOCK_HW, BLOCK_C, NUM_ROW_BLOCKS):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg1_1.device
    n = int(arg1_1.shape[0])
    c = int(arg1_1.shape[1])
    h = int(arg1_1.shape[2])
    w = int(arg1_1.shape[3])
    n_rows = n * h * w

    # Slice arg0[:, :, :h, :w] then flatten to NHWC-flat (n_rows, C).
    crop_bf = arg0_1[:, :, :h, :w].contiguous(memory_format=torch.channels_last)
    crop_flat = crop_bf.permute(0, 2, 3, 1).contiguous().view(n_rows, c)
    act_flat = arg1_1.permute(0, 2, 3, 1).contiguous().view(n_rows, c)
    mean_1d = arg2_1.view(c)
    invstd_1d = arg3_1.view(c)

    num_row_blocks = n_rows // BLOCK_HW
    num_c_blocks = c // BLOCK_C
    assert num_row_blocks == NUM_ROW_BLOCKS, f"{num_row_blocks} != {NUM_ROW_BLOCKS}"

    partial_sum = torch.empty((num_row_blocks, c), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_row_blocks, c), device=device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    dot_out = torch.empty((c,), device=device, dtype=torch.float32)
    weight_grad_out = torch.empty((c,), device=device, dtype=torch.float32)

    # Final output: channels-last strided.
    out_full = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(int(s) for s in arg1_1.stride()),
        device=device, dtype=torch.bfloat16,
    )
    out_flat = torch.empty((n_rows, c), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_blocks, num_c_blocks, 1),
        _producer_partials_kernel,
        (crop_flat, act_flat, mean_1d, invstd_1d, arg4_1, arg5_1,
         partial_sum, partial_dot,
         BLOCK_HW, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d,
         sum_out, dot_out, weight_grad_out,
         NUM_ROW_BLOCKS, BLOCK_C),
    )
    ct.launch(
        stream,
        (num_row_blocks, num_c_blocks, 1),
        _bn_backward_epilogue_kernel,
        (crop_flat, act_flat, mean_1d, invstd_1d, arg4_1, arg5_1,
         sum_out, dot_out, out_flat,
         BLOCK_HW, BLOCK_C, BN_SCALE),
    )

    # Restore output strides to channels-last.
    out_full.copy_(out_flat.view(n, h, w, c).permute(0, 3, 1, 2))
    return sum_out, weight_grad_out, out_full
