"""cuTile port of mean_var_mean_dfa4dc2f8f9a: ConvNeXtV2 spatial mean + LayerNorm.

Two cuTile kernels mirror the Triton oracle:
1. Spatial-mean kernel: BLOCK_ROWS rows of (n, channel), compute
   `pooled[n, c] = mean(x + residual, [-1,-2])` bf16, and partial sums for
   the channel-wise mean/variance across channels within a batch.
2. LayerNorm affine: for each batch, combine partial sums into channel_mean +
   rsqrt, then affine (weight, bias).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _spatial_mean_kernel(
    x_ptr,           # bf16 flat storage
    residual_ptr,    # bf16 flat storage
    pooled_ptr,      # bf16 [TOTAL_ROWS]
    partials_ptr,    # f32  [2, TOTAL_BLOCKS] (sum, sumsq stacked)
    X_STRIDE_N: ct.Constant[int],
    X_STRIDE_C: ct.Constant[int],
    X_STRIDE_H: ct.Constant[int],
    X_STRIDE_W: ct.Constant[int],
    R_STRIDE_N: ct.Constant[int],
    R_STRIDE_C: ct.Constant[int],
    R_STRIDE_H: ct.Constant[int],
    R_STRIDE_W: ct.Constant[int],
    TOTAL_ROWS: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    WIDTH: ct.Constant[int],
    HW: ct.Constant[int],
    TOTAL_BLOCKS: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    pid = ct.bid(0)
    rows = pid * BLOCK_ROWS + ct.arange(BLOCK_ROWS, dtype=ct.int32)
    hw = ct.arange(BLOCK_HW, dtype=ct.int32)
    row_valid = rows < TOTAL_ROWS
    hw_valid = hw < HW

    n = rows // CHANNELS
    c = rows - n * CHANNELS
    h = hw // WIDTH
    w = hw - h * WIDTH

    n_2d = ct.reshape(n, (BLOCK_ROWS, 1))
    c_2d = ct.reshape(c, (BLOCK_ROWS, 1))
    h_2d = ct.reshape(h, (1, BLOCK_HW))
    w_2d = ct.reshape(w, (1, BLOCK_HW))
    x_off = (n_2d * X_STRIDE_N + c_2d * X_STRIDE_C
             + h_2d * X_STRIDE_H + w_2d * X_STRIDE_W)
    r_off = (n_2d * R_STRIDE_N + c_2d * R_STRIDE_C
             + h_2d * R_STRIDE_H + w_2d * R_STRIDE_W)
    row_valid_2d = ct.reshape(row_valid, (BLOCK_ROWS, 1))
    hw_valid_2d = ct.reshape(hw_valid, (1, BLOCK_HW))
    mask = row_valid_2d & hw_valid_2d

    x = ct.astype(ct.gather(x_ptr, (x_off,), mask=mask), ct.float32)
    residual = ct.astype(ct.gather(residual_ptr, (r_off,), mask=mask), ct.float32)
    add_bf = ct.astype(x + residual, ct.bfloat16)
    add_f = ct.astype(add_bf, ct.float32)
    zero_2d = ct.full((BLOCK_ROWS, BLOCK_HW), 0.0, dtype=ct.float32)
    add_masked = ct.where(mask, add_f, zero_2d)
    spatial_sum = ct.sum(add_masked, axis=1)
    pooled_f = spatial_sum * (1.0 / HW)
    pooled_bf = ct.astype(pooled_f, ct.bfloat16)
    ct.scatter(pooled_ptr, (rows,), pooled_bf, mask=row_valid)

    # Compute per-block partial sums over the pooled_f32 values (of the rows
    # in this block).
    pooled_f32 = ct.astype(pooled_bf, ct.float32)
    zero_1d = ct.full((BLOCK_ROWS,), 0.0, dtype=ct.float32)
    valid_pooled = ct.where(row_valid, pooled_f32, zero_1d)
    partial_sum = ct.sum(valid_pooled)
    partial_sumsq = ct.sum(valid_pooled * valid_pooled)
    # partials layout: [0..TB) = sums, [TB..2*TB) = sumsq
    ct.store(partials_ptr, index=(pid,), tile=ct.reshape(partial_sum, (1,)))
    ct.store(partials_ptr, index=(TOTAL_BLOCKS + pid,),
             tile=ct.reshape(partial_sumsq, (1,)))


@ct.kernel
def _layernorm_affine_kernel(
    pooled_ptr,       # bf16 [BATCH * CHANNELS]
    partials_ptr,     # f32 [2 * TOTAL_BLOCKS]
    weight_ptr,       # f32 [CHANNELS]
    bias_ptr,         # f32 [CHANNELS]
    channel_mean_ptr, # f32 [BATCH]
    rsqrt_ptr,        # f32 [BATCH]
    out_ptr,          # bf16 [BATCH, CHANNELS]
    CHANNELS: ct.Constant[int],
    TOTAL_BLOCKS: ct.Constant[int],
    BLOCKS_PER_BATCH: ct.Constant[int],
    BLOCK_P: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    col_valid = cols < CHANNELS
    partial_offsets = ct.arange(BLOCK_P, dtype=ct.int32)
    partial_valid = partial_offsets < BLOCKS_PER_BATCH
    partial_base = n * BLOCKS_PER_BATCH + partial_offsets

    partial_sum = ct.gather(
        partials_ptr, (partial_base,), mask=partial_valid,
    )
    partial_sumsq = ct.gather(
        partials_ptr, (partial_base + TOTAL_BLOCKS,), mask=partial_valid,
    )
    zero_p = ct.full((BLOCK_P,), 0.0, dtype=ct.float32)
    partial_sum_m = ct.where(partial_valid, partial_sum, zero_p)
    partial_sumsq_m = ct.where(partial_valid, partial_sumsq, zero_p)
    total = ct.sum(partial_sum_m)
    total_sumsq = ct.sum(partial_sumsq_m)

    pooled_bf = ct.gather(
        pooled_ptr,
        (ct.full((BLOCK_C,), n * CHANNELS, dtype=ct.int32) + cols,),
        mask=col_valid,
    )
    pooled_f = ct.astype(pooled_bf, ct.float32)

    channel_mean = total * (1.0 / CHANNELS)
    ct.store(channel_mean_ptr, index=(n,), tile=ct.reshape(channel_mean, (1,)))

    mean_square = channel_mean * channel_mean
    variance = (total_sumsq * (1.0 / CHANNELS)) - mean_square
    zero_scalar = ct.astype(0.0, ct.float32)
    variance_pos = ct.where(variance > zero_scalar, variance, zero_scalar)
    invstd = ct.rsqrt(variance_pos + EPS)
    ct.store(rsqrt_ptr, index=(n,), tile=ct.reshape(invstd, (1,)))

    weight = ct.gather(weight_ptr, (cols,), mask=col_valid)
    bias = ct.gather(bias_ptr, (cols,), mask=col_valid)
    zero_c = ct.full((BLOCK_C,), 0.0, dtype=ct.float32)
    centered = ct.where(col_valid, pooled_f - channel_mean, zero_c)
    normed = centered * invstd
    affine = normed * weight + bias
    ct.scatter(
        out_ptr,
        (ct.full((BLOCK_C,), n * CHANNELS, dtype=ct.int32) + cols,),
        ct.astype(affine, ct.bfloat16),
        mask=col_valid,
    )


def _next_pow2(n):
    r = 1
    while r < n:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="c50453fe", BLOCK_ROWS=64, BLOCK_C=1024)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_C: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1,
        shape_param_0, shape_param_1, shape_param_2,
    ) = inputs
    device = arg0_1.device
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    total_rows = batch * channels
    total_blocks = (total_rows + BLOCK_ROWS - 1) // BLOCK_ROWS
    blocks_per_batch = (channels + BLOCK_ROWS - 1) // BLOCK_ROWS
    BLOCK_HW = _next_pow2(hw)
    BLOCK_P = _next_pow2(blocks_per_batch)

    pooled = torch.empty_strided(
        (batch, 1, 1, channels),
        (channels, channels, channels, 1),
        device=device, dtype=torch.bfloat16,
    )
    channel_mean = torch.empty_strided(
        (batch, 1, 1, 1), (1, 1, 1, 1),
        device=device, dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (batch, 1, 1, 1), (1, 1, 1, 1),
        device=device, dtype=torch.float32,
    )
    out = torch.empty_strided(
        (batch, channels), (channels, 1),
        device=device, dtype=torch.bfloat16,
    )
    partials = torch.empty((2 * total_blocks,), device=device, dtype=torch.float32)

    x_flat = torch.as_strided(arg0_1, (arg0_1.numel(),), (1,))
    res_flat = torch.as_strided(arg1_1, (arg1_1.numel(),), (1,))
    pooled_flat = pooled.view(total_rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (total_blocks, 1, 1), _spatial_mean_kernel,
        (x_flat, res_flat, pooled_flat, partials,
         int(arg0_1.stride(0)), int(arg0_1.stride(1)),
         int(arg0_1.stride(2)), int(arg0_1.stride(3)),
         int(arg1_1.stride(0)), int(arg1_1.stride(1)),
         int(arg1_1.stride(2)), int(arg1_1.stride(3)),
         total_rows, channels, width, hw, total_blocks, BLOCK_ROWS, BLOCK_HW),
    )
    ct.launch(
        stream, (batch, 1, 1), _layernorm_affine_kernel,
        (pooled_flat, partials, arg2_1, arg3_1,
         channel_mean.view(batch), rsqrt.view(batch),
         out.view(total_rows // channels * channels),
         channels, total_blocks, blocks_per_batch, BLOCK_P, BLOCK_C),
    )

    # Reshape returned pooled to the permuted layout.
    # pooled has shape (batch, 1, 1, channels) contiguous (channels, ..., 1)
    # The oracle returns "permute" which is pooled reshaped to [128, 1, 1, 640].
    # Our pooled already has that shape.
    permute = pooled
    return permute, channel_mean, rsqrt, out
