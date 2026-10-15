"""cuTile port of sum_sum_b00464d36daa: DenseNet BN-backward + slice-alias.

Matches Triton's 3-kernel plan: split-K partial reduce over N*HW, per-channel
finalize, then the BN-backward epilogue. cuTile primitives used: ct.load with
padding_mode=ZERO for the masked K-tile loads, ct.where for the ReLU-mask
select, ct.sum for the two partial reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 256
H = 56
W = 56
HW = H * W  # 3136
SCALE = 7.971938775510203e-05
SLICE_START = 224


@ct.kernel
def _partial_reduce_kernel(
    mask_ptr,             # bf16 [N, C, HW]
    fill_ptr,             # bf16 [1]
    source_ptr,           # bf16 [N, C, HW]
    centered_source_ptr,  # bf16 [N, C, HW]
    mean_ptr,             # f32  [C]
    partial_sum_ptr,      # f32  [NUM_BLOCKS, C]
    partial_dot_ptr,      # f32  [NUM_BLOCKS, C]
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    """Grid: (C, num_blocks, 1). One (c, k-tile) partial per program."""
    c = ct.bid(0)
    block = ct.bid(1)
    k_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    k_global = block * BLOCK_K + k_idx
    active = k_global < K_TOTAL
    n = k_global // HW_C
    spatial = k_global - n * HW_C

    n_bc = n
    c_bc = ct.full((BLOCK_K,), c, dtype=ct.int32)
    hw_bc = spatial

    mask_v = ct.gather(mask_ptr, (n_bc, c_bc, hw_bc), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n_bc, c_bc, hw_bc), mask=active,
                         padding_value=ct.bfloat16(0.0))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1,)), (BLOCK_K,))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    grad_bf = ct.where(mask_v <= zero_bf, fill_bc, source_v)
    grad = ct.astype(grad_bf, ct.float32)

    centered_src = ct.astype(
        ct.gather(centered_source_ptr, (n_bc, c_bc, hw_bc), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    mean_bc = ct.broadcast_to(mean, (BLOCK_K,))
    centered = centered_src - mean_bc

    grad = ct.where(active, grad, 0.0)
    centered = ct.where(active, centered, 0.0)

    partial_sum = ct.sum(grad)
    partial_dot = ct.sum(grad * centered)
    ct.store(partial_sum_ptr, index=(block, c),
             tile=ct.reshape(partial_sum, (1, 1)))
    ct.store(partial_dot_ptr, index=(block, c),
             tile=ct.reshape(partial_dot, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [NUM_BLOCKS, C]
    partial_dot_ptr,   # f32 [NUM_BLOCKS, C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    scale_grad_ptr,    # f32 [C]
    mean_term_ptr,     # f32 [C]
    variance_ptr,      # f32 [C]
    output_scale_ptr,  # f32 [C]
    C_C: ct.Constant[int],
    NUM_BLOCKS: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    BLOCK_BLOCKS_: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    """Grid: (C // BLOCK_C_,). Reduces partial rows along block axis."""
    c_block = ct.bid(0)
    part_sum = ct.astype(
        ct.load(partial_sum_ptr, index=(0, c_block), shape=(BLOCK_BLOCKS_, BLOCK_C_)),
        ct.float32,
    )
    part_dot = ct.astype(
        ct.load(partial_dot_ptr, index=(0, c_block), shape=(BLOCK_BLOCKS_, BLOCK_C_)),
        ct.float32,
    )
    sum_v = ct.sum(part_sum, axis=0)
    dot_v = ct.sum(part_dot, axis=0)
    invstd = ct.astype(ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,)),
                       ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C_,)),
                       ct.float32)

    scale_grad = dot_v * invstd
    mean_term = sum_v * SCALE_C
    variance = (dot_v * SCALE_C) * (invstd * invstd)
    output_scale = invstd * weight

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_v)
    ct.store(scale_grad_ptr, index=(c_block,), tile=scale_grad)
    ct.store(mean_term_ptr, index=(c_block,), tile=mean_term)
    ct.store(variance_ptr, index=(c_block,), tile=variance)
    ct.store(output_scale_ptr, index=(c_block,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    mask_ptr,             # bf16 [N, C, HW]
    fill_ptr,             # bf16 [1]
    source_ptr,           # bf16 [N, C, HW]
    centered_source_ptr,  # bf16 [N, C, HW]
    mean_ptr,             # f32  [C]
    mean_term_ptr,        # f32  [C]
    variance_ptr,         # f32  [C]
    output_scale_ptr,     # f32  [C]
    dense_out_ptr,        # bf16 [N, C, HW]
    HW_C: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    """Grid: (C, num_blocks, 1). Writes bf16 gradient for full-tile K span."""
    c = ct.bid(0)
    block = ct.bid(1)
    k_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    k_global = block * BLOCK_K + k_idx
    active = k_global < K_TOTAL
    n = k_global // HW_C
    spatial = k_global - n * HW_C

    c_bc = ct.full((BLOCK_K,), c, dtype=ct.int32)

    mask_v = ct.gather(mask_ptr, (n, c_bc, spatial), mask=active,
                       padding_value=ct.bfloat16(0.0))
    source_v = ct.gather(source_ptr, (n, c_bc, spatial), mask=active,
                         padding_value=ct.bfloat16(0.0))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1,)), (BLOCK_K,))
    zero_bf = ct.astype(0.0, ct.bfloat16)
    grad_bf = ct.where(mask_v <= zero_bf, fill_bc, source_v)
    grad = ct.astype(grad_bf, ct.float32)

    centered_src = ct.astype(
        ct.gather(centered_source_ptr, (n, c_bc, spatial), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    mean_term = ct.astype(ct.load(mean_term_ptr, index=(c,), shape=(1,)),
                          ct.float32)
    variance = ct.astype(ct.load(variance_ptr, index=(c,), shape=(1,)),
                         ct.float32)
    output_scale = ct.astype(ct.load(output_scale_ptr, index=(c,), shape=(1,)),
                             ct.float32)
    centered = centered_src - ct.broadcast_to(mean, (BLOCK_K,))
    variance_bc = ct.broadcast_to(variance, (BLOCK_K,))
    mean_term_bc = ct.broadcast_to(mean_term, (BLOCK_K,))
    output_scale_bc = ct.broadcast_to(output_scale, (BLOCK_K,))

    corrected = grad - centered * variance_bc
    centered_grad = corrected - mean_term_bc
    dense_bf16 = ct.astype(centered_grad * output_scale_bc, ct.bfloat16)
    ct.scatter(dense_out_ptr, (n, c_bc, spatial), dense_bf16, mask=active)


@oracle_impl(hardware="B200", point="878cf9f5", BLOCK_K=8192)
def oracle_forward(inputs, *, BLOCK_K):
    mask_input, fill, source, centered_source, mean, invstd, weight = inputs
    device = source.device

    K_TOTAL = N * HW
    num_blocks = (K_TOTAL + BLOCK_K - 1) // BLOCK_K
    block_blocks_pow2 = 1 << (num_blocks - 1).bit_length()

    partial_sum = torch.empty((num_blocks, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_blocks, C), device=device, dtype=torch.float32)

    mask_3d = mask_input.view(N, C, HW)
    source_3d = source.view(N, C, HW)
    centered_3d = centered_source.view(N, C, HW)
    fill_1d = fill.view(1)
    mean_1d = mean.view(C)
    invstd_1d = invstd.view(C)
    weight_1d = weight.view(C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, num_blocks, 1), _partial_reduce_kernel,
        (mask_3d, fill_1d, source_3d, centered_3d, mean_1d,
         partial_sum, partial_dot,
         C, HW, K_TOTAL, BLOCK_K),
    )

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    variance = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    BLOCK_C = 8
    assert C % BLOCK_C == 0
    ct.launch(
        stream, (C // BLOCK_C, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, scale_grad, mean_term, variance, output_scale,
         C, num_blocks, BLOCK_C, block_blocks_pow2, SCALE),
    )

    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    dense_out_3d = dense_out.view(N, C, HW)
    ct.launch(
        stream, (C, num_blocks, 1), _epilogue_kernel,
        (mask_3d, fill_1d, source_3d, centered_3d, mean_1d,
         mean_term, variance, output_scale, dense_out_3d,
         HW, K_TOTAL, BLOCK_K),
    )

    slice_1 = dense_out[:, SLICE_START:C, :, :]
    return sum_out, scale_grad, dense_out, slice_1
