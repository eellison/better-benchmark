"""cuTile port of sum_sum_9b7fd1c0bb67: DenseNet BN-backward + slice add residual.

Split-K design mirroring the Triton reference to unlock parallelism across the
(N, HW) axes rather than a single-program-per-channel scheme (grid=(80,) was
severely under-occupying B200's 148 SMs and loading huge (N, 1, HW) tiles).

Three kernels:
  1. partial_reduce: grid (C, num_n_blocks, num_hw_blocks). Each block reads a
     (BLOCK_N, 1, BLOCK_HW) tile for one channel, computes local sum_where and
     sum(where * centered), writes into partial buffers.
  2. finalize: grid (C, 1, 1). Reduces partials into scalar sum_where and
     sum_centered per channel. Emits the two return-scalar buffers plus three
     f32 intermediate vectors (mean_term, variance_term, output_scale).
  3. epilogue: grid (C, num_n_blocks, num_hw_blocks). Applies BN-backward with
     the finalized intermediates and adds the sliced residual (bf16).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 7.62939453125e-06
SLICE_START = 16


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    mask_input_arr,      # bf16 [N, C, HW]
    fill_arr,            # bf16 [1]
    rhs_arr,             # bf16 [N, C, HW]
    activation_arr,      # bf16 [N, C, HW]
    mean_arr,            # f32 [C]
    partial_sum_arr,     # f32 [C, NUM_K_BLOCKS]
    partial_dot_arr,     # f32 [C, NUM_K_BLOCKS]
    NUM_HW_BLOCKS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c = ct.bid(0)
    n_block = ct.bid(1)
    hw_block = ct.bid(2)

    mask_input = ct.load(mask_input_arr, index=(n_block, c, hw_block),
                         shape=(BLOCK_N, 1, BLOCK_HW))
    rhs = ct.load(rhs_arr, index=(n_block, c, hw_block),
                  shape=(BLOCK_N, 1, BLOCK_HW))
    activation = ct.load(activation_arr, index=(n_block, c, hw_block),
                         shape=(BLOCK_N, 1, BLOCK_HW))
    fill_scalar = ct.load(fill_arr, index=(0,), shape=(1,))

    mask_input_f = ct.astype(mask_input, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    activation_f = ct.astype(activation, ct.float32)
    fill_f = ct.astype(fill_scalar, ct.float32)
    fill_bc = ct.reshape(fill_f, (1, 1, 1))

    where_value = ct.where(mask_input_f <= 0.0, fill_bc, rhs_f)

    mean = ct.load(mean_arr, index=(c,), shape=(1,))
    mean_bc = ct.reshape(mean, (1, 1, 1))
    centered = activation_f - mean_bc

    sum_where = ct.sum(where_value)
    sum_centered = ct.sum(where_value * centered)

    k_block = n_block * NUM_HW_BLOCKS + hw_block
    ct.store(partial_sum_arr, index=(c, k_block),
             tile=ct.reshape(sum_where, (1, 1)))
    ct.store(partial_dot_arr, index=(c, k_block),
             tile=ct.reshape(sum_centered, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_arr,     # f32 [C, NUM_K_BLOCKS]
    partial_dot_arr,     # f32 [C, NUM_K_BLOCKS]
    invstd_arr,          # f32 [C]
    weight_arr,          # f32 [C]
    sum_out_arr,         # f32 [C]
    weight_grad_arr,     # f32 [C]
    mean_term_arr,       # f32 [C]
    variance_term_arr,   # f32 [C]
    output_scale_arr,    # f32 [C]
    NUM_K_BLOCKS: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c = ct.bid(0)
    partial_sum = ct.load(
        partial_sum_arr, index=(c, 0), shape=(1, BLOCK_TILES),
        padding_mode=ct.PaddingMode.ZERO,
    )
    partial_dot = ct.load(
        partial_dot_arr, index=(c, 0), shape=(1, BLOCK_TILES),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_idx = ct.arange(BLOCK_TILES, dtype=ct.int32)
    valid = ct.reshape(tile_idx < NUM_K_BLOCKS, (1, BLOCK_TILES))
    zero_2d = ct.zeros((1, BLOCK_TILES), dtype=ct.float32)
    partial_sum_m = ct.where(valid, partial_sum, zero_2d)
    partial_dot_m = ct.where(valid, partial_dot, zero_2d)
    sum_where = ct.sum(partial_sum_m)
    sum_centered = ct.sum(partial_dot_m)

    invstd = ct.load(invstd_arr, index=(c,), shape=(1,))
    weight = ct.load(weight_arr, index=(c,), shape=(1,))

    sum_where_1 = ct.reshape(sum_where, (1,))
    sum_centered_1 = ct.reshape(sum_centered, (1,))
    mean_term = sum_where_1 * SCALE_
    invstd_sq = invstd * invstd
    variance_term = sum_centered_1 * SCALE_ * invstd_sq
    output_scale = invstd * weight
    weight_grad_val = sum_centered_1 * invstd

    ct.store(sum_out_arr, index=(c,), tile=sum_where_1)
    ct.store(weight_grad_arr, index=(c,), tile=weight_grad_val)
    ct.store(mean_term_arr, index=(c,), tile=mean_term)
    ct.store(variance_term_arr, index=(c,), tile=variance_term)
    ct.store(output_scale_arr, index=(c,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    residual_arr,        # bf16 [N, INPUT_C, HW]
    mask_input_arr,      # bf16 [N, C, HW]
    fill_arr,            # bf16 [1]
    rhs_arr,             # bf16 [N, C, HW]
    activation_arr,      # bf16 [N, C, HW]
    mean_arr,            # f32 [C]
    mean_term_arr,       # f32 [C]
    variance_term_arr,   # f32 [C]
    output_scale_arr,    # f32 [C]
    add_out_arr,         # bf16 [N, C, HW]
    BLOCK_N: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c = ct.bid(0)
    n_block = ct.bid(1)
    hw_block = ct.bid(2)

    mask_input = ct.load(mask_input_arr, index=(n_block, c, hw_block),
                         shape=(BLOCK_N, 1, BLOCK_HW))
    rhs = ct.load(rhs_arr, index=(n_block, c, hw_block),
                  shape=(BLOCK_N, 1, BLOCK_HW))
    activation = ct.load(activation_arr, index=(n_block, c, hw_block),
                         shape=(BLOCK_N, 1, BLOCK_HW))
    fill_scalar = ct.load(fill_arr, index=(0,), shape=(1,))

    mask_input_f = ct.astype(mask_input, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    activation_f = ct.astype(activation, ct.float32)
    fill_f = ct.astype(fill_scalar, ct.float32)
    fill_bc = ct.reshape(fill_f, (1, 1, 1))

    where_value = ct.where(mask_input_f <= 0.0, fill_bc, rhs_f)

    mean = ct.load(mean_arr, index=(c,), shape=(1,))
    mean_bc = ct.reshape(mean, (1, 1, 1))
    centered = activation_f - mean_bc

    mean_term = ct.reshape(
        ct.load(mean_term_arr, index=(c,), shape=(1,)), (1, 1, 1))
    variance_term = ct.reshape(
        ct.load(variance_term_arr, index=(c,), shape=(1,)), (1, 1, 1))
    output_scale = ct.reshape(
        ct.load(output_scale_arr, index=(c,), shape=(1,)), (1, 1, 1))

    grad = (where_value - centered * variance_term - mean_term) * output_scale
    grad_bf16 = ct.astype(grad, ct.bfloat16)

    residual = ct.load(residual_arr,
                       index=(n_block, c + SLICE_START, hw_block),
                       shape=(BLOCK_N, 1, BLOCK_HW))
    residual_f = ct.astype(residual, ct.float32)
    out_f = residual_f + ct.astype(grad_bf16, ct.float32)
    out_bf16 = ct.astype(out_f, ct.bfloat16)

    ct.store(add_out_arr, index=(n_block, c, hw_block), tile=out_bf16)


@oracle_impl(hardware="B200", point="3f23a743", BLOCK_N=32, BLOCK_HW=128)
@oracle_impl(hardware="B200", point="5474fc63", BLOCK_N=32, BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_HW: int):
    residual, mask, fill, rhs, activation, mean, invstd, weight = inputs
    n, c, h, w = mask.shape
    hw = h * w
    input_c = int(residual.shape[1])

    num_n_blocks = n // BLOCK_N
    num_hw_blocks = hw // BLOCK_HW
    num_k_blocks = num_n_blocks * num_hw_blocks
    block_tiles = _next_pow2(num_k_blocks)

    device = mask.device
    sum_out = torch.empty_strided((c,), (1,), device=device, dtype=torch.float32)
    weight_grad = torch.empty_strided((c,), (1,), device=device,
                                      dtype=torch.float32)
    mean_term = torch.empty_strided((c,), (1,), device=device,
                                    dtype=torch.float32)
    variance_term = torch.empty_strided((c,), (1,), device=device,
                                        dtype=torch.float32)
    output_scale = torch.empty_strided((c,), (1,), device=device,
                                       dtype=torch.float32)
    partial_sum = torch.empty_strided(
        (c, num_k_blocks), (num_k_blocks, 1),
        device=device, dtype=torch.float32,
    )
    partial_dot = torch.empty_like(partial_sum)
    add_out = torch.empty_strided(
        (n, c, h, w),
        (c * hw, hw, w, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    residual_3d = residual.view(n, input_c, hw)
    mask_3d = mask.view(n, c, hw)
    rhs_3d = rhs.view(n, c, hw)
    activation_3d = activation.view(n, c, hw)
    mean_1d = mean.view(c)
    invstd_1d = invstd.view(c)
    weight_1d = weight.view(c)
    add_3d = add_out.view(n, c, hw)
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, num_n_blocks, num_hw_blocks),
        _partial_reduce_kernel,
        (mask_3d, fill_1d, rhs_3d, activation_3d, mean_1d,
         partial_sum, partial_dot,
         num_hw_blocks, BLOCK_N, BLOCK_HW),
    )
    ct.launch(
        stream,
        (c, 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, weight_grad, mean_term, variance_term, output_scale,
         num_k_blocks, block_tiles, SCALE),
    )
    ct.launch(
        stream,
        (c, num_n_blocks, num_hw_blocks),
        _epilogue_kernel,
        (residual_3d, mask_3d, fill_1d, rhs_3d, activation_3d,
         mean_1d, mean_term, variance_term, output_scale, add_3d,
         BLOCK_N, BLOCK_HW),
    )
    return sum_out, weight_grad, add_out, add_out[:, :SLICE_START, :, :]
