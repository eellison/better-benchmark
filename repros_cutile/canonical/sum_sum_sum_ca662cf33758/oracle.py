"""cuTile port of sum_sum_sum_ca662cf33758: pytorch_unet BN-backward with
ReLU-mask.

For each shape, compute:
  - grad = where(mask_src <= 0, fill, where_rhs)   (bf16)
  - sum_1 = sum(grad, dim=[0,2,3])                 (f32 per-C)
  - sum_2 = sum(grad * (activation - mean), dim=[0,2,3])
  - dense_out = ((grad - centered*var_term) - mean_term) * (invstd*weight)
  - sum_3 = sum(bf16(dense_out), dim=[0,2,3])
Returns: (sum_1, mul_8=sum_2*invstd, dense_bf16, sum_3_f32)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_HW = 1024


def _next_power_of_2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _partial_reduce_kernel(
    mask_src_ptr,       # bf16 [C, HW]
    fill_ptr,           # bf16 (1,)
    where_rhs_ptr,      # bf16 [C, HW]
    activation_ptr,     # bf16 [C, HW]
    mean_ptr,           # f32 [C]
    partial_sum_ptr,    # f32 [num_blocks, C]
    partial_dot_ptr,    # f32 [num_blocks, C]
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
):
    c = ct.bid(0)
    block = ct.bid(1)

    mask_src = ct.load(
        mask_src_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    where_rhs = ct.load(
        where_rhs_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    activation = ct.load(
        activation_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill_bf16 = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))

    mask_f = ct.astype(mask_src, ct.float32)
    rhs_f = ct.astype(where_rhs, ct.float32)
    fill_f = ct.astype(fill_bf16, ct.float32)
    fill_2d = ct.reshape(fill_f, (1, 1)) + ct.zeros((1, BLOCK_HW_), dtype=ct.float32)
    grad = ct.where(mask_f <= 0.0, fill_2d, rhs_f)

    activation_f = ct.astype(activation, ct.float32)
    centered = activation_f - mean

    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    hw_start = block * BLOCK_HW_
    hw_mask_1d = (hw_start + hw_idx) < HW_
    hw_mask_2d = ct.reshape(hw_mask_1d, (1, BLOCK_HW_))
    grad_masked = ct.where(hw_mask_2d, grad, 0.0)
    dot_masked = ct.where(hw_mask_2d, grad * centered, 0.0)

    sum_v = ct.sum(grad_masked)
    dot_v = ct.sum(dot_masked)
    ct.store(partial_sum_ptr, index=(block, c), tile=ct.reshape(sum_v, (1, 1)))
    ct.store(partial_dot_ptr, index=(block, c), tile=ct.reshape(dot_v, (1, 1)))


@ct.kernel
def _finalize_first_kernel(
    partial_sum_ptr,    # f32 [num_blocks_pad, C]
    partial_dot_ptr,    # f32 [num_blocks_pad, C]
    invstd_ptr,         # f32 [C] (arg5_1)
    sum_out_ptr,        # f32 [C]
    dot_out_ptr,        # f32 [C]
    mul8_out_ptr,       # f32 [C]  = sum_2 * invstd
    NUM_BLOCKS_PAD: ct.Constant[int],
    NUM_BLOCKS_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    tile_sum = ct.load(
        partial_sum_ptr, index=(0, c_block), shape=(NUM_BLOCKS_PAD, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_dot = ct.load(
        partial_dot_ptr, index=(0, c_block), shape=(NUM_BLOCKS_PAD, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    b_idx = ct.arange(NUM_BLOCKS_PAD, dtype=ct.int32)
    b_mask = b_idx < NUM_BLOCKS_
    b_mask_2d_i = ct.reshape(ct.astype(b_mask, ct.int32), (NUM_BLOCKS_PAD, 1)) + ct.zeros((NUM_BLOCKS_PAD, BLOCK_C_), dtype=ct.int32)
    b_mask_2d = b_mask_2d_i != 0
    tile_sum_masked = ct.where(b_mask_2d, tile_sum, 0.0)
    tile_dot_masked = ct.where(b_mask_2d, tile_dot, 0.0)

    sum_v = ct.sum(tile_sum_masked, axis=0)
    dot_v = ct.sum(tile_dot_masked, axis=0)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C_,))
    ct.store(sum_out_ptr, index=(c_block,), tile=sum_v)
    ct.store(dot_out_ptr, index=(c_block,), tile=dot_v)
    ct.store(mul8_out_ptr, index=(c_block,), tile=dot_v * invstd)


@ct.kernel
def _epilogue_kernel(
    mask_src_ptr,
    fill_ptr,
    where_rhs_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,          # arg5_1
    weight_ptr,          # arg6_1
    sum_ptr,
    dot_ptr,
    dense_ptr,           # bf16 [C, HW]
    partial_dense_sum_ptr,   # f32 [num_blocks, C]
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c = ct.bid(0)
    block = ct.bid(1)

    mask_src = ct.load(
        mask_src_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    where_rhs = ct.load(
        where_rhs_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    activation = ct.load(
        activation_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    fill_bf16 = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    sum_v = ct.load(sum_ptr, index=(c,), shape=(1,))
    dot_v = ct.load(dot_ptr, index=(c,), shape=(1,))

    mask_f = ct.astype(mask_src, ct.float32)
    rhs_f = ct.astype(where_rhs, ct.float32)
    fill_f = ct.astype(fill_bf16, ct.float32)
    fill_2d = ct.reshape(fill_f, (1, 1)) + ct.zeros((1, BLOCK_HW_), dtype=ct.float32)
    grad = ct.where(mask_f <= 0.0, fill_2d, rhs_f)

    activation_f = ct.astype(activation, ct.float32)
    centered = activation_f - mean

    mean_term = sum_v * SCALE_
    dot_mean = dot_v * SCALE_
    invstd_sq = invstd * invstd
    correction_scale = dot_mean * invstd_sq
    correction = centered * correction_scale
    after = grad - correction
    centered_grad = after - mean_term
    output_scale = invstd * weight
    dense_f = centered_grad * output_scale
    dense_bf16 = ct.astype(dense_f, ct.bfloat16)
    ct.store(dense_ptr, index=(c, block), tile=dense_bf16)

    dense_f_bf = ct.astype(dense_bf16, ct.float32)
    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    hw_start = block * BLOCK_HW_
    hw_mask_1d = (hw_start + hw_idx) < HW_
    hw_mask_2d = ct.reshape(hw_mask_1d, (1, BLOCK_HW_))
    dense_masked = ct.where(hw_mask_2d, dense_f_bf, 0.0)
    total = ct.sum(dense_masked)
    ct.store(partial_dense_sum_ptr, index=(block, c), tile=ct.reshape(total, (1, 1)))


@ct.kernel
def _finalize_dense_sum_kernel(
    partial_dense_sum_ptr,
    dense_sum_out_ptr,
    NUM_BLOCKS_PAD: ct.Constant[int],
    NUM_BLOCKS_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    tile = ct.load(
        partial_dense_sum_ptr, index=(0, c_block), shape=(NUM_BLOCKS_PAD, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    b_idx = ct.arange(NUM_BLOCKS_PAD, dtype=ct.int32)
    b_mask = b_idx < NUM_BLOCKS_
    b_mask_2d_i = ct.reshape(ct.astype(b_mask, ct.int32), (NUM_BLOCKS_PAD, 1)) + ct.zeros((NUM_BLOCKS_PAD, BLOCK_C_), dtype=ct.int32)
    b_mask_2d = b_mask_2d_i != 0
    tile_masked = ct.where(b_mask_2d, tile, 0.0)
    total = ct.sum(tile_masked, axis=0)
    # Round through bf16 (Triton oracle stores bf16 sum then converts back to f32)
    ct.store(dense_sum_out_ptr, index=(c_block,), tile=ct.astype(ct.astype(total, ct.bfloat16), ct.float32))


# Per-point BLOCK_K args are ignored; we use fixed BLOCK_HW=1024 which is pow2
# and works for all HW (padding via masks).
@oracle_impl(hardware="B200", point="39b5812f", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="9309c5e2", BLOCK_K=4096)
@oracle_impl(hardware="B200", point="0439afe2", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="457eec03", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="f63148ef", BLOCK_K=1024)
def oracle_forward(inputs, **_kwargs):
    (
        arg0_1,      # bf16 [1, C, H, W] mask_src
        arg1_1,      # bf16 [] fill
        arg2_1,      # bf16 [1, C, H, W] where_rhs
        arg3_1,      # bf16 [1, C, H, W] activation
        arg4_1,      # f32 [1, C, 1, 1] mean
        arg5_1,      # f32 [C] invstd
        arg6_1,      # f32 [C] weight
    ) = inputs
    device = arg0_1.device
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    assert n == 1

    mask_src_ch = arg0_1.view(c, hw)
    fill_flat = arg1_1.view(1)
    where_rhs_ch = arg2_1.view(c, hw)
    activation_ch = arg3_1.view(c, hw)
    mean_1d = arg4_1.view(c)

    num_blocks = (hw + BLOCK_HW - 1) // BLOCK_HW
    num_blocks_pad = _next_power_of_2(num_blocks)

    partial_sum = torch.empty((num_blocks, c), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_blocks, c), device=device, dtype=torch.float32)
    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty((c,), device=device, dtype=torch.float32)
    mul8_out = torch.empty((c,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (n, c, h, w), (c * hw, hw, w, 1),
        device=device, dtype=torch.bfloat16,
    )
    # Padded scratch to accept full BLOCK_HW writes then narrow-copy.
    hw_padded = num_blocks * BLOCK_HW
    dense_pad = torch.empty((c, hw_padded), device=device, dtype=torch.bfloat16)
    partial_dense_sum = torch.empty((num_blocks, c), device=device, dtype=torch.float32)
    dense_sum_out = torch.empty((c,), device=device, dtype=torch.float32)

    # NOTE: The eager repro hard-codes 1.6293013555787278e-06 (= 1/613760)
    # regardless of the actual HW. Match that behavior.
    SCALE = 1.6293013555787278e-06

    BLOCK_C = 8
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c, num_blocks, 1),
        _partial_reduce_kernel,
        (mask_src_ch, fill_flat, where_rhs_ch, activation_ch, mean_1d,
         partial_sum, partial_dot, hw, BLOCK_HW),
    )
    ct.launch(
        stream,
        (ct.cdiv(c, BLOCK_C), 1, 1),
        _finalize_first_kernel,
        (partial_sum, partial_dot, arg5_1,
         sum_out, dot_tmp, mul8_out,
         num_blocks_pad, num_blocks, BLOCK_C),
    )
    ct.launch(
        stream,
        (c, num_blocks, 1),
        _epilogue_kernel,
        (mask_src_ch, fill_flat, where_rhs_ch, activation_ch, mean_1d,
         arg5_1, arg6_1, sum_out, dot_tmp, dense_pad, partial_dense_sum,
         hw, BLOCK_HW, SCALE),
    )
    # Copy valid HW region from padded to real dense_out view.
    dense_out.view(c, hw).copy_(dense_pad.narrow(1, 0, hw))
    ct.launch(
        stream,
        (ct.cdiv(c, BLOCK_C), 1, 1),
        _finalize_dense_sum_kernel,
        (partial_dense_sum, dense_sum_out, num_blocks_pad, num_blocks, BLOCK_C),
    )

    return sum_out, mul8_out, dense_out, dense_sum_out
