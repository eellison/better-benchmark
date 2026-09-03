"""cuTile port of sum_sum_sum_60a418792eff: PyTorch-UNet BN-backward with
max-pool scatter.

Shape: N=1, C=256, H=160, W=239, R=HW=38240 (non-power-of-2).
Approach: pre-materialize the scatter-add (via torch under graph capture) and
the `add` tensor, then compute the BN-backward reductions and epilogue in
cuTile kernels (partial reduce -> finalize -> epilogue -> sum-3 finalize).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 1
C = 256
SRC_C = 512
H = 160
W = 239
HW = H * W  # 38240
POOL_H = 80
POOL_W = 119
POOL_HW = POOL_H * POOL_W  # 9520
R = N * HW  # 38240
REDUCTION_SCALE = 2.615062761506276e-05
BLOCK_HW = 2048  # pow2, HW=38240 -> 19 tiles (last is partial)


@ct.kernel
def _partial_reduce_kernel(
    add_ptr,           # bf16 [C, HW] (view of (1, C, H, W))
    x_ptr,             # bf16 [C, HW] (view of arg3_1)
    scalar_ptr,        # bf16 [] (arg8_1) - flat (1,)
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    bias_ptr,          # f32 [C]
    partial_sum_ptr,   # f32 [num_blocks, C]
    partial_dot_ptr,   # f32 [num_blocks, C]
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
):
    c = ct.bid(0)
    block = ct.bid(1)

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    scalar_bf16 = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_f = ct.astype(scalar_bf16, ct.float32)

    x = ct.load(
        x_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    add_bf16 = ct.load(
        add_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )

    x_f = ct.astype(x, ct.float32)
    add_f = ct.astype(add_bf16, ct.float32)

    centered = x_f - mean
    affine = centered * invstd * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    # relu(affine) <= 0 <=> affine <= 0 (relu of negative is 0, relu of positive is >0)
    zero_bf16 = ct.astype(ct.full((1, BLOCK_HW_), 0.0, dtype=ct.float32), ct.bfloat16)
    take_scalar = affine_bf16 <= zero_bf16

    scalar_2d = ct.reshape(scalar_f, (1, 1)) + ct.zeros((1, BLOCK_HW_), dtype=ct.float32)
    selected = ct.where(take_scalar, scalar_2d, add_f)

    # Mask HW < HW_
    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    hw_start = block * BLOCK_HW_
    hw_mask_1d = (hw_start + hw_idx) < HW_
    hw_mask_2d = ct.reshape(hw_mask_1d, (1, BLOCK_HW_))

    selected_masked = ct.where(hw_mask_2d, selected, 0.0)
    centered_masked = ct.where(hw_mask_2d, centered, 0.0)

    sum_val = ct.sum(selected_masked)
    dot_val = ct.sum(selected_masked * centered_masked)

    ct.store(partial_sum_ptr, index=(block, c), tile=ct.reshape(sum_val, (1, 1)))
    ct.store(partial_dot_ptr, index=(block, c), tile=ct.reshape(dot_val, (1, 1)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [NUM_BLOCKS_PAD, C]
    partial_dot_ptr,   # f32 [NUM_BLOCKS_PAD, C]
    invstd_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    dot_tmp_ptr,       # f32 [C]
    mul10_ptr,         # f32 [C] = sum_2 * invstd (product of dot with squeeze_1==invstd)
    NUM_BLOCKS_PAD: ct.Constant[int],
    NUM_BLOCKS_: ct.Constant[int],
    C_: ct.Constant[int],
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
    ct.store(dot_tmp_ptr, index=(c_block,), tile=dot_v)
    ct.store(mul10_ptr, index=(c_block,), tile=dot_v * invstd)


@ct.kernel
def _epilogue_kernel(
    add_ptr,           # bf16 [C, HW]
    x_ptr,             # bf16 [C, HW]
    scalar_ptr,        # bf16 (1,)
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    bias_ptr,          # f32 [C]
    sum_ptr,           # f32 [C]
    dot_ptr,           # f32 [C]
    dense_ptr,         # bf16 [C, HW]
    partial_dense_sum_ptr,  # f32 [num_blocks, C]
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c = ct.bid(0)
    block = ct.bid(1)

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    sum_v = ct.load(sum_ptr, index=(c,), shape=(1,))
    dot_v = ct.load(dot_ptr, index=(c,), shape=(1,))
    scalar_bf16 = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_f = ct.astype(scalar_bf16, ct.float32)

    x = ct.load(
        x_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    add_bf16 = ct.load(
        add_ptr, index=(c, block), shape=(1, BLOCK_HW_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)
    add_f = ct.astype(add_bf16, ct.float32)

    centered = x_f - mean
    affine = centered * invstd * weight + bias
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf16 = ct.astype(ct.full((1, BLOCK_HW_), 0.0, dtype=ct.float32), ct.bfloat16)
    take_scalar = affine_bf16 <= zero_bf16
    scalar_2d = ct.reshape(scalar_f, (1, 1)) + ct.zeros((1, BLOCK_HW_), dtype=ct.float32)
    selected = ct.where(take_scalar, scalar_2d, add_f)

    sum_scaled = sum_v * SCALE_
    invstd_sq = invstd * invstd
    dot_scaled = dot_v * SCALE_ * invstd_sq
    channel_weight = invstd * weight
    out_f = (selected - centered * dot_scaled - sum_scaled) * channel_weight
    out_bf16 = ct.astype(out_f, ct.bfloat16)

    # Store with mask via padded output then narrow
    ct.store(dense_ptr, index=(c, block), tile=out_bf16)

    hw_idx = ct.arange(BLOCK_HW_, dtype=ct.int32)
    hw_start = block * BLOCK_HW_
    hw_mask_1d = (hw_start + hw_idx) < HW_
    hw_mask_2d = ct.reshape(hw_mask_1d, (1, BLOCK_HW_))
    out_f_bf = ct.astype(out_bf16, ct.float32)
    out_masked = ct.where(hw_mask_2d, out_f_bf, 0.0)
    dense_sum = ct.sum(out_masked)
    ct.store(partial_dense_sum_ptr, index=(block, c), tile=ct.reshape(dense_sum, (1, 1)))


@ct.kernel
def _finalize_dense_sum_kernel(
    partial_dense_sum_ptr,  # f32 [NUM_BLOCKS_PAD, C]
    dense_sum_out_ptr,      # f32 [C]
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
    ct.store(dense_sum_out_ptr, index=(c_block,), tile=ct.astype(ct.astype(total, ct.bfloat16), ct.float32))


def _next_power_of_2(v):
    return 1 << (int(v) - 1).bit_length()


@oracle_impl(
    hardware="B200", point="6c540e4e",
    GROUP_R=1280, REDUCE_BLOCK_R=128, BLOCK_C=8, FINAL_BLOCK_C=8, OUT_BLOCK_R=128,
)
def oracle_forward(inputs, **_kwargs):
    (
        arg0_1,        # bf16 [1, 512, 160, 239]
        arg1_1,        # bf16 [1, 256, 80, 119]
        arg2_1,        # i8 [1, 256, 80, 119] (max-pool offsets)
        arg3_1,        # bf16 [1, 256, 160, 239] (activation)
        arg4_1,        # f32 [1, 256, 1, 1] (mean)
        arg5_1,        # f32 [1, 256, 1, 1] (invstd)
        arg6_1,        # f32 [256] (weight)
        arg7_1,        # f32 [256] (bias)
        arg8_1,        # bf16 [] (scalar fill)
        sp0, sp1, sp2, sp3, sp4, sp5, sp6,
    ) = inputs
    device = arg3_1.device

    # 1. Compute the max-pool scatter in torch.
    slice_1 = arg0_1.narrow(1, 0, 256)
    view_arg1 = arg1_1.view(sp1)  # (256, 9520)
    full = torch.zeros(sp0, device=device, dtype=torch.float32)  # (256, 38240)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg2_1, sp2, sp3, sp4, [0, 0], [1, 1]
    ).view(sp5)  # (256, 9520)
    scatter_add = torch.scatter_add(full, 1, indices, view_arg1.to(torch.float32))
    scatter_bf16 = scatter_add.view(sp6).to(torch.bfloat16)
    add_tensor = slice_1 + scatter_bf16  # (1, 256, 160, 239) bf16

    # Reshape to (C, HW)
    add_ch = add_tensor.view(C, HW)
    x_ch = arg3_1.view(C, HW)
    scalar_flat = arg8_1.view(1)

    mean_1d = arg4_1.view(C)
    invstd_1d = arg5_1.view(C)

    num_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW  # ceil
    num_blocks_pad = _next_power_of_2(num_blocks)

    partial_sum = torch.empty((num_blocks, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_blocks, C), device=device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    dot_tmp = torch.empty((C,), device=device, dtype=torch.float32)
    mul10_out = torch.empty((C,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    dense_ch = dense_out.view(C, HW)
    partial_dense_sum = torch.empty((num_blocks, C), device=device, dtype=torch.float32)
    dense_sum_out = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, num_blocks, 1),
        _partial_reduce_kernel,
        (add_ch, x_ch, scalar_flat, mean_1d, invstd_1d, arg6_1, arg7_1,
         partial_sum, partial_dot, HW, BLOCK_HW),
    )

    BLOCK_C = 8
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), 1, 1),
        _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d,
         sum_out, dot_tmp, mul10_out,
         num_blocks_pad, num_blocks, C, BLOCK_C),
    )

    ct.launch(
        stream,
        (C, num_blocks, 1),
        _epilogue_kernel,
        (add_ch, x_ch, scalar_flat, mean_1d, invstd_1d, arg6_1, arg7_1,
         sum_out, dot_tmp, dense_ch, partial_dense_sum,
         HW, BLOCK_HW, REDUCTION_SCALE),
    )

    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), 1, 1),
        _finalize_dense_sum_kernel,
        (partial_dense_sum, dense_sum_out, num_blocks_pad, num_blocks, BLOCK_C),
    )

    # sum_2 = dot_tmp -- but the return is (sum_1, mul_10 = sum_2 * squeeze_1, dense, sum_3=dense_sum_bf16)
    return sum_out, mul10_out, dense_out, dense_sum_out
