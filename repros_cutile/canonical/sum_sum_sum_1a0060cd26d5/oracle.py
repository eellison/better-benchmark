"""cuTile port of sum_sum_sum_1a0060cd26d5: UNet BN-backward with maxpool scatter.

Strategy: precompute the tricky scatter_add via torch (giving `add`), then
run cuTile kernels for the BN-backward math:
  * Reduce kernel: one program per (channel_group, hw_tile) writing partial
    sum/dot into (n_tiles, C) buffers.
  * Finalize kernel: sum partial_sum/partial_dot across tiles into (C,) buffers.
  * Epilogue kernel: produce bf16 grad and its per-channel f32 sum-of-bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 64
OUT_H = 640
OUT_W = 959
SRC_H = 320
SRC_W = 479
OUT_HW = OUT_H * OUT_W          # 613760
SRC_HW = SRC_H * SRC_W          # 153280
SCALE = 1.6293013555787278e-06
BLOCK_HW = 256                  # HW tile size — must be pow2
BLOCK_C = 64                    # process all 64 channels per program (pow2)
N_HW_TILES = (OUT_HW + BLOCK_HW - 1) // BLOCK_HW  # 2398 (not pow2)


@ct.kernel
def _reduce_kernel(
    add_ptr,      # bf16 [C, OUT_HW]
    source_ptr,   # bf16 [C, OUT_HW]
    mean_ptr,     # f32  [C]
    invstd_ptr,   # f32  [C]
    weight_ptr,   # f32  [C]
    bias_ptr,     # f32  [C]
    fill_ptr,     # bf16 scalar
    partial_sum_ptr,   # f32 [n_tiles, C]
    partial_dot_ptr,   # f32 [n_tiles, C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    tile = ct.bid(0)             # tile of HW (tile-space index)

    # Load per-channel scalars for all C channels
    c_idx = ct.arange(BLOCK_C_, dtype=ct.int32)  # (BLOCK_C,)
    c_active = c_idx < C_
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C_,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C_,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C_,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C_,),
                   padding_mode=ct.PaddingMode.ZERO)
    fill_v = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_v, ct.float32)
    fill_s = ct.reshape(fill_f, (1,))

    # Load 2D tile shape (BLOCK_C, BLOCK_HW)
    add_bf = ct.load(add_ptr, index=(0, tile), shape=(BLOCK_C_, BLOCK_HW_),
                     padding_mode=ct.PaddingMode.ZERO)
    src_bf = ct.load(source_ptr, index=(0, tile), shape=(BLOCK_C_, BLOCK_HW_),
                     padding_mode=ct.PaddingMode.ZERO)
    add_f = ct.astype(add_bf, ct.float32)
    src_f = ct.astype(src_bf, ct.float32)

    mean_2d = ct.reshape(mean, (BLOCK_C_, 1))
    invstd_2d = ct.reshape(invstd, (BLOCK_C_, 1))
    weight_2d = ct.reshape(weight, (BLOCK_C_, 1))
    bias_2d = ct.reshape(bias, (BLOCK_C_, 1))

    centered = src_f - mean_2d
    affine = centered * invstd_2d * weight_2d + bias_2d
    gate = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    scatter_skip = ct.astype(ct.astype(add_f, ct.bfloat16), ct.float32)
    where_val = ct.where(gate <= ct.full((BLOCK_C_, BLOCK_HW_), 0.0, dtype=ct.float32),
                         fill_s, scatter_skip)

    # Mask by hw active
    hw_off = ct.arange(BLOCK_HW_, dtype=ct.int32) + tile * BLOCK_HW_
    hw_active = hw_off < HW_
    active_2d = ct.reshape(c_active, (BLOCK_C_, 1)) & ct.reshape(hw_active, (1, BLOCK_HW_))
    zero_2d = ct.full((BLOCK_C_, BLOCK_HW_), 0.0, dtype=ct.float32)
    where_masked = ct.where(active_2d, where_val, zero_2d)
    dot_masked = ct.where(active_2d, where_val * centered, zero_2d)

    row_sum = ct.sum(where_masked, axis=1)  # (BLOCK_C,)
    row_dot = ct.sum(dot_masked, axis=1)

    ct.store(partial_sum_ptr, index=(tile, 0), tile=ct.reshape(row_sum, (1, BLOCK_C_)))
    ct.store(partial_dot_ptr, index=(tile, 0), tile=ct.reshape(row_dot, (1, BLOCK_C_)))


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,   # f32 [n_tiles, C]
    partial_dot_ptr,   # f32 [n_tiles, C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    mul10_out_ptr,     # f32 [C]
    mean_term_ptr,     # f32 [C]
    correction_scale_ptr,  # f32 [C]
    output_scale_ptr,      # f32 [C]
    N_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    tiles_arr = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tiles_active = tiles_arr < N_TILES  # (BLOCK_TILES,)

    ps = ct.load(partial_sum_ptr, index=(0, 0), shape=(BLOCK_TILES, BLOCK_C_),
                 padding_mode=ct.PaddingMode.ZERO)
    pd = ct.load(partial_dot_ptr, index=(0, 0), shape=(BLOCK_TILES, BLOCK_C_),
                 padding_mode=ct.PaddingMode.ZERO)
    zero_2d = ct.full((BLOCK_TILES, BLOCK_C_), 0.0, dtype=ct.float32)
    active_2d = ct.reshape(tiles_active, (BLOCK_TILES, 1)) & ct.full(
        (BLOCK_TILES, BLOCK_C_), True, dtype=ct.bool_)
    ps_m = ct.where(active_2d, ps, zero_2d)
    pd_m = ct.where(active_2d, pd, zero_2d)

    sum_exact = ct.sum(ps_m, axis=0)   # (BLOCK_C,)
    dot_exact = ct.sum(pd_m, axis=0)

    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C_,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C_,),
                     padding_mode=ct.PaddingMode.ZERO)

    mean_term = sum_exact * SCALE_
    invstd_sq = invstd * invstd
    correction_scale = dot_exact * SCALE_ * invstd_sq
    output_scale = invstd * weight
    mul10 = dot_exact * invstd

    ct.store(sum_out_ptr, index=(0,), tile=sum_exact)
    ct.store(mul10_out_ptr, index=(0,), tile=mul10)
    ct.store(mean_term_ptr, index=(0,), tile=mean_term)
    ct.store(correction_scale_ptr, index=(0,), tile=correction_scale)
    ct.store(output_scale_ptr, index=(0,), tile=output_scale)


@ct.kernel
def _epilogue_kernel(
    add_ptr,          # bf16 [C, OUT_HW]
    source_ptr,       # bf16 [C, OUT_HW]
    mean_ptr,         # f32 [C]
    invstd_ptr,       # f32 [C]  (unused; kept for parity)
    weight_ptr,       # f32 [C]
    bias_ptr,         # f32 [C]
    fill_ptr,         # bf16 scalar
    mean_term_ptr,    # f32 [C]
    correction_scale_ptr,  # f32 [C]
    output_scale_ptr,      # f32 [C]
    out_ptr,          # bf16 [C, OUT_HW]
    partial_out_sum_ptr,   # f32 [n_tiles, C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    tile = ct.bid(0)

    c_idx = ct.arange(BLOCK_C_, dtype=ct.int32)
    c_active = c_idx < C_
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C_,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C_,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C_,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C_,),
                   padding_mode=ct.PaddingMode.ZERO)
    fill_v = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_v, ct.float32)
    fill_s = ct.reshape(fill_f, (1,))

    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(BLOCK_C_,),
                        padding_mode=ct.PaddingMode.ZERO)
    correction_scale = ct.load(correction_scale_ptr, index=(0,), shape=(BLOCK_C_,),
                               padding_mode=ct.PaddingMode.ZERO)
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(BLOCK_C_,),
                           padding_mode=ct.PaddingMode.ZERO)

    add_bf = ct.load(add_ptr, index=(0, tile), shape=(BLOCK_C_, BLOCK_HW_),
                     padding_mode=ct.PaddingMode.ZERO)
    src_bf = ct.load(source_ptr, index=(0, tile), shape=(BLOCK_C_, BLOCK_HW_),
                     padding_mode=ct.PaddingMode.ZERO)
    add_f = ct.astype(add_bf, ct.float32)
    src_f = ct.astype(src_bf, ct.float32)

    mean_2d = ct.reshape(mean, (BLOCK_C_, 1))
    invstd_2d = ct.reshape(invstd, (BLOCK_C_, 1))
    weight_2d = ct.reshape(weight, (BLOCK_C_, 1))
    bias_2d = ct.reshape(bias, (BLOCK_C_, 1))
    mean_term_2d = ct.reshape(mean_term, (BLOCK_C_, 1))
    correction_scale_2d = ct.reshape(correction_scale, (BLOCK_C_, 1))
    output_scale_2d = ct.reshape(output_scale, (BLOCK_C_, 1))

    centered = src_f - mean_2d
    affine = centered * invstd_2d * weight_2d + bias_2d
    gate = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    scatter_skip = ct.astype(ct.astype(add_f, ct.bfloat16), ct.float32)
    where_val = ct.where(gate <= ct.full((BLOCK_C_, BLOCK_HW_), 0.0, dtype=ct.float32),
                         fill_s, scatter_skip)

    correction = centered * correction_scale_2d
    grad_pre = (where_val - correction) - mean_term_2d
    grad = grad_pre * output_scale_2d
    grad_bf = ct.astype(grad, ct.bfloat16)
    ct.store(out_ptr, index=(0, tile), tile=grad_bf)

    hw_off = ct.arange(BLOCK_HW_, dtype=ct.int32) + tile * BLOCK_HW_
    hw_active = hw_off < HW_
    active_2d = ct.reshape(c_active, (BLOCK_C_, 1)) & ct.reshape(hw_active, (1, BLOCK_HW_))
    zero_2d = ct.full((BLOCK_C_, BLOCK_HW_), 0.0, dtype=ct.float32)
    grad_f = ct.astype(grad_bf, ct.float32)
    grad_masked = ct.where(active_2d, grad_f, zero_2d)
    row_partial = ct.sum(grad_masked, axis=1)  # (BLOCK_C,)
    ct.store(partial_out_sum_ptr, index=(tile, 0), tile=ct.reshape(row_partial, (1, BLOCK_C_)))


@ct.kernel
def _finalize_out_sum_kernel(
    partial_out_sum_ptr,  # f32 [n_tiles, C]
    out_sum_ptr,          # f32 [C]
    N_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    tiles_arr = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tiles_active = tiles_arr < N_TILES

    ps = ct.load(partial_out_sum_ptr, index=(0, 0), shape=(BLOCK_TILES, BLOCK_C_),
                 padding_mode=ct.PaddingMode.ZERO)
    zero_2d = ct.full((BLOCK_TILES, BLOCK_C_), 0.0, dtype=ct.float32)
    active_2d = ct.reshape(tiles_active, (BLOCK_TILES, 1)) & ct.full(
        (BLOCK_TILES, BLOCK_C_), True, dtype=ct.bool_)
    ps_m = ct.where(active_2d, ps, zero_2d)
    total = ct.sum(ps_m, axis=0)  # (BLOCK_C,)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(out_sum_ptr, index=(0,), tile=rounded)


def _next_pow2(n):
    return 1 << (int(n) - 1).bit_length()


@oracle_impl(
    hardware="B200", point="6ea2f11f",
    REDUCE_BLOCK_HW=256, REDUCE_BLOCK_C=8, FINAL_BLOCK_C=8,
    EPILOGUE_BLOCK_HW=256, EPILOGUE_BLOCK_C=8, OUT_SUM_BLOCK_C=8,
)
def oracle_forward(inputs, **_kwargs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
        *_shape_params,
    ) = inputs
    device = arg0_1.device

    # Precompute scatter_add via torch
    skip = arg0_1[:, 0:C, :, :].contiguous()
    full_zero = torch.zeros((C, OUT_HW), dtype=torch.float32, device=device)
    view_arg1 = arg1_1.view(C, SRC_HW)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg2_1, [2, 2], [OUT_H, OUT_W], [2, 2], [0, 0], [1, 1],
    ).view(C, SRC_HW)
    values = view_arg1.to(torch.float32)
    scatter_add = torch.scatter_add(full_zero, 1, indices, values).view(1, C, OUT_H, OUT_W)
    scatter_bf = scatter_add.to(torch.bfloat16)
    add = skip + scatter_bf  # bf16 [1, 64, 640, 959]

    add_2d = add.view(C, OUT_HW)
    source_2d = arg3_1.view(C, OUT_HW)
    mean_1d = arg4_1.view(C)
    invstd_1d = arg5_1.view(C)
    weight_1d = arg6_1
    bias_1d = arg7_1
    fill_view = arg8_1.view(1)

    n_hw_tiles = (OUT_HW + BLOCK_HW - 1) // BLOCK_HW  # 2398
    partial_sum = torch.zeros((n_hw_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.zeros((n_hw_tiles, C), device=device, dtype=torch.float32)
    sum_out = torch.zeros((C,), device=device, dtype=torch.float32)
    mul10_out = torch.zeros((C,), device=device, dtype=torch.float32)
    mean_term = torch.zeros((C,), device=device, dtype=torch.float32)
    correction_scale = torch.zeros((C,), device=device, dtype=torch.float32)
    output_scale = torch.zeros((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_hw_tiles, 1, 1), _reduce_kernel,
        (add_2d, source_2d, mean_1d, invstd_1d, weight_1d, bias_1d, fill_view,
         partial_sum, partial_dot, C, OUT_HW, BLOCK_HW, C),
    )

    block_tiles = _next_pow2(n_hw_tiles)  # 4096
    ct.launch(
        stream, (1, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, invstd_1d, weight_1d,
         sum_out, mul10_out, mean_term, correction_scale, output_scale,
         n_hw_tiles, block_tiles, C, C, SCALE),
    )

    out_tensor = torch.empty_strided(
        (1, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_2d = out_tensor.view(C, OUT_HW)
    partial_out_sum = torch.zeros((n_hw_tiles, C), device=device, dtype=torch.float32)
    ct.launch(
        stream, (n_hw_tiles, 1, 1), _epilogue_kernel,
        (add_2d, source_2d, mean_1d, invstd_1d, weight_1d, bias_1d, fill_view,
         mean_term, correction_scale, output_scale,
         out_2d, partial_out_sum, C, OUT_HW, BLOCK_HW, C),
    )

    sum3_out = torch.zeros((C,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (1, 1, 1), _finalize_out_sum_kernel,
        (partial_out_sum, sum3_out, n_hw_tiles, block_tiles, C, C),
    )

    return sum_out, mul10_out, out_tensor, sum3_out
