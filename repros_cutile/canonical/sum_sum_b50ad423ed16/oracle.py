"""cuTile port of sum_sum_b50ad423ed16: GhostNet BN-backward-like scope.

The scope produces:
  sum_1  = sum(where.f32, dim=[0,2,3])       # f32[C]
  mul_8  = sum(where * (arg4 - mean), ...) * arg6   # f32[C]
  out_bf = bf16 dense gradient tensor (channels-last)

C=36 is not power-of-2, so we use BLOCK_C=64 (padded) with a channel mask.
Three cuTile launches: partial reduce, finalize, epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C = 36
IN_C = 72
H = 56
W = 56
HW = H * W
R = N * HW           # 1_605_632
NUMEL = N * C * HW   # 57_802_752
SCALE = 6.228077168367346e-07


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _reduce_kernel(
    wide_ptr,        # bf16 [R, IN_C]
    rhs_ptr,         # bf16 [R, C]
    mask_ptr,        # bf16 [R, C]
    fill_ptr,        # bf16 [1]
    activation_ptr,  # bf16 [R, C]
    mean_ptr,        # f32 [C]  (padded read of BLOCK_C)
    partial_where_ptr,  # f32 [NUM_CHUNKS, BLOCK_C]
    partial_dot_ptr,    # f32 [NUM_CHUNKS, BLOCK_C]
    C_C: ct.Constant[int],
    IN_C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    CHUNK_R: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    chunk = ct.bid(0)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_scalar = ct.sum(fill)
    fill_f32 = ct.astype(fill_scalar, ct.float32)
    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))

    c_idx = ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask = ct.reshape(c_idx < C_C, (1, BLOCK_C))

    acc_where = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)
    acc_dot = ct.full(shape=(1, BLOCK_C), fill_value=0.0, dtype=ct.float32)

    tiles_per_chunk = CHUNK_R // BLOCK_R
    for r_off in range(tiles_per_chunk):
        r_tile = chunk * tiles_per_chunk + r_off
        # wide is [R, IN_C=72]. Load a (BLOCK_R, BLOCK_C=64) tile - covers
        # the first BLOCK_C channels (which is < IN_C).
        lhs = ct.load(wide_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
        # rhs/mask/activation are [R, C=36]. Load with padding since BLOCK_C > C.
        rhs = ct.load(rhs_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
        mask_t = ct.load(mask_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
        activation = ct.load(activation_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                             padding_mode=ct.PaddingMode.ZERO)

        lhs_f = ct.astype(lhs, ct.float32)
        rhs_f = ct.astype(rhs, ct.float32)
        add_val = lhs_f + rhs_f
        mask_f = ct.astype(mask_t, ct.float32)
        where_val = ct.where(mask_f <= 0.0, fill_f32, add_val)
        # Mask out padded channels before accumulation.
        where_val = ct.where(c_mask, where_val, 0.0)
        activation_f = ct.astype(activation, ct.float32)
        centered = activation_f - mean_2d
        centered = ct.where(c_mask, centered, 0.0)
        acc_where = acc_where + ct.sum(where_val, axis=0, keepdims=True)
        acc_dot = acc_dot + ct.sum(where_val * centered, axis=0, keepdims=True)

    ct.store(partial_where_ptr, index=(chunk, 0), tile=acc_where)
    ct.store(partial_dot_ptr, index=(chunk, 0), tile=acc_dot)


@ct.kernel
def _finalize_kernel(
    partial_where_ptr,   # f32 [NUM_CHUNKS, BLOCK_C]
    partial_dot_ptr,     # f32 [NUM_CHUNKS, BLOCK_C]
    invstd_ptr,          # f32 [C]
    weight_ptr,          # f32 [C]
    sum_out_ptr,         # f32 [C]
    vec_out_ptr,         # f32 [C]
    mean_term_ptr,       # f32 [C]
    dot_coeff_ptr,       # f32 [C]
    out_scale_ptr,       # f32 [C]
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    where_partials = ct.load(
        partial_where_ptr, index=(0, 0), shape=(BLOCK_CHUNKS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    dot_partials = ct.load(
        partial_dot_ptr, index=(0, 0), shape=(BLOCK_CHUNKS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    chunk_idx = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    active = ct.reshape(chunk_idx < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    where_partials = ct.where(active, where_partials, 0.0)
    dot_partials = ct.where(active, dot_partials, 0.0)
    sum_where = ct.sum(where_partials, axis=0)
    sum_dot = ct.sum(dot_partials, axis=0)

    invstd = ct.load(invstd_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    dot_scaled = sum_dot * SCALE_
    invstd_sq = invstd * invstd

    # Stores truncate to C automatically since sum_out has shape (C,).
    ct.store(sum_out_ptr, index=(0,), tile=sum_where)
    ct.store(vec_out_ptr, index=(0,), tile=sum_dot * invstd)
    ct.store(mean_term_ptr, index=(0,), tile=sum_where * SCALE_)
    ct.store(dot_coeff_ptr, index=(0,), tile=dot_scaled * invstd_sq)
    ct.store(out_scale_ptr, index=(0,), tile=invstd * weight)


@ct.kernel
def _epilogue_kernel(
    wide_ptr,        # bf16 [R, IN_C]
    rhs_ptr,         # bf16 [R, C]
    mask_ptr,        # bf16 [R, C]
    fill_ptr,        # bf16 [1]
    activation_ptr,  # bf16 [R, C]
    mean_ptr,        # f32 [C]
    mean_term_ptr,   # f32 [C]
    dot_coeff_ptr,   # f32 [C]
    out_scale_ptr,   # f32 [C]
    out_ptr,         # bf16 [R, C]
    C_C: ct.Constant[int],
    IN_C_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    r_tile = ct.bid(0)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_scalar = ct.sum(fill)
    fill_f32 = ct.astype(fill_scalar, ct.float32)
    fill_bf16 = ct.astype(fill_scalar, ct.bfloat16)
    fill_bf16_as_f32 = ct.astype(fill_bf16, ct.float32)

    mean = ct.load(mean_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    mean_2d = ct.reshape(mean, (1, BLOCK_C))
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    mean_term_2d = ct.reshape(mean_term, (1, BLOCK_C))
    dot_coeff = ct.load(dot_coeff_ptr, index=(0,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    dot_coeff_2d = ct.reshape(dot_coeff, (1, BLOCK_C))
    out_scale = ct.load(out_scale_ptr, index=(0,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    out_scale_2d = ct.reshape(out_scale, (1, BLOCK_C))

    lhs = ct.load(wide_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                  padding_mode=ct.PaddingMode.ZERO)
    rhs = ct.load(rhs_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                  padding_mode=ct.PaddingMode.ZERO)
    mask_t = ct.load(mask_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    activation = ct.load(activation_ptr, index=(r_tile, 0), shape=(BLOCK_R, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)

    lhs_f = ct.astype(lhs, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    add_f = lhs_f + rhs_f
    add_bf = ct.astype(add_f, ct.bfloat16)
    mask_f = ct.astype(mask_t, ct.float32)
    keep = mask_f <= 0.0
    where_f = ct.where(keep, fill_f32, add_f)
    where_bf_f = ct.where(keep, fill_bf16_as_f32, ct.astype(add_bf, ct.float32))

    activation_f = ct.astype(activation, ct.float32)
    centered = activation_f - mean_2d
    correction = centered * dot_coeff_2d
    after_correction_f = where_f - correction
    after_mean_f = after_correction_f - mean_term_2d
    out_f32_full = after_mean_f * out_scale_2d
    out_f32 = ct.astype(out_f32_full, ct.bfloat16)

    after_correction_bf = where_bf_f - correction
    after_mean_bf = after_correction_bf - mean_term_2d
    out_bf_full = after_mean_bf * out_scale_2d
    out_bf = ct.astype(out_bf_full, ct.bfloat16)

    out_bf_abs = ct.astype(out_bf, ct.float32)
    condition = ct.where(out_bf_abs >= 0.0, out_bf_abs, -out_bf_abs) >= 4.0
    out_final = ct.where(condition, out_f32, out_bf)

    ct.store(out_ptr, index=(r_tile, 0), tile=out_final)


@oracle_impl(
    hardware="B200",
    point="4c3b4612",
    CHUNK_R=1024,
    BLOCK_R=64,
    EPILOGUE_BLOCK_R=64,
    BLOCK_C=64,
)
def oracle_forward(
    inputs,
    *,
    CHUNK_R: int,
    BLOCK_R: int,
    EPILOGUE_BLOCK_R: int,
    BLOCK_C: int,
):
    wide, rhs, mask, fill, activation, mean, invstd, weight = inputs
    device = wide.device

    # Channels-last views expose element addressing as [R, C_dim] with
    # row-major C_dim stride 1 and row stride C_dim.
    wide_2d = torch.as_strided(wide, (R, IN_C), (IN_C, 1))
    rhs_2d = torch.as_strided(rhs, (R, C), (C, 1))
    mask_2d = torch.as_strided(mask, (R, C), (C, 1))
    activation_2d = torch.as_strided(activation, (R, C), (C, 1))

    mean_1d = mean.reshape(C)
    fill_1d = fill.reshape(1)

    num_chunks = (R + CHUNK_R - 1) // CHUNK_R
    block_chunks = _next_power_of_2(num_chunks)

    partial_where = torch.empty((num_chunks, BLOCK_C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_chunks, BLOCK_C), device=device, dtype=torch.float32)
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    vec_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    dot_coeff = torch.empty((C,), device=device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, C * W, C),
        device=device,
        dtype=torch.bfloat16,
    )
    dense_out_2d = torch.as_strided(dense_out, (R, C), (C, 1))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_chunks, 1, 1),
        _reduce_kernel,
        (wide_2d, rhs_2d, mask_2d, fill_1d, activation_2d, mean_1d,
         partial_where, partial_dot, C, IN_C, BLOCK_C, CHUNK_R, BLOCK_R),
    )
    ct.launch(
        stream, (1, 1, 1),
        _finalize_kernel,
        (partial_where, partial_dot, invstd, weight, sum_out, vec_out,
         mean_term, dot_coeff, out_scale,
         num_chunks, block_chunks, C, BLOCK_C, SCALE),
    )
    ct.launch(
        stream, ((R + EPILOGUE_BLOCK_R - 1) // EPILOGUE_BLOCK_R, 1, 1),
        _epilogue_kernel,
        (wide_2d, rhs_2d, mask_2d, fill_1d, activation_2d, mean_1d,
         mean_term, dot_coeff, out_scale, dense_out_2d,
         C, IN_C, BLOCK_C, EPILOGUE_BLOCK_R),
    )
    return sum_out, vec_out, dense_out
