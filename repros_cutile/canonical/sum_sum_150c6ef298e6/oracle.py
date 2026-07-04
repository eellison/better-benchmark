"""cuTile port of sum_sum_150c6ef298e6: phlippe_densenet BN backward + slice add.

Multi-kernel plan (matches the Triton oracle structure):
  1. `_reduce_kernel`: per-(c, k-tile) block computes partial where_f32 and
     where_f32 * centered contributions and atomic-adds them into per-channel
     `sum_where[C]` and `sum_mul[C]` accumulators.
  2. `_epilogue_kernel`: full-tensor (N, C, H, W) block computes:
        where = where(mask <= 0, full, where_rhs).f32
        centered = centered_src.f32 - mean
        term = where * C_hw_scale - sum_where*scale - centered*sum_mul*scale*invstd^2
        grad = term * invstd * weight
        out = residual[N, slice_c:slice_c+C, H, W] + grad(bf16)
     bf16 add.rn / sub.rn / mul.rn in Triton are cuTile's default rounding.

Outputs match the Triton oracle:
  sum_where[C] (f32), mul8[C] = sum_mul * invstd (f32),
  out[N, C, H, W] bf16, slice_view[N, SLICE_C, H, W] bf16 (view of out).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 128
SRC_C = 144
SLICE_C = 16
SCALE = 3.0517578125e-05


@ct.kernel
def _reduce_kernel(
    mask_ptr,          # bf16 [N, C, H, W]
    full_ptr,          # bf16 [1]
    where_rhs_ptr,     # bf16 [N, C, H, W]
    centered_src_ptr,  # bf16 [N, C, H, W]
    mean_ptr,          # f32  [C]
    sum_where_ptr,     # f32  [C]  (zeroed, atomic accumulator)
    sum_mul_ptr,       # f32  [C]  (zeroed, atomic accumulator)
    C_: ct.Constant[int],
    HW: ct.Constant[int],
    TOTAL_SPATIAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    k = ct.arange(BLOCK_K, dtype=ct.int32) + tile * BLOCK_K
    active = k < TOTAL_SPATIAL

    n = k // HW
    spatial = k - n * HW
    h = spatial // 16  # for a758b9fd HW=16*16, for c976107e HW=8*8
    # Use 4D gather: (n, c, spatial_h, spatial_w). But since HW may be
    # 256 or 64, we treat H*W as a 1D spatial and flatten via manual indexing.
    # Simpler: gather from a flat view [N*C*HW] with offset n*(C*HW)+c*HW+spatial.
    offsets = n * (C_ * HW) + c * HW + spatial
    zero_bf = ct.astype(0.0, ct.bfloat16)
    zero_f = ct.astype(0.0, ct.float32)

    mask_bf = ct.gather(mask_ptr, offsets, mask=active, padding_value=zero_bf)
    src_bf = ct.gather(where_rhs_ptr, offsets, mask=active, padding_value=zero_bf)
    full_v = ct.load(full_ptr, index=(0,), shape=(1,))
    full_bf = ct.reshape(full_v, (1,))
    # broadcast full_bf to (BLOCK_K,)
    full_bc = ct.broadcast_to(full_bf, (BLOCK_K,))
    where_bf = ct.where(mask_bf <= zero_bf, full_bc, src_bf)
    where_f32 = ct.astype(where_bf, ct.float32)

    centered_bf = ct.gather(centered_src_ptr, offsets, mask=active,
                            padding_value=zero_bf)
    centered_srcf = ct.astype(centered_bf, ct.float32)
    mean_scalar = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_bc = ct.broadcast_to(mean_scalar, (BLOCK_K,))
    centered = centered_srcf - mean_bc
    product = where_f32 * centered

    where_masked = ct.where(active, where_f32, zero_f)
    prod_masked = ct.where(active, product, zero_f)
    where_sum = ct.sum(where_masked)
    mul_sum = ct.sum(prod_masked)

    # Scalar atomic add into per-channel f32 accumulators.
    c_scalar = ct.full((1,), c, dtype=ct.int32)
    where_sum_1 = ct.reshape(where_sum, (1,))
    mul_sum_1 = ct.reshape(mul_sum, (1,))
    ct.atomic_add(sum_where_ptr, (c_scalar,), where_sum_1)
    ct.atomic_add(sum_mul_ptr, (c_scalar,), mul_sum_1)


@ct.kernel
def _epilogue_kernel(
    residual_ptr,      # bf16 [N, SRC_C, H, W]
    mask_ptr,          # bf16 [N, C, H, W]
    full_ptr,          # bf16 [1]
    where_rhs_ptr,     # bf16 [N, C, H, W]
    centered_src_ptr,  # bf16 [N, C, H, W]
    mean_ptr,          # f32  [C]
    invstd_ptr,        # f32  [C]
    weight_ptr,        # f32  [C]
    sum_where_ptr,     # f32  [C]
    sum_mul_ptr,       # f32  [C]
    out_ptr,           # bf16 [N, C, H, W]
    C_: ct.Constant[int],
    SRC_C_: ct.Constant[int],
    SLICE_C_: ct.Constant[int],
    HW: ct.Constant[int],
    TOTAL: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    off = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    active = off < TOTAL

    spatial = off - (off // HW) * HW
    c_idx = (off // HW) - ((off // HW) // C_) * C_
    n = off // (C_ * HW)
    residual_off = n * (SRC_C_ * HW) + (c_idx + SLICE_C_) * HW + spatial

    zero_bf = ct.astype(0.0, ct.bfloat16)
    zero_f = ct.astype(0.0, ct.float32)

    mask_bf = ct.gather(mask_ptr, off, mask=active, padding_value=zero_bf)
    src_bf = ct.gather(where_rhs_ptr, off, mask=active, padding_value=zero_bf)
    full_v = ct.load(full_ptr, index=(0,), shape=(1,))
    full_bf = ct.reshape(full_v, (1,))
    full_bc = ct.broadcast_to(full_bf, (BLOCK,))
    where_bf = ct.where(mask_bf <= zero_bf, full_bc, src_bf)
    where_f32 = ct.astype(where_bf, ct.float32)

    centered_bf = ct.gather(centered_src_ptr, off, mask=active, padding_value=zero_bf)
    centered_src = ct.astype(centered_bf, ct.float32)
    mean_val = ct.gather(mean_ptr, c_idx, mask=active, padding_value=zero_f)
    centered = centered_src - mean_val

    sum_where_v = ct.gather(sum_where_ptr, c_idx, mask=active, padding_value=zero_f)
    sum_mul_v = ct.gather(sum_mul_ptr, c_idx, mask=active, padding_value=zero_f)
    invstd = ct.gather(invstd_ptr, c_idx, mask=active, padding_value=zero_f)
    weight = ct.gather(weight_ptr, c_idx, mask=active, padding_value=zero_f)

    mean_term = sum_where_v * SCALE_
    sum_mul_scaled = sum_mul_v * SCALE_
    invstd_sq = invstd * invstd
    variance_term = sum_mul_scaled * invstd_sq
    centered_scaled = centered * variance_term
    sub1 = where_f32 - centered_scaled
    sub2 = sub1 - mean_term
    out_weight = invstd * weight
    grad_f = sub2 * out_weight
    grad = ct.astype(grad_f, ct.bfloat16)

    residual_bf = ct.gather(residual_ptr, residual_off, mask=active,
                            padding_value=zero_bf)
    added_f = ct.astype(residual_bf, ct.float32) + ct.astype(grad, ct.float32)
    added = ct.astype(added_f, ct.bfloat16)

    ct.scatter(out_ptr, off, added, mask=active)


def _launch(inputs, *, BLOCK_K: int, EPILOGUE_BLOCK: int):
    residual, mask, full, where_rhs, centered_src, mean, invstd, weight = inputs
    device = mask.device
    h = int(mask.shape[2])
    w = int(mask.shape[3])
    hw = h * w
    total_spatial = N * hw
    total = N * C * hw

    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_mul = torch.zeros((C,), device=device, dtype=torch.float32)

    # Views: use 1D flat views for gather.
    mask_flat = mask.contiguous().view(-1)
    where_rhs_flat = where_rhs.contiguous().view(-1)
    centered_src_flat = centered_src.contiguous().view(-1)
    residual_flat = residual.contiguous().view(-1)
    mean_1d = mean.view(C)
    full_1d = full.view(1)

    out = torch.empty_strided((N, C, h, w), (C * hw, hw, w, 1),
                              device=device, dtype=torch.bfloat16)
    out_flat = out.view(-1)

    num_tiles = (total_spatial + BLOCK_K - 1) // BLOCK_K
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, num_tiles, 1), _reduce_kernel,
        (mask_flat, full_1d, where_rhs_flat, centered_src_flat, mean_1d,
         sum_where, sum_mul,
         C, hw, total_spatial, BLOCK_K),
    )

    mul8 = sum_mul * invstd

    num_epi_tiles = (total + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK
    ct.launch(
        stream, (num_epi_tiles, 1, 1), _epilogue_kernel,
        (residual_flat, mask_flat, full_1d, where_rhs_flat, centered_src_flat,
         mean_1d, invstd, weight, sum_where, sum_mul, out_flat,
         C, SRC_C, SLICE_C, hw, total, SCALE, EPILOGUE_BLOCK),
    )

    slice_view = torch.as_strided(out, (N, SLICE_C, h, w), (C * hw, hw, w, 1))
    return sum_where, mul8, out, slice_view


@oracle_impl(hardware="B200", point="a758b9fd", BLOCK_K=1024, EPILOGUE_BLOCK=256,
             reduce_warps=4, final_warps=4, epilogue_warps=4)
@oracle_impl(hardware="B200", point="c976107e", BLOCK_K=1024, EPILOGUE_BLOCK=256,
             reduce_warps=4, final_warps=4, epilogue_warps=4)
def oracle_forward(inputs, *, BLOCK_K, EPILOGUE_BLOCK,
                   reduce_warps, final_warps, epilogue_warps):
    del reduce_warps, final_warps, epilogue_warps
    return _launch(inputs, BLOCK_K=BLOCK_K, EPILOGUE_BLOCK=EPILOGUE_BLOCK)
