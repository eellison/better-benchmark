"""cuTile port of sum_sum_6e7e9c6fec80: DenseNet BN backward tail (channels=960).

Cooperative style with two cuTile kernels launched sequentially:
1. Per-channel BN backward: sibling reductions, dense bf16 gradient.
2. Slice-range residual add (channels 928:960).

Inline PTX RTNE arithmetic (add.rn.f32/mul.rn.f32/sub.rn.f32) is expressed
with plain +/*/- since cuTile is RTNE by default. Two supported shape
points (14x14 and 7x7).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 960
SLICE_START = 928
SLICE_C = 32
RESIDUAL0_C = 1024
RESIDUAL1_C = 992
SCALE = 0.0012755102040816326


@ct.kernel
def _bn_tail_kernel(
    mask_ptr,          # bf16 (N, C, H, W)
    fill_ptr,          # bf16 (1,)
    source_ptr,        # bf16 (N, C, H, W)
    centered_ptr,      # bf16 (N, C, H, W)
    mean_ptr,          # f32 (C,)
    scale_ptr,         # f32 (C,)
    affine_ptr,        # f32 (C,)
    sum_where_ptr,     # f32 (C,)
    scaled_sum_ptr,    # f32 (C,)
    grad_ptr,          # bf16 (N, C, H, W)
    HW_C: ct.Constant[int],
    R_C: ct.Constant[int],
    W_C: ct.Constant[int],
    BLOCK_R_C: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R_C, dtype=ct.int32)
    active = rows < R_C
    n = rows // HW_C
    spatial = rows - n * HW_C
    h_idx = spatial // W_C
    w_idx = spatial - h_idx * W_C

    zero_i = ct.zeros((BLOCK_R_C,), dtype=ct.int32)
    n_safe = ct.where(active, n, zero_i)
    h_safe = ct.where(active, h_idx, zero_i)
    w_safe = ct.where(active, w_idx, zero_i)
    c_full = ct.full((BLOCK_R_C,), c, dtype=ct.int32)

    mask_bf = ct.gather(mask_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    source_bf = ct.gather(source_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    centered_bf = ct.gather(centered_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)

    fill_bf = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_tile = ct.full((BLOCK_R_C,), 0.0, dtype=ct.bfloat16) + fill_bf

    zero_f = ct.full((BLOCK_R_C,), 0.0, dtype=ct.float32)
    zero_bf = ct.full((BLOCK_R_C,), 0.0, dtype=ct.bfloat16)

    where_value_bf = ct.where(mask_bf <= zero_bf, fill_tile, source_bf)
    where_value = ct.astype(where_value_bf, ct.float32)
    where_active = ct.where(active, where_value, zero_f)

    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    centered_f = ct.astype(centered_bf, ct.float32) - mean_1
    centered = ct.where(active, centered_f, zero_f)

    product = where_value * centered
    sum_where = ct.sum(where_active)
    sum_centered = ct.sum(ct.where(active, product, zero_f))

    scale_1 = ct.load(scale_ptr, index=(c,), shape=(1,))
    affine_1 = ct.load(affine_ptr, index=(c,), shape=(1,))

    mean_term = sum_where * SCALE_C
    centered_scale = sum_centered * SCALE_C
    scale_sq = scale_1 * scale_1
    variance_term = centered_scale * scale_sq
    affine_term = scale_1 * affine_1

    correction = centered * variance_term
    tmp = where_value - correction
    tmp = tmp - mean_term
    grad = ct.astype(tmp * affine_term, ct.bfloat16)

    ct.scatter(grad_ptr, (n_safe, c_full, h_safe, w_safe), grad, mask=active)

    sum_where_1 = ct.full((1,), sum_where, dtype=ct.float32)
    scaled_sum_1 = sum_centered * scale_1  # (1,)
    ct.store(sum_where_ptr, index=(c,), tile=sum_where_1)
    ct.store(scaled_sum_ptr, index=(c,), tile=scaled_sum_1)


@ct.kernel
def _residual_add_kernel(
    residual0_ptr,  # bf16 (N, RESIDUAL0_C, H, W)
    residual1_ptr,  # bf16 (N, RESIDUAL1_C, H, W)
    dense_ptr,      # bf16 (N, C, H, W)
    add_out_ptr,    # bf16 (N, SLICE_C, H, W)
    HW_C: ct.Constant[int],
    R_C: ct.Constant[int],
    W_C: ct.Constant[int],
    BLOCK_R_C: ct.Constant[int],
):
    c = ct.bid(0)  # 0..SLICE_C-1
    rows = ct.arange(BLOCK_R_C, dtype=ct.int32)
    active = rows < R_C
    n = rows // HW_C
    spatial = rows - n * HW_C
    h_idx = spatial // W_C
    w_idx = spatial - h_idx * W_C

    zero_i = ct.zeros((BLOCK_R_C,), dtype=ct.int32)
    n_safe = ct.where(active, n, zero_i)
    h_safe = ct.where(active, h_idx, zero_i)
    w_safe = ct.where(active, w_idx, zero_i)

    dense_c = ct.full((BLOCK_R_C,), c + SLICE_START, dtype=ct.int32)
    r0_c = ct.full((BLOCK_R_C,), c + SLICE_START, dtype=ct.int32)
    r1_c = ct.full((BLOCK_R_C,), c + SLICE_START, dtype=ct.int32)
    out_c = ct.full((BLOCK_R_C,), c, dtype=ct.int32)

    r0 = ct.gather(residual0_ptr, (n_safe, r0_c, h_safe, w_safe), mask=active)
    r1 = ct.gather(residual1_ptr, (n_safe, r1_c, h_safe, w_safe), mask=active)
    dense_v = ct.gather(dense_ptr, (n_safe, dense_c, h_safe, w_safe), mask=active)

    r0_f = ct.astype(r0, ct.float32)
    r1_f = ct.astype(r1, ct.float32)
    first_bf = ct.astype(r0_f + r1_f, ct.bfloat16)
    second_bf = ct.astype(
        ct.astype(first_bf, ct.float32) + ct.astype(dense_v, ct.float32),
        ct.bfloat16,
    )
    ct.scatter(add_out_ptr, (n_safe, out_c, h_safe, w_safe), second_bf, mask=active)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _launch(inputs, *, BLOCK_R: int, num_warps: int, num_stages: int):
    del num_warps, num_stages  # cuTile chooses these
    (
        residual0,
        residual1,
        mask_input,
        full,
        source,
        centered_source,
        mean,
        scale,
        affine,
    ) = inputs
    n = int(mask_input.shape[0])
    h = int(mask_input.shape[2])
    w = int(mask_input.shape[3])
    hw = h * w
    total_spatial = n * hw

    device = mask_input.device
    sum_where = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scaled_sum_centered = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    grad = torch.empty_strided(
        (n, C, h, w),
        _contiguous_stride((n, C, h, w)),
        device=device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (n, SLICE_C, h, w),
        _contiguous_stride((n, SLICE_C, h, w)),
        device=device,
        dtype=torch.bfloat16,
    )

    mean_1d = mean.view(C)
    scale_1d = scale.view(C)
    affine_1d = affine.view(C)
    full_1d = full.view(1)

    block_r = _next_pow2(total_spatial)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_tail_kernel,
        (mask_input, full_1d, source, centered_source, mean_1d, scale_1d, affine_1d,
         sum_where, scaled_sum_centered, grad,
         hw, total_spatial, w, block_r, SCALE),
    )
    ct.launch(
        stream,
        (SLICE_C, 1, 1),
        _residual_add_kernel,
        (residual0, residual1, grad, add_out,
         hw, total_spatial, w, block_r),
    )
    return sum_where, scaled_sum_centered, grad, add_out


@oracle_impl(hardware="B200", point="0b1ad29c", BLOCK_R=1024, num_warps=8, num_stages=4)
@oracle_impl(hardware="B200", point="69c44c42", BLOCK_R=256, num_warps=4, num_stages=4)
def oracle_forward(inputs, *, BLOCK_R: int, num_warps: int, num_stages: int):
    return _launch(inputs, BLOCK_R=BLOCK_R, num_warps=num_warps, num_stages=num_stages)
