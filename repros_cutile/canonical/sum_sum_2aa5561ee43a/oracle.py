"""cuTile port of sum_sum_2aa5561ee43a: DenseNet BN backward tail.

One channel-resident cuTile program per output channel; shares the masked
`where(mask <= 0, fill, source)` producer across both f32 channel reductions,
writes the full bf16 gradient tensor, and emits the returned bf16
channel-416:448 residual add.

Inline PTX (add.rn.f32/mul.rn.f32/sub.rn.f32/RTNE bf16 cast) is expressed via
plain arithmetic since cuTile is RTNE by default.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 448
H = 28
W = 28
HW = H * W
R = N * HW
SLICE_START = 416
SLICE_C = C - SLICE_START  # 32
RESIDUAL0_C = 512
RESIDUAL1_C = 480
SCALE = 0.00031887755102040814
# Next power of two >= R (3136) is 4096 for cuTile tile shapes.
BLOCK_R = 4096


@ct.kernel
def _bn_tail_kernel(
    mask_ptr,          # bf16 (N, C, H, W) contiguous flattened -> (R, C)? use (C, R)
    fill_ptr,          # bf16 scalar
    source_ptr,        # bf16 (N, C, H, W)
    centered_ptr,      # bf16 (N, C, H, W)
    mean_ptr,          # f32 (C,)
    invstd_ptr,        # f32 (C,)
    weight_ptr,        # f32 (C,)
    sum_out_ptr,       # f32 (C,)
    scale_grad_ptr,    # f32 (C,)
    dense_out_ptr,     # bf16 (N, C, H, W)
    R_C: ct.Constant[int],
    BLOCK_R_C: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R_C, dtype=ct.int32)
    active = rows < R_C

    # (n, hw) indexing into (C, R)-flat layout.
    n = rows // HW
    spatial = rows - n * HW
    # offsets = n * (C * HW) + c * HW + spatial in the (N, C, H, W) flat layout
    # We load using ct.gather with 4D indices (n, c, h, w).
    h_idx = spatial // W
    w_idx = spatial - h_idx * W
    c_full = ct.full((BLOCK_R_C,), c, dtype=ct.int32)

    # Zero-safe indices for OOB rows (rows >= R) — bounds masked out anyway.
    n_safe = ct.where(active, n, ct.zeros((BLOCK_R_C,), dtype=ct.int32))
    h_safe = ct.where(active, h_idx, ct.zeros((BLOCK_R_C,), dtype=ct.int32))
    w_safe = ct.where(active, w_idx, ct.zeros((BLOCK_R_C,), dtype=ct.int32))

    mask_bf = ct.gather(mask_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    source_bf = ct.gather(source_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    centered_bf = ct.gather(centered_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)

    fill_bf = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_scalar_bf = ct.reshape(fill_bf, (1,))
    fill_tile = ct.full((BLOCK_R_C,), 0.0, dtype=ct.bfloat16) + fill_scalar_bf

    zero_bf = ct.full((BLOCK_R_C,), 0.0, dtype=ct.bfloat16)
    where_bf = ct.where(mask_bf <= zero_bf, fill_tile, source_bf)
    where_f = ct.astype(where_bf, ct.float32)
    zero_f = ct.full((BLOCK_R_C,), 0.0, dtype=ct.float32)
    where_active = ct.where(active, where_f, zero_f)

    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_scalar = ct.reshape(mean_1, (1,))
    centered_f = ct.astype(centered_bf, ct.float32) - mean_scalar
    centered = ct.where(active, centered_f, zero_f)

    product = where_active * centered
    sum_where = ct.sum(where_active)
    sum_centered = ct.sum(product)

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd_scalar = ct.reshape(invstd_1, (1,))
    weight_scalar = ct.reshape(weight_1, (1,))

    mean_term = sum_where * SCALE_C
    dot_scaled = sum_centered * SCALE_C
    invstd_sq = invstd_scalar * invstd_scalar
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd_scalar * weight_scalar

    after_variance = where_active - centered * variance_term
    after_mean = after_variance - mean_term
    dense_f = after_mean * output_scale
    dense_bf = ct.astype(dense_f, ct.bfloat16)

    # Store the scatter of dense_bf back into (N, C, H, W).
    ct.scatter(dense_out_ptr, (n_safe, c_full, h_safe, w_safe), dense_bf, mask=active)

    # Reductions store. sum_where/sum_centered are 0D scalars.
    sum_where_1 = ct.full((1,), sum_where, dtype=ct.float32)
    scale_grad_1 = sum_centered * invstd_scalar  # (1,) tile
    ct.store(sum_out_ptr, index=(c,), tile=sum_where_1)
    ct.store(scale_grad_ptr, index=(c,), tile=scale_grad_1)


@ct.kernel
def _residual_add_kernel(
    residual0_ptr,  # bf16 (N, RESIDUAL0_C, H, W)
    residual1_ptr,  # bf16 (N, RESIDUAL1_C, H, W)
    dense_ptr,      # bf16 (N, C, H, W)
    add_out_ptr,    # bf16 (N, SLICE_C, H, W)
    BLOCK_R_C: ct.Constant[int],
):
    c = ct.bid(0)  # 0..SLICE_C-1
    rows = ct.arange(BLOCK_R_C, dtype=ct.int32)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    h_idx = spatial // W
    w_idx = spatial - h_idx * W

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

    # Two sequential bf16 adds with bf16 rounding boundaries. cuTile bf16
    # add is RTNE, so we cast to f32, add, and back to bf16 twice.
    r0_f = ct.astype(r0, ct.float32)
    r1_f = ct.astype(r1, ct.float32)
    sum1_bf = ct.astype(r0_f + r1_f, ct.bfloat16)

    sum2_bf = ct.astype(
        ct.astype(sum1_bf, ct.float32) + ct.astype(dense_v, ct.float32),
        ct.bfloat16,
    )

    ct.scatter(add_out_ptr, (n_safe, out_c, h_safe, w_safe), sum2_bf, mask=active)


@oracle_impl(hardware="B200", point="cbafcb75", BLOCK_R=4096, num_warps=8)
def oracle_forward(inputs, **_kwargs):
    residual0, residual1, mask_input, fill, source, centered_source, mean, invstd, weight = inputs
    device = source.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W),
        (SLICE_C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # Flatten mean to 1D from f32[1, C, 1, 1].
    mean_1d = mean.view(C)
    invstd_1d = invstd.view(C)
    weight_1d = weight.view(C)
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_tail_kernel,
        (mask_input, fill_1d, source, centered_source, mean_1d, invstd_1d, weight_1d,
         sum_out, scale_grad, dense_out, R, BLOCK_R, SCALE),
    )
    ct.launch(
        stream,
        (SLICE_C, 1, 1),
        _residual_add_kernel,
        (residual0, residual1, dense_out, add_out, BLOCK_R),
    )
    return sum_out, scale_grad, dense_out, add_out
