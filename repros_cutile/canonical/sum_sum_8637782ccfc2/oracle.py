"""cuTile port of sum_sum_8637782ccfc2: DenseNet BN-backward tail + 4-way
residual slice add on channels 864:896.

Single fused kernel per channel: for each channel c, compute
sum_where, mul8 = sum(where * centered) * invstd, grad_bf16 (full plane),
and — if c >= SLICE_START — the bf16 sum of four residual slices added to
grad_bf16.

inline PTX rn.f32 arithmetic is default in cuTile so we drop it.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 896
SLICE_START = 864
SLICE_C = 32
SCALE = 0.0012755102040816326


@ct.kernel
def _bn_tail_kernel(
    r0,           # bf16 (N, 1024, H, W)
    r1,           # bf16 (N, 992, H, W)
    r2,           # bf16 (N, 960, H, W)
    r3,           # bf16 (N, 928, H, W)
    mask_in,      # bf16 (N, C, H, W)
    fill,         # bf16 () scalar
    where_rhs,    # bf16 (N, C, H, W)
    centered_src, # bf16 (N, C, H, W)
    mean,         # f32 (C,)
    invstd,       # f32 (C,)
    weight,       # f32 (C,)
    sum_where_out,   # f32 (C,)
    mul8_out,        # f32 (C,)
    dense_out,       # bf16 (N, C, H, W)
    add_out,         # bf16 (N, SLICE_C, H, W)
    H: ct.Constant[int],
    W: ct.Constant[int],
    HW: ct.Constant[int],
    TOTAL_R: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < TOTAL_R
    n = rows // HW
    spatial = rows - n * HW
    h_idx = spatial // W
    w_idx = spatial - h_idx * W

    c_b = ct.full((BLOCK_R,), c, dtype=ct.int32)

    mask_bf = ct.gather(mask_in, (n, c_b, h_idx, w_idx), mask=active,
                        padding_value=0)
    src = ct.gather(where_rhs, (n, c_b, h_idx, w_idx), mask=active,
                    padding_value=0)
    fill_scalar = ct.load(fill, index=(0,), shape=(1,))
    fill_scalar = ct.reshape(fill_scalar, ())
    where_bf = ct.where(mask_bf <= ct.astype(0.0, ct.bfloat16), fill_scalar, src)
    where_f = ct.where(active, ct.astype(where_bf, ct.float32), 0.0)

    csrc = ct.gather(centered_src, (n, c_b, h_idx, w_idx), mask=active,
                     padding_value=0)
    csrc_f = ct.astype(csrc, ct.float32)
    mean_scalar = ct.load(mean, index=(c,), shape=(1,))
    mean_scalar = ct.reshape(mean_scalar, ())
    centered = ct.where(active, csrc_f - mean_scalar, 0.0)

    prod = where_f * centered
    sum_where = ct.sum(where_f)
    sum_mul = ct.sum(prod)

    invstd_scalar = ct.load(invstd, index=(c,), shape=(1,))
    invstd_scalar = ct.reshape(invstd_scalar, ())
    weight_scalar = ct.load(weight, index=(c,), shape=(1,))
    weight_scalar = ct.reshape(weight_scalar, ())

    mean_term = sum_where * SCALE
    sum_mul_scaled = sum_mul * SCALE
    invstd_sq = invstd_scalar * invstd_scalar
    variance_term = sum_mul_scaled * invstd_sq
    out_weight = invstd_scalar * weight_scalar

    correction = where_f - centered * variance_term - mean_term
    grad = correction * out_weight
    grad_bf = ct.astype(grad, ct.bfloat16)

    # Store per-channel scalar outputs (only c=fixed-per-program).
    ct.store(sum_where_out, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(mul8_out, index=(c,),
             tile=ct.reshape(sum_mul * invstd_scalar, (1,)))

    # Store grad_bf into dense_out[n, c, h, w].
    ct.scatter(dense_out, (n, c_b, h_idx, w_idx), grad_bf, mask=active)

    # Residual slice add — only for channels in [SLICE_START, C).
    in_slice = c >= SLICE_START
    if in_slice:
        # slice_idx = c - SLICE_START
        v0 = ct.gather(r0, (n, c_b, h_idx, w_idx), mask=active,
                       padding_value=0)
        v1 = ct.gather(r1, (n, c_b, h_idx, w_idx), mask=active,
                       padding_value=0)
        v2 = ct.gather(r2, (n, c_b, h_idx, w_idx), mask=active,
                       padding_value=0)
        v3 = ct.gather(r3, (n, c_b, h_idx, w_idx), mask=active,
                       padding_value=0)
        residual = v0 + v1
        residual = residual + v2
        residual = residual + v3
        added = residual + grad_bf
        slice_c_b = ct.full((BLOCK_R,), c - SLICE_START, dtype=ct.int32)
        ct.scatter(add_out, (n, slice_c_b, h_idx, w_idx), added, mask=active)


def _launch(inputs, *, BLOCK_R: int):
    (
        residual0,
        residual1,
        residual2,
        residual3,
        mask,
        fill,
        where_rhs,
        centered_src,
        mean,
        invstd,
        weight,
    ) = inputs
    device = mask.device
    h = int(mask.shape[2])
    w = int(mask.shape[3])
    hw = h * w
    total_r = N * hw

    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    mul8 = torch.empty((C,), device=device, dtype=torch.float32)
    dense = torch.empty_strided(
        (N, C, h, w), (C * hw, hw, w, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, h, w), (SLICE_C * hw, hw, w, 1),
        device=device, dtype=torch.bfloat16,
    )
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_tail_kernel,
        (
            residual0, residual1, residual2, residual3,
            mask, fill_1d, where_rhs, centered_src,
            mean.view(C), invstd.view(C), weight,
            sum_where, mul8, dense, add_out,
            h, w, hw, total_r, BLOCK_R,
        ),
    )
    return sum_where, mul8, dense, add_out


@oracle_impl(hardware="B200", point="acf23a3b", BLOCK_R=1024)
@oracle_impl(hardware="B200", point="66dba80f", BLOCK_R=256)
def oracle_forward(inputs, *, BLOCK_R: int):
    return _launch(inputs, BLOCK_R=BLOCK_R)
