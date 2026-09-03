"""cuTile port of sum_sum_3fb78031bd33: DenseNet BN-backward tail + 15-way
residual slice add on channels 512:544.

Single fused kernel per channel: for each channel c, compute sum(where) and
mul8, grad_bf16, and for channels in [SLICE_START, C) also add the 15-way
residual slice sum + grad_bf16 into add_out.

Reference uses sequential bf16 add — cuTile bf16 defaults to rtne rounding
so plain a+b matches.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 544
SLICE_START = 512
SLICE_C = 32
SCALE = 0.0012755102040816326


@ct.kernel
def _bn_tail15_kernel(
    r0, r1, r2, r3, r4, r5, r6, r7,
    r8, r9, r10, r11, r12, r13, r14,
    mask_in,      # bf16 (N, C, H, W)
    fill,         # bf16 (1,)
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

    ct.store(sum_where_out, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(mul8_out, index=(c,),
             tile=ct.reshape(sum_mul * invstd_scalar, (1,)))

    ct.scatter(dense_out, (n, c_b, h_idx, w_idx), grad_bf, mask=active)

    in_slice = c >= SLICE_START
    if in_slice:
        v0 = ct.gather(r0, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v1 = ct.gather(r1, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v2 = ct.gather(r2, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v3 = ct.gather(r3, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v4 = ct.gather(r4, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v5 = ct.gather(r5, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v6 = ct.gather(r6, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v7 = ct.gather(r7, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v8 = ct.gather(r8, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v9 = ct.gather(r9, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v10 = ct.gather(r10, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v11 = ct.gather(r11, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v12 = ct.gather(r12, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v13 = ct.gather(r13, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)
        v14 = ct.gather(r14, (n, c_b, h_idx, w_idx), mask=active, padding_value=0)

        residual = v0 + v1
        residual = residual + v2
        residual = residual + v3
        residual = residual + v4
        residual = residual + v5
        residual = residual + v6
        residual = residual + v7
        residual = residual + v8
        residual = residual + v9
        residual = residual + v10
        residual = residual + v11
        residual = residual + v12
        residual = residual + v13
        residual = residual + v14
        added = residual + grad_bf
        slice_c_b = ct.full((BLOCK_R,), c - SLICE_START, dtype=ct.int32)
        ct.scatter(add_out, (n, slice_c_b, h_idx, w_idx), added, mask=active)


def _launch(inputs, *, H, W, BLOCK_R, SPLIT_SIDE, SIDE_WARPS):
    (
        r0, r1, r2, r3, r4, r5, r6, r7,
        r8, r9, r10, r11, r12, r13, r14,
        mask_input, fill, source, centered_source,
        mean, invstd, weight,
    ) = inputs
    device = source.device
    hw = H * W
    total_r = N * hw
    c_size = C  # 544
    slice_c = SLICE_C  # 32

    sum_where = torch.empty((c_size,), device=device, dtype=torch.float32)
    mul8 = torch.empty((c_size,), device=device, dtype=torch.float32)
    dense = torch.empty_strided(
        (N, c_size, H, W), (c_size * hw, hw, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, slice_c, H, W), (slice_c * hw, hw, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (c_size, 1, 1),
        _bn_tail15_kernel,
        (
            r0, r1, r2, r3, r4, r5, r6, r7,
            r8, r9, r10, r11, r12, r13, r14,
            mask_input, fill_1d, source, centered_source,
            mean.view(c_size), invstd.view(c_size), weight,
            sum_where, mul8, dense, add_out,
            H, W, hw, total_r, BLOCK_R,
        ),
    )
    return sum_where, mul8, dense, add_out


@oracle_impl(hardware="B200", point="38d67daf", H=14, W=14, BLOCK_R=1024, SPLIT_SIDE=True, SIDE_WARPS=8)
@oracle_impl(hardware="B200", point="dde2872d", H=7, W=7, BLOCK_R=256, SPLIT_SIDE=False, SIDE_WARPS=4)
def oracle_forward(inputs, *, H, W, BLOCK_R, SPLIT_SIDE, SIDE_WARPS):
    return _launch(inputs, H=H, W=W, BLOCK_R=BLOCK_R,
                   SPLIT_SIDE=SPLIT_SIDE, SIDE_WARPS=SIDE_WARPS)
