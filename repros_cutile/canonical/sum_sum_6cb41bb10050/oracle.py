"""cuTile port of sum_sum_6cb41bb10050: DenseNet BN-backward + 10-residual slice add.

Mirrors Triton's 3-kernel structure (both shape hashes use split_reduction=False):
- _reduce_vectors_kernel: per-channel `ct.sum` (sum_where, sum_centered).
- _full_epilogue_kernel: full-tensor BN-backward output (bf16 dense).
- _slice_add_kernel: 10-residual slice add + BN-backward slice.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 704
SLICE_START = 672
SLICE_C = 32
SCALE = 0.0012755102040816326


def _bf16_add_impl(a, b):
    return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32), ct.bfloat16)


@ct.kernel
def _reduce_vectors_kernel(
    mask_ptr,           # bf16 flat [N*C*HW]
    full_ptr,           # bf16 scalar
    where_rhs_ptr,      # bf16 flat [N*C*HW]
    centered_src_ptr,   # bf16 flat [N*C*HW]
    mean_ptr,           # f32 [C]
    invstd_ptr,         # f32 [C]
    sum_where_out_ptr,  # f32 [C]
    sum_mul_ptr,        # f32 [C]
    mul8_out_ptr,       # f32 [C]
    HW: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    k = ct.arange(BLOCK_R, dtype=ct.int32)
    total_spatial = N * HW
    active = k < total_spatial
    n = k // HW
    spatial = k - n * HW
    offsets = n * (C * HW) + c * HW + spatial

    mask_value_bf = ct.gather(mask_ptr, offsets)
    source_bf = ct.gather(where_rhs_ptr, offsets)
    full_scalar = ct.load(full_ptr, index=(0,), shape=(1,))
    mask_f = ct.astype(mask_value_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    full_f = ct.astype(full_scalar, ct.float32)
    where_bf16 = ct.where(mask_f <= zero_f,
                          ct.broadcast_to(full_f, (BLOCK_R,)),
                          ct.astype(source_bf, ct.float32))
    where_f32 = where_bf16

    centered_src_bf = ct.gather(centered_src_ptr, offsets)
    centered_src = ct.astype(centered_src_bf, ct.float32)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean_bc = ct.broadcast_to(mean_scalar, (BLOCK_R,))
    centered = centered_src - mean_bc
    product = where_f32 * centered

    where_active = ct.where(active, where_f32, 0.0)
    product_active = ct.where(active, product, 0.0)
    sum_where = ct.sum(where_active)
    sum_mul = ct.sum(product_active)

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, ())

    ct.store(sum_where_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(sum_mul_ptr, index=(c,), tile=ct.reshape(sum_mul, (1,)))
    ct.store(mul8_out_ptr, index=(c,), tile=ct.reshape(sum_mul * invstd, (1,)))


@ct.kernel
def _full_epilogue_kernel(
    mask_ptr,
    full_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_where_ptr,
    sum_mul_ptr,
    out_ptr,
    HW: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = offsets < TOTAL
    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    safe_off = ct.where(active, offsets, zero_i)
    c = (safe_off // HW) - ((safe_off // HW) // C) * C

    mask_value_bf = ct.gather(mask_ptr, safe_off)
    source_bf = ct.gather(where_rhs_ptr, safe_off)
    full_scalar = ct.load(full_ptr, index=(0,), shape=(1,))
    mask_f = ct.astype(mask_value_bf, ct.float32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    full_f = ct.astype(full_scalar, ct.float32)
    where_f32 = ct.where(mask_f <= zero_f,
                         ct.broadcast_to(full_f, (BLOCK,)),
                         ct.astype(source_bf, ct.float32))

    centered_src_bf = ct.gather(centered_src_ptr, safe_off)
    centered_src = ct.astype(centered_src_bf, ct.float32)
    mean = ct.gather(mean_ptr, c)
    centered = centered_src - mean

    sum_where = ct.gather(sum_where_ptr, c)
    sum_mul = ct.gather(sum_mul_ptr, c)
    invstd = ct.gather(invstd_ptr, c)
    weight = ct.gather(weight_ptr, c)

    mean_term = sum_where * SCALE
    sum_mul_scaled = sum_mul * SCALE
    invstd_sq = invstd * invstd
    variance_term = sum_mul_scaled * invstd_sq
    centered_scaled = centered * variance_term
    sub1 = where_f32 - centered_scaled
    sub2 = sub1 - mean_term
    out_weight = invstd * weight
    grad_f32 = sub2 * out_weight
    grad = ct.astype(grad_f32, ct.bfloat16)
    ct.scatter(out_ptr, offsets, grad, mask=active)


@ct.kernel
def _slice_add_kernel(
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr, r6_ptr, r7_ptr, r8_ptr, r9_ptr,
    mask_ptr,
    full_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_where_ptr,
    sum_mul_ptr,
    add_out_ptr,
    HW: ct.Constant[int],
    ADD_TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    out_offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = out_offsets < ADD_TOTAL
    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    safe_off = ct.where(active, out_offsets, zero_i)
    spatial = safe_off - (safe_off // HW) * HW
    slice_c_idx = (safe_off // HW) - ((safe_off // HW) // SLICE_C) * SLICE_C
    n = safe_off // (SLICE_C * HW)
    c = SLICE_START + slice_c_idx

    off0 = n * (1024 * HW) + c * HW + spatial
    off1 = n * (992 * HW) + c * HW + spatial
    off2 = n * (960 * HW) + c * HW + spatial
    off3 = n * (928 * HW) + c * HW + spatial
    off4 = n * (896 * HW) + c * HW + spatial
    off5 = n * (864 * HW) + c * HW + spatial
    off6 = n * (832 * HW) + c * HW + spatial
    off7 = n * (800 * HW) + c * HW + spatial
    off8 = n * (768 * HW) + c * HW + spatial
    off9 = n * (736 * HW) + c * HW + spatial
    input_offsets = n * (C * HW) + c * HW + spatial

    r0 = ct.gather(r0_ptr, off0)
    r1 = ct.gather(r1_ptr, off1)
    r2 = ct.gather(r2_ptr, off2)
    r3 = ct.gather(r3_ptr, off3)
    r4 = ct.gather(r4_ptr, off4)
    r5 = ct.gather(r5_ptr, off5)
    r6 = ct.gather(r6_ptr, off6)
    r7 = ct.gather(r7_ptr, off7)
    r8 = ct.gather(r8_ptr, off8)
    r9 = ct.gather(r9_ptr, off9)

    residual = _bf16_add_impl(r0, r1)
    residual = _bf16_add_impl(residual, r2)
    residual = _bf16_add_impl(residual, r3)
    residual = _bf16_add_impl(residual, r4)
    residual = _bf16_add_impl(residual, r5)
    residual = _bf16_add_impl(residual, r6)
    residual = _bf16_add_impl(residual, r7)
    residual = _bf16_add_impl(residual, r8)
    residual = _bf16_add_impl(residual, r9)

    mask_value_bf = ct.gather(mask_ptr, input_offsets)
    source_bf = ct.gather(where_rhs_ptr, input_offsets)
    full_scalar = ct.load(full_ptr, index=(0,), shape=(1,))
    mask_f = ct.astype(mask_value_bf, ct.float32)
    zero_f = ct.zeros((BLOCK,), dtype=ct.float32)
    full_f = ct.astype(full_scalar, ct.float32)
    where_f32 = ct.where(mask_f <= zero_f,
                         ct.broadcast_to(full_f, (BLOCK,)),
                         ct.astype(source_bf, ct.float32))

    centered_src_bf = ct.gather(centered_src_ptr, input_offsets)
    centered_src = ct.astype(centered_src_bf, ct.float32)
    mean = ct.gather(mean_ptr, c)
    centered = centered_src - mean

    sum_where = ct.gather(sum_where_ptr, c)
    sum_mul = ct.gather(sum_mul_ptr, c)
    invstd = ct.gather(invstd_ptr, c)
    weight = ct.gather(weight_ptr, c)

    mean_term = sum_where * SCALE
    sum_mul_scaled = sum_mul * SCALE
    invstd_sq = invstd * invstd
    variance_term = sum_mul_scaled * invstd_sq
    centered_scaled = centered * variance_term
    sub1 = where_f32 - centered_scaled
    sub2 = sub1 - mean_term
    out_weight = invstd * weight
    grad = ct.astype(sub2 * out_weight, ct.bfloat16)
    added = _bf16_add_impl(residual, grad)
    ct.scatter(add_out_ptr, out_offsets, added, mask=active)


@oracle_impl(hardware="B200", point="4db20f60", EPILOGUE_BLOCK=256, ADD_BLOCK=256)
@oracle_impl(hardware="B200", point="24d0862c", EPILOGUE_BLOCK=256, ADD_BLOCK=256)
def oracle_forward(inputs, *, EPILOGUE_BLOCK: int, ADD_BLOCK: int):
    (
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,
        mask_input, fill, where_rhs, centered_src, mean, invstd, weight,
    ) = inputs
    device = mask_input.device
    _, _, h, w = mask_input.shape
    hw = int(h) * int(w)
    total_spatial = N * hw
    total = N * C * hw
    add_total = N * SLICE_C * hw
    # Round BLOCK_R up to next power of 2 for the tile size
    def _next_p2(v: int) -> int:
        r = 1
        while r < v:
            r <<= 1
        return r
    BLOCK_R = _next_p2(total_spatial)

    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    sum_mul = torch.empty((C,), device=device, dtype=torch.float32)
    mul8 = torch.empty((C,), device=device, dtype=torch.float32)
    grad = torch.empty_strided(
        (N, C, h, w), (C * hw, hw, w, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, h, w), (SLICE_C * hw, hw, w, 1),
        device=device, dtype=torch.bfloat16,
    )

    def _flat(t):
        return t.contiguous().view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _reduce_vectors_kernel,
        (_flat(mask_input), fill.view(1), _flat(where_rhs), _flat(centered_src),
         mean.view(C).contiguous(), invstd.view(C).contiguous(),
         sum_where, sum_mul, mul8, hw, BLOCK_R),
    )
    ct.launch(
        stream,
        ((total + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK, 1, 1),
        _full_epilogue_kernel,
        (_flat(mask_input), fill.view(1), _flat(where_rhs), _flat(centered_src),
         mean.view(C).contiguous(), invstd.view(C).contiguous(),
         weight.view(C).contiguous(),
         sum_where, sum_mul, grad.view(-1),
         hw, total, EPILOGUE_BLOCK),
    )
    ct.launch(
        stream,
        ((add_total + ADD_BLOCK - 1) // ADD_BLOCK, 1, 1),
        _slice_add_kernel,
        (_flat(r0), _flat(r1), _flat(r2), _flat(r3), _flat(r4),
         _flat(r5), _flat(r6), _flat(r7), _flat(r8), _flat(r9),
         _flat(mask_input), fill.view(1), _flat(where_rhs), _flat(centered_src),
         mean.view(C).contiguous(), invstd.view(C).contiguous(),
         weight.view(C).contiguous(),
         sum_where, sum_mul, add_out.view(-1),
         hw, add_total, ADD_BLOCK),
    )
    return sum_where, mul8, grad, add_out
