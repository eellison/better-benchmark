"""cuTile port of sum_sum_b2f43ab7b175: DenseNet BN-backward tail with residual add.

Fair port: one cuTile kernel mirroring Triton's `_densenet_bn_tail_kernel`.
Grid is (C,). Each program reduces along the (N*H*W) axis with `ct.sum`
in-kernel, then writes the sum/scale-grad vectors, the full BN-backward
output, and (for c in [544, 576)) the 14-input residual-add slice output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 576
N = 4
SLICE_C = 32


@ct.kernel
def _densenet_bn_tail_kernel(
    residual0_ptr,   # bf16 flat [N * 1024 * HW]
    residual1_ptr,   # bf16 flat [N * 992 * HW]
    residual2_ptr,
    residual3_ptr,
    residual4_ptr,
    residual5_ptr,
    residual6_ptr,
    residual7_ptr,
    residual8_ptr,
    residual9_ptr,
    residual10_ptr,
    residual11_ptr,
    residual12_ptr,
    residual13_ptr,   # bf16 flat [N * 608 * HW]
    mask_ptr,         # bf16 flat [N * 576 * HW]
    fill_ptr,         # bf16 [1]
    where_rhs_ptr,    # bf16 flat [N * 576 * HW]
    centered_src_ptr, # bf16 flat [N * 576 * HW]
    mean_ptr,         # f32 [576]
    invstd_ptr,       # f32 [576]
    weight_ptr,       # f32 [576]
    sum_out_ptr,      # f32 [576]
    scaled_dot_out_ptr, # f32 [576]
    full_out_ptr,     # bf16 flat [N * 576 * HW]
    slice_out_ptr,    # bf16 flat [N * 32 * HW]
    HW: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    INV_KTOTAL: ct.Constant[float],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    r_idx = ct.arange(BLOCK_K, dtype=ct.int32)
    active_1d = r_idx < K_TOTAL
    n = r_idx // HW
    hw = r_idx - n * HW
    base = n * (576 * HW) + c * HW + hw

    # Load producer inputs.
    mask_bf = ct.gather(mask_ptr, base)
    rhs_bf = ct.gather(where_rhs_ptr, base)
    centered_bf = ct.gather(centered_src_ptr, base)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill, ct.float32)

    mask_f = ct.astype(mask_bf, ct.float32)
    rhs_f = ct.astype(rhs_bf, ct.float32)
    centered_src = ct.astype(centered_bf, ct.float32)

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    mean_bc = ct.full((BLOCK_K,), 0.0, dtype=ct.float32) + mean
    invstd_bc = ct.full((BLOCK_K,), 0.0, dtype=ct.float32) + invstd
    weight_bc = ct.full((BLOCK_K,), 0.0, dtype=ct.float32) + weight
    fill_bc = ct.full((BLOCK_K,), 0.0, dtype=ct.float32) + fill_f

    zero_bc = ct.zeros((BLOCK_K,), dtype=ct.float32)
    source = ct.where(mask_f <= zero_bc, fill_bc, rhs_f)
    centered = centered_src - mean_bc

    source_for_sum = ct.where(active_1d, source, zero_bc)
    centered_for_sum = ct.where(active_1d, centered, zero_bc)
    sum_value = ct.sum(source_for_sum, axis=0)
    dot_value = ct.sum(source_for_sum * centered_for_sum, axis=0)

    mean_term = sum_value * INV_KTOTAL
    dot_scaled = dot_value * INV_KTOTAL
    coeff = dot_scaled * invstd * invstd
    output_scale = invstd * weight

    mean_term_bc = ct.full((BLOCK_K,), 0.0, dtype=ct.float32) + mean_term
    coeff_bc = ct.full((BLOCK_K,), 0.0, dtype=ct.float32) + coeff
    output_scale_bc = ct.full((BLOCK_K,), 0.0, dtype=ct.float32) + output_scale

    out_f32 = (source - centered * coeff_bc - mean_term_bc) * output_scale_bc
    out_bf = ct.astype(out_f32, ct.bfloat16)

    # Scalar per-channel writes: use `ct.store` on a 1-elem shape.
    ct.store(sum_out_ptr, index=(c,), tile=sum_value)
    ct.store(scaled_dot_out_ptr, index=(c,), tile=dot_value * invstd)
    # Full-tensor scatter.
    ct.scatter(full_out_ptr, base, out_bf, mask=active_1d)

    # Residual add slice (only for c in [544, 576)).
    is_slice = c >= 544
    if is_slice:
        # 14-input residual add.
        offset0 = n * (1024 * HW) + c * HW + hw
        offset1 = n * (992 * HW) + c * HW + hw
        offset2 = n * (960 * HW) + c * HW + hw
        offset3 = n * (928 * HW) + c * HW + hw
        offset4 = n * (896 * HW) + c * HW + hw
        offset5 = n * (864 * HW) + c * HW + hw
        offset6 = n * (832 * HW) + c * HW + hw
        offset7 = n * (800 * HW) + c * HW + hw
        offset8 = n * (768 * HW) + c * HW + hw
        offset9 = n * (736 * HW) + c * HW + hw
        offset10 = n * (704 * HW) + c * HW + hw
        offset11 = n * (672 * HW) + c * HW + hw
        offset12 = n * (640 * HW) + c * HW + hw
        offset13 = n * (608 * HW) + c * HW + hw
        r0 = ct.astype(ct.gather(residual0_ptr, offset0), ct.float32)
        r1 = ct.astype(ct.gather(residual1_ptr, offset1), ct.float32)
        r2 = ct.astype(ct.gather(residual2_ptr, offset2), ct.float32)
        r3 = ct.astype(ct.gather(residual3_ptr, offset3), ct.float32)
        r4 = ct.astype(ct.gather(residual4_ptr, offset4), ct.float32)
        r5 = ct.astype(ct.gather(residual5_ptr, offset5), ct.float32)
        r6 = ct.astype(ct.gather(residual6_ptr, offset6), ct.float32)
        r7 = ct.astype(ct.gather(residual7_ptr, offset7), ct.float32)
        r8 = ct.astype(ct.gather(residual8_ptr, offset8), ct.float32)
        r9 = ct.astype(ct.gather(residual9_ptr, offset9), ct.float32)
        r10 = ct.astype(ct.gather(residual10_ptr, offset10), ct.float32)
        r11 = ct.astype(ct.gather(residual11_ptr, offset11), ct.float32)
        r12 = ct.astype(ct.gather(residual12_ptr, offset12), ct.float32)
        r13 = ct.astype(ct.gather(residual13_ptr, offset13), ct.float32)

        # Match Triton's rounded chain of adds.
        acc = ct.astype(ct.astype(r0 + r1, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r2, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r3, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r4, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r5, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r6, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r7, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r8, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r9, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r10, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r11, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r12, ct.bfloat16), ct.float32)
        acc = ct.astype(ct.astype(acc + r13, ct.bfloat16), ct.float32)

        slice_c = c - 544
        slice_off = n * (32 * HW) + slice_c * HW + hw
        slice_val = ct.astype(acc + ct.astype(out_bf, ct.float32), ct.bfloat16)
        slice_active = active_1d
        ct.scatter(slice_out_ptr, slice_off, slice_val, mask=slice_active)


@oracle_impl(hardware="B200", point="5f5b775d", HW=196, K_TOTAL=784, BLOCK_K=1024)
@oracle_impl(hardware="B200", point="d4009f05", HW=49, K_TOTAL=196, BLOCK_K=256)
def oracle_forward(inputs, *, HW: int, K_TOTAL: int, BLOCK_K: int):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1,
        arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1,
    ) = inputs
    device = arg14_1.device

    # The captured repro hard-codes reduce_scale = 1/784 (H=W=14 case); both
    # shape points reuse the same forward() so we match the constant.
    INV_KTOTAL = 0.0012755102040816326

    def _flat(t):
        return t.reshape(-1)

    residuals_flat = [_flat(t) for t in (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1,
    )]
    mask_flat = _flat(arg14_1)
    where_rhs_flat = _flat(arg16_1)
    centered_src_flat = _flat(arg17_1)
    mean_flat = arg18_1.view(C)
    invstd_flat = arg19_1
    weight_flat = arg20_1
    fill_1d = arg15_1.reshape(1).contiguous()

    h = int(arg14_1.shape[2])
    w = int(arg14_1.shape[3])

    full_out = torch.empty_strided(
        (N, C, h, w), (C * HW, HW, w, 1), device=device, dtype=torch.bfloat16,
    )
    slice_out = torch.empty_strided(
        (N, SLICE_C, h, w), (SLICE_C * HW, HW, w, 1), device=device, dtype=torch.bfloat16,
    )
    full_flat = _flat(full_out)
    slice_flat = _flat(slice_out)

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _densenet_bn_tail_kernel,
        (*residuals_flat, mask_flat, fill_1d, where_rhs_flat, centered_src_flat,
         mean_flat, invstd_flat, weight_flat,
         sum_out, scaled_dot_out, full_flat, slice_flat,
         HW, K_TOTAL, INV_KTOTAL, BLOCK_K),
    )
    return sum_out, scaled_dot_out, full_out, slice_out
