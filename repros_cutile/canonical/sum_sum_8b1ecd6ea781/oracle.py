"""cuTile port of sum_sum_8b1ecd6ea781: DenseNet BN-backward tail + residual add.

Mirrors Triton's single-kernel structure: for each channel c, reduce over
`R = N * HW` producing `sum_where` and `sum(where * centered)` with
`ct.sum(...)`. Uses these in-kernel to write BN gradient bf16 output; for
c in [800, 832) also gathers the 6 residual slices and writes a bf16
per-slice residual add.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 832
N = 4
SLICE_START = 800
SLICE_C = 32
SCALE = 0.0012755102040816326


@ct.kernel
def _densenet_bn_tail_kernel(
    a0_ptr, a1_ptr, a2_ptr, a3_ptr, a4_ptr, a5_ptr,  # bf16 residuals
    mask_ptr,           # bf16 [N,832,H,W]
    fill_ptr,           # bf16 scalar
    where_rhs_ptr,      # bf16 [N,832,H,W]
    centered_src_ptr,   # bf16 [N,832,H,W]
    mean_ptr,           # f32 [832]
    invstd_ptr,         # f32 [832]
    weight_ptr,         # f32 [832]
    sum_out_ptr,        # f32 [832]
    scaled_dot_out_ptr, # f32 [832]
    full_out_ptr,       # bf16 [N,832,H,W]
    slice_out_ptr,      # bf16 [N,32,H,W]
    HW: ct.Constant[int],
    K_TOTAL: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    r = ct.arange(BLOCK_K, dtype=ct.int32)
    active = r < K_TOTAL
    n = r // HW
    hw = r - n * HW
    base = n * (832 * HW) + c * HW + hw

    mask_bf = ct.gather(mask_ptr, base)
    mask_value = ct.astype(mask_bf, ct.float32)
    rhs_bf = ct.gather(where_rhs_ptr, base)
    rhs = ct.astype(rhs_bf, ct.float32)
    fill_bf = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_val = ct.astype(fill_bf, ct.float32)
    fill_bcast = ct.broadcast_to(fill_val, (BLOCK_K,))
    source = ct.where(mask_value <= 0.0, fill_bcast, rhs)

    centered_src_bf = ct.gather(centered_src_ptr, base)
    centered_src = ct.astype(centered_src_bf, ct.float32)
    mean_bf = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean_scalar = ct.reshape(mean_bf, ())
    mean = ct.broadcast_to(mean_scalar, (BLOCK_K,))
    centered = centered_src - mean

    source_masked = ct.where(active, source, 0.0)
    centered_masked = ct.where(active, centered, 0.0)
    sum_value = ct.sum(source_masked)  # scalar
    dot_value = ct.sum(source_masked * centered_masked)  # scalar

    invstd_bf = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    invstd_scalar = ct.reshape(invstd_bf, ())
    weight_bf = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    weight_scalar = ct.reshape(weight_bf, ())
    mean_term = sum_value * SCALE
    invstd_sq = invstd_scalar * invstd_scalar
    coeff = dot_value * SCALE * invstd_sq
    output_scale = invstd_scalar * weight_scalar

    correction = centered * ct.broadcast_to(coeff, (BLOCK_K,))
    corrected = source - correction - ct.broadcast_to(mean_term, (BLOCK_K,))
    out_f32 = corrected * ct.broadcast_to(output_scale, (BLOCK_K,))
    out_bf16 = ct.astype(out_f32, ct.bfloat16)

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    scale_grad_val = dot_value * invstd_scalar
    ct.store(scaled_dot_out_ptr, index=(c,),
             tile=ct.reshape(scale_grad_val, (1,)))
    # Store dense bf16 output via scatter with mask.
    ct.scatter(full_out_ptr, base, out_bf16, mask=active)

    # Slice-add branch for c in [800, 832).
    if c >= SLICE_START:
        slice_c_val = c - SLICE_START
        slice_offsets = n * (SLICE_C * HW) + slice_c_val * HW + hw

        r0 = ct.gather(a0_ptr, n * (1024 * HW) + c * HW + hw)
        r1 = ct.gather(a1_ptr, n * (992 * HW) + c * HW + hw)
        r2 = ct.gather(a2_ptr, n * (960 * HW) + c * HW + hw)
        r3 = ct.gather(a3_ptr, n * (928 * HW) + c * HW + hw)
        r4 = ct.gather(a4_ptr, n * (896 * HW) + c * HW + hw)
        r5 = ct.gather(a5_ptr, n * (864 * HW) + c * HW + hw)

        def _bf16_add(a, b):
            return ct.astype(
                ct.astype(a, ct.float32) + ct.astype(b, ct.float32),
                ct.bfloat16,
            )

        residual = _bf16_add(r0, r1)
        residual = _bf16_add(residual, r2)
        residual = _bf16_add(residual, r3)
        residual = _bf16_add(residual, r4)
        residual = _bf16_add(residual, r5)
        slice_add = _bf16_add(residual, out_bf16)
        ct.scatter(slice_out_ptr, slice_offsets, slice_add, mask=active)


@oracle_impl(hardware="B200", point="100b69a7", HW=196, K_TOTAL=784, BLOCK_K=1024)
@oracle_impl(hardware="B200", point="3982ddad", HW=49, K_TOTAL=196, BLOCK_K=256)
def oracle_forward(inputs, *, HW, K_TOTAL, BLOCK_K):
    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12) = inputs
    device = a6.device
    h = int(a6.shape[2])
    w = int(a6.shape[3])

    # a0..a5: bf16[N, {1024,992,960,928,896,864}, H, W] contiguous
    # a6: bf16[N, 832, H, W] (mask)  a7: bf16 scalar (fill)
    # a8: bf16[N, 832, H, W] (where rhs)  a9: bf16[N, 832, H, W] (centered_src)
    # a10: f32[1,832,1,1] (mean); a11: f32[832] (invstd); a12: f32[832] (weight)
    def _flat(t):
        return t.contiguous().view(-1)

    a0f = _flat(a0)
    a1f = _flat(a1)
    a2f = _flat(a2)
    a3f = _flat(a3)
    a4f = _flat(a4)
    a5f = _flat(a5)
    a6f = _flat(a6)
    a7f = a7.contiguous().view(1)
    a8f = _flat(a8)
    a9f = _flat(a9)
    a10f = a10.contiguous().view(C)
    a11f = a11.contiguous().view(C)
    a12f = a12.contiguous().view(C)

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    full_out = torch.empty_strided(
        (N, C, h, w),
        (C * HW, HW, w, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    slice_out = torch.empty_strided(
        (N, SLICE_C, h, w),
        (SLICE_C * HW, HW, w, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _densenet_bn_tail_kernel,
        (a0f, a1f, a2f, a3f, a4f, a5f,
         a6f, a7f, a8f, a9f, a10f, a11f, a12f,
         sum_out, scaled_dot_out,
         full_out.view(-1), slice_out.view(-1),
         HW, K_TOTAL, BLOCK_K),
    )
    return sum_out, scaled_dot_out, full_out, slice_out
