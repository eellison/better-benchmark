"""cuTile port of sum_sum_2928b7b46154: DenseNet BN-backward tail with
19 residual slice-adds.

Matches Triton's single-kernel plan: one program per channel c reduces
across R = N*HW rows to compute the two channel sums, emits the bf16
BN-backward dense output, and (for c in [SLICE_START, C)) also sums 19
sliced residuals with the sliced dense output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 416
H = 14
W = 14
HW = H * W
R = N * HW
SLICE_START = 384
SLICE_C = C - SLICE_START
SCALE = 0.0012755102040816326


@ct.kernel
def _bn_tail_kernel(
    r0_ptr,       # bf16 [N, 1024, H, W]
    r1_ptr,       # bf16 [N,  992, H, W]
    r2_ptr,       # bf16 [N,  960, H, W]
    r3_ptr,       # bf16 [N,  928, H, W]
    r4_ptr,       # bf16 [N,  896, H, W]
    r5_ptr,       # bf16 [N,  864, H, W]
    r6_ptr,       # bf16 [N,  832, H, W]
    r7_ptr,       # bf16 [N,  800, H, W]
    r8_ptr,       # bf16 [N,  768, H, W]
    r9_ptr,       # bf16 [N,  736, H, W]
    r10_ptr,      # bf16 [N,  704, H, W]
    r11_ptr,      # bf16 [N,  672, H, W]
    r12_ptr,      # bf16 [N,  640, H, W]
    r13_ptr,      # bf16 [N,  608, H, W]
    r14_ptr,      # bf16 [N,  576, H, W]
    r15_ptr,      # bf16 [N,  544, H, W]
    r16_ptr,      # bf16 [N,  512, H, W]
    r17_ptr,      # bf16 [N,  480, H, W]
    r18_ptr,      # bf16 [N,  448, H, W]
    mask_input_ptr,      # bf16 [N, C, H, W]
    fill_ptr,            # bf16 [1]
    source_ptr,          # bf16 [N, C, H, W]
    centered_source_ptr, # bf16 [N, C, H, W]
    mean_ptr,            # f32 [C]
    invstd_ptr,          # f32 [C]
    weight_ptr,          # f32 [C]
    sum_out_ptr,         # f32 [C]
    scale_grad_ptr,      # f32 [C]
    dense_out_ptr,       # bf16 [N, C, H, W]
    add_out_ptr,         # bf16 [N, SLICE_C, H, W]
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    h_idx = spatial // W
    w_idx = spatial - h_idx * W

    zero_i = ct.zeros((BLOCK_R,), dtype=ct.int32)
    n_safe = ct.where(active, n, zero_i)
    h_safe = ct.where(active, h_idx, zero_i)
    w_safe = ct.where(active, w_idx, zero_i)
    c_full = ct.full((BLOCK_R,), c, dtype=ct.int32)

    mask_value = ct.gather(mask_input_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    fill_1 = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bf = ct.full((BLOCK_R,), 0.0, dtype=ct.bfloat16) + ct.reshape(fill_1, (1,))
    source_value = ct.gather(source_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
    zero_bf = ct.zeros((BLOCK_R,), dtype=ct.bfloat16)
    where_bf16 = ct.where(mask_value <= zero_bf, fill_bf, source_value)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    where_f32 = ct.where(active, ct.astype(where_bf16, ct.float32), zero_f)

    centered_source_bf = ct.gather(
        centered_source_ptr, (n_safe, c_full, h_safe, w_safe), mask=active
    )
    centered_source = ct.astype(centered_source_bf, ct.float32)
    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean = ct.reshape(mean_1, (1,))
    centered = ct.where(active, centered_source - mean, zero_f)

    product = where_f32 * centered
    sum_where = ct.sum(where_f32)
    sum_centered = ct.sum(product)

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd = ct.reshape(invstd_1, (1,))
    weight = ct.reshape(weight_1, (1,))

    mean_term = sum_where * SCALE
    dot_scaled = sum_centered * SCALE
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight

    after_variance = where_f32 - centered * variance_term
    after_mean = after_variance - mean_term
    dense_f = after_mean * output_scale
    dense_bf16 = ct.astype(dense_f, ct.bfloat16)

    sum_where_1 = ct.full((1,), sum_where, dtype=ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=sum_where_1)
    scale_grad_val = sum_centered * invstd
    ct.store(scale_grad_ptr, index=(c,), tile=scale_grad_val)

    ct.scatter(
        dense_out_ptr, (n_safe, c_full, h_safe, w_safe), dense_bf16, mask=active,
    )

    # For c in [SLICE_START, C), compute 19-residual sum + dense_bf16 into add_out.
    if c >= SLICE_START:
        slice_c = c - SLICE_START
        slice_c_full = ct.full((BLOCK_R,), slice_c, dtype=ct.int32)
        v0 = ct.gather(r0_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v1 = ct.gather(r1_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v2 = ct.gather(r2_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v3 = ct.gather(r3_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v4 = ct.gather(r4_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v5 = ct.gather(r5_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v6 = ct.gather(r6_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v7 = ct.gather(r7_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v8 = ct.gather(r8_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v9 = ct.gather(r9_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v10 = ct.gather(r10_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v11 = ct.gather(r11_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v12 = ct.gather(r12_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v13 = ct.gather(r13_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v14 = ct.gather(r14_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v15 = ct.gather(r15_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v16 = ct.gather(r16_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v17 = ct.gather(r17_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)
        v18 = ct.gather(r18_ptr, (n_safe, c_full, h_safe, w_safe), mask=active)

        # Sequential bf16 adds (RTNE via f32-add-cast).
        def _bf16_add(a, b):
            return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32), ct.bfloat16)

        residual = _bf16_add(v0, v1)
        residual = _bf16_add(residual, v2)
        residual = _bf16_add(residual, v3)
        residual = _bf16_add(residual, v4)
        residual = _bf16_add(residual, v5)
        residual = _bf16_add(residual, v6)
        residual = _bf16_add(residual, v7)
        residual = _bf16_add(residual, v8)
        residual = _bf16_add(residual, v9)
        residual = _bf16_add(residual, v10)
        residual = _bf16_add(residual, v11)
        residual = _bf16_add(residual, v12)
        residual = _bf16_add(residual, v13)
        residual = _bf16_add(residual, v14)
        residual = _bf16_add(residual, v15)
        residual = _bf16_add(residual, v16)
        residual = _bf16_add(residual, v17)
        residual = _bf16_add(residual, v18)
        add_value = _bf16_add(residual, dense_bf16)

        ct.scatter(
            add_out_ptr, (n_safe, slice_c_full, h_safe, w_safe), add_value, mask=active,
        )


@oracle_impl(hardware="B200", point="f315b950", BLOCK_R=1024)
def oracle_forward(inputs, *, BLOCK_R: int):
    (
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,
        r10, r11, r12, r13, r14, r15, r16, r17, r18,
        mask_input, fill, source, centered_source, mean, invstd, weight,
    ) = inputs
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

    mean_1d = mean.view(C)
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_tail_kernel,
        (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,
         r10, r11, r12, r13, r14, r15, r16, r17, r18,
         mask_input, fill_1d, source, centered_source, mean_1d, invstd, weight,
         sum_out, scale_grad, dense_out, add_out, BLOCK_R),
    )

    return sum_out, scale_grad, dense_out, add_out
