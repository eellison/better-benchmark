"""cuTile port of sum_sum_94f014e4cfa6: DenseNet BN backward tail (512 ch, 14x14).

Mirrors Triton's single `_tail_kernel`: per-channel program computes
sum(where) and sum(where * centered) via in-kernel `ct.sum(..., axis=0)`,
uses those finalized scalars in the BN-backward epilogue, then does the
16-way residual slice-add. Same structural pattern as sum_sum_91f994494908
with 16 residual sources instead of 17.

BLOCK_R=1024 matches Triton's setting; N*HW=784 fits within a 1024 tile with mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 512
H = 14
W = 14
HW = H * W  # 196
R = N * HW  # 784
SLICE_START = 480
SLICE_C = C - SLICE_START  # 32
SCALE = 0.0012755102040816326


@ct.kernel
def _bn_tail_kernel(
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr, r6_ptr, r7_ptr,
    r8_ptr, r9_ptr, r10_ptr, r11_ptr, r12_ptr, r13_ptr, r14_ptr, r15_ptr,
    mask_input_ptr,        # bf16 [N,C,H,W] contig
    fill_ptr,              # bf16 scalar
    source_ptr,            # bf16 [N,C,H,W] contig
    centered_source_ptr,   # f32 [N,C,H,W] contig
    mean_ptr,              # f32 [C]
    invstd_ptr,            # f32 [C]
    weight_ptr,            # f32 [C]
    sum_out_ptr,           # f32 [C]
    scale_grad_ptr,        # f32 [C]
    dense_out_ptr,         # bf16 [N,C,H,W] contig
    add_out_ptr,           # bf16 [N, SLICE_C, H, W] contig
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    dense_offsets = n * (C * HW) + c * HW + spatial

    mask_bf = ct.gather(mask_input_ptr, dense_offsets)
    mask_value = ct.astype(mask_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_value = ct.astype(fill_scalar, ct.float32)
    source_bf = ct.gather(source_ptr, dense_offsets)
    source_value = ct.astype(source_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    where_f = ct.where(mask_value <= zero_f,
                       ct.broadcast_to(fill_value, (BLOCK_R,)), source_value)
    where_f = ct.where(active, where_f, 0.0)

    centered_source = ct.gather(centered_source_ptr, dense_offsets)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean = ct.broadcast_to(mean_scalar, (BLOCK_R,))
    centered = ct.where(active, centered_source - mean, 0.0)

    product = where_f * centered
    sum_where = ct.sum(where_f)  # scalar
    sum_centered = ct.sum(product)  # scalar

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_scalar = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, (1,))
    weight = ct.reshape(weight_scalar, (1,))
    mean_term = sum_where * SCALE
    dot_scaled = sum_centered * SCALE
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight

    variance_term_bc = ct.broadcast_to(variance_term, (BLOCK_R,))
    mean_term_bc = ct.broadcast_to(mean_term, (BLOCK_R,))
    output_scale_bc = ct.broadcast_to(output_scale, (BLOCK_R,))
    after_variance = where_f - centered * variance_term_bc
    after_mean = after_variance - mean_term_bc
    dense = after_mean * output_scale_bc
    dense_bf = ct.astype(dense, ct.bfloat16)

    # Store per-channel scalars once.
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    scale_grad_val = sum_centered * ct.reshape(invstd, ())
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(scale_grad_val, (1,)))
    # Store dense output via scatter with mask.
    ct.scatter(dense_out_ptr, dense_offsets, dense_bf, mask=active)

    # Slice add for c in [SLICE_START, C).
    in_slice = c >= SLICE_START
    if in_slice:
        slice_c_val = c - SLICE_START
        add_offsets = n * (SLICE_C * HW) + slice_c_val * HW + spatial

        v0 = ct.gather(r0_ptr, n * (1024 * HW) + c * HW + spatial)
        v1 = ct.gather(r1_ptr, n * (992 * HW) + c * HW + spatial)
        v2 = ct.gather(r2_ptr, n * (960 * HW) + c * HW + spatial)
        v3 = ct.gather(r3_ptr, n * (928 * HW) + c * HW + spatial)
        v4 = ct.gather(r4_ptr, n * (896 * HW) + c * HW + spatial)
        v5 = ct.gather(r5_ptr, n * (864 * HW) + c * HW + spatial)
        v6 = ct.gather(r6_ptr, n * (832 * HW) + c * HW + spatial)
        v7 = ct.gather(r7_ptr, n * (800 * HW) + c * HW + spatial)
        v8 = ct.gather(r8_ptr, n * (768 * HW) + c * HW + spatial)
        v9 = ct.gather(r9_ptr, n * (736 * HW) + c * HW + spatial)
        v10 = ct.gather(r10_ptr, n * (704 * HW) + c * HW + spatial)
        v11 = ct.gather(r11_ptr, n * (672 * HW) + c * HW + spatial)
        v12 = ct.gather(r12_ptr, n * (640 * HW) + c * HW + spatial)
        v13 = ct.gather(r13_ptr, n * (608 * HW) + c * HW + spatial)
        v14 = ct.gather(r14_ptr, n * (576 * HW) + c * HW + spatial)
        v15 = ct.gather(r15_ptr, n * (544 * HW) + c * HW + spatial)

        def _bf16_add(a, b):
            return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32),
                             ct.bfloat16)

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
        add_value = _bf16_add(residual, dense_bf)
        ct.scatter(add_out_ptr, add_offsets, add_value, mask=active)


@oracle_impl(hardware="B200", point="f543b665", BLOCK_R=1024)
def oracle_forward(inputs, *, BLOCK_R: int):
    (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
     mask_input, fill, source, centered_source, mean, invstd, weight) = inputs
    device = source.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W), (SLICE_C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    def _flat(t):
        size = 1
        for s in t.shape:
            size *= s
        return t.contiguous().view(size)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _bn_tail_kernel,
        (_flat(r0), _flat(r1), _flat(r2), _flat(r3), _flat(r4),
         _flat(r5), _flat(r6), _flat(r7), _flat(r8), _flat(r9),
         _flat(r10), _flat(r11), _flat(r12), _flat(r13), _flat(r14),
         _flat(r15),
         _flat(mask_input), fill.view(1), _flat(source),
         _flat(centered_source),
         mean.view(C).contiguous(), invstd.view(C).contiguous(),
         weight.view(C).contiguous(),
         sum_out, scale_grad,
         dense_out.view(-1), add_out.view(-1),
         BLOCK_R),
    )
    return sum_out, scale_grad, dense_out, add_out
