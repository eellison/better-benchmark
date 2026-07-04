"""cuTile port of sum_sum_6435ae06bfdf: DenseNet-121 BN-backward + 20-residual slice-add.

Mirrors Triton's two-kernel structure: `_dense_kernel` does per-channel
reductions with `ct.sum(..., axis=0)` and writes BN-backward dense output;
`_slice_add_kernel` loads back the dense output and adds 20 residual slices.

BLOCK_R=1024 matches Triton; N*HW=784 fits within a 1024 tile with mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 384
H = 14
W = 14
HW = H * W  # 196
R = N * HW  # 784
SLICE_START = 352
SLICE_C = C - SLICE_START  # 32
SCALE = 0.0012755102040816326


@ct.kernel
def _dense_kernel(
    mask_input_ptr,        # bf16 [N,C,H,W] flat
    fill_ptr,              # bf16 scalar
    source_ptr,            # bf16 [N,C,H,W] flat
    centered_source_ptr,   # bf16 [N,C,H,W] flat
    mean_ptr,              # f32 [C]
    invstd_ptr,            # f32 [C]
    weight_ptr,            # f32 [C]
    sum_out_ptr,           # f32 [C]
    scale_grad_ptr,        # f32 [C]
    dense_out_ptr,         # bf16 [N,C,H,W] flat
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

    centered_source_bf = ct.gather(centered_source_ptr, dense_offsets)
    centered_source = ct.astype(centered_source_bf, ct.float32)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean = ct.broadcast_to(mean_scalar, (BLOCK_R,))
    centered = ct.where(active, centered_source - mean, 0.0)

    product = where_f * centered
    sum_where = ct.sum(where_f)
    sum_centered = ct.sum(product)

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

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    scale_grad_val = sum_centered * ct.reshape(invstd, ())
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(scale_grad_val, (1,)))
    ct.scatter(dense_out_ptr, dense_offsets, dense_bf, mask=active)


@ct.kernel
def _slice_add_kernel(
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr, r6_ptr, r7_ptr, r8_ptr,
    r9_ptr, r10_ptr, r11_ptr, r12_ptr, r13_ptr, r14_ptr, r15_ptr, r16_ptr,
    r17_ptr, r18_ptr, r19_ptr,
    dense_out_ptr,
    add_out_ptr,
    BLOCK_R: ct.Constant[int],
):
    slice_c_val = ct.bid(0)
    c = SLICE_START + slice_c_val
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW

    dense_offsets = n * (C * HW) + c * HW + spatial
    add_offsets = n * (SLICE_C * HW) + slice_c_val * HW + spatial
    dense_bf = ct.gather(dense_out_ptr, dense_offsets)

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
    v16 = ct.gather(r16_ptr, n * (512 * HW) + c * HW + spatial)
    v17 = ct.gather(r17_ptr, n * (480 * HW) + c * HW + spatial)
    v18 = ct.gather(r18_ptr, n * (448 * HW) + c * HW + spatial)
    v19 = ct.gather(r19_ptr, n * (416 * HW) + c * HW + spatial)

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
    residual = _bf16_add(residual, v16)
    residual = _bf16_add(residual, v17)
    residual = _bf16_add(residual, v18)
    residual = _bf16_add(residual, v19)
    add_value = _bf16_add(residual, dense_bf)
    ct.scatter(add_out_ptr, add_offsets, add_value, mask=active)


@oracle_impl(hardware="B200", point="4d29b7b6", BLOCK_R=1024)
def oracle_forward(inputs, *, BLOCK_R: int):
    (
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,
        r10, r11, r12, r13, r14, r15, r16, r17, r18, r19,
        mask_input, fill, source, centered_source, mean, invstd, weight,
    ) = inputs
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
        stream, (C, 1, 1), _dense_kernel,
        (_flat(mask_input), fill.view(1), _flat(source),
         _flat(centered_source),
         mean.view(C).contiguous(), invstd.view(C).contiguous(),
         weight.view(C).contiguous(),
         sum_out, scale_grad,
         dense_out.view(-1),
         BLOCK_R),
    )
    ct.launch(
        stream, (SLICE_C, 1, 1), _slice_add_kernel,
        (_flat(r0), _flat(r1), _flat(r2), _flat(r3), _flat(r4),
         _flat(r5), _flat(r6), _flat(r7), _flat(r8), _flat(r9),
         _flat(r10), _flat(r11), _flat(r12), _flat(r13), _flat(r14),
         _flat(r15), _flat(r16), _flat(r17), _flat(r18), _flat(r19),
         dense_out.view(-1), add_out.view(-1),
         BLOCK_R),
    )
    return sum_out, scale_grad, dense_out, add_out
