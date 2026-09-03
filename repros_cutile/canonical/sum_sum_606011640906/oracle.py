"""cuTile port of sum_sum_606011640906: DenseNet BN-backward + avg_pool2d_backward.

Mirrors the Triton `_densenet_pool_kernel` which computes per-channel sum(where)
and sum(where * centered) via `tl.sum(..., axis=0)` in-kernel, then uses those
finalized scalars in the BN-backward epilogue, adds twelve bf16 residual slices,
and expands the parent into a 2x2 nearest-neighbor pool output. cuTile matches
this with `ct.sum(..., axis=0)` and per-program-c work at grid (C,).

BLOCK_R=4096 matches Triton; N*HW=3136 fits within a 4096 tile with mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 128
H = 28
W = 28
HW = H * W  # 784
OUT_H = 56
OUT_W = 56
OUT_HW = OUT_H * OUT_W  # 3136
R = N * HW  # 3136
SCALE = 0.00031887755102040814


@ct.kernel
def _densenet_pool_kernel(
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr,
    r6_ptr, r7_ptr, r8_ptr, r9_ptr, r10_ptr, r11_ptr,
    mask_ptr,             # bf16 [N,C,HW] flat
    fill_ptr,             # bf16 scalar (view(1))
    source_ptr,           # bf16 [N,C,HW] flat
    centered_source_ptr,  # bf16 [N,C,HW] flat
    mean_ptr,             # f32 [C]
    invstd_ptr,           # f32 [C]
    weight_ptr,           # f32 [C]
    sum_out_ptr,          # f32 [C]
    scale_grad_ptr,       # f32 [C]
    pool_out_ptr,         # bf16 [N,C,OUT_HW] flat
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    base = n * (C * HW) + c * HW + spatial

    mask_bf = ct.gather(mask_ptr, base)
    mask_f = ct.astype(mask_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill_scalar, ct.float32)
    source_bf = ct.gather(source_ptr, base)
    source_f = ct.astype(source_bf, ct.float32)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    where_f = ct.where(mask_f <= zero_f,
                       ct.broadcast_to(fill_f, (BLOCK_R,)),
                       source_f)
    where_f = ct.where(active, where_f, 0.0)

    centered_source_bf = ct.gather(centered_source_ptr, base)
    centered_source = ct.astype(centered_source_bf, ct.float32)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean = ct.broadcast_to(mean_scalar, (BLOCK_R,))
    centered = ct.where(active, centered_source - mean, 0.0)

    product = where_f * centered
    sum_where = ct.sum(where_f)      # scalar
    sum_centered = ct.sum(product)   # scalar

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
    bn_f32 = after_mean * output_scale_bc
    bn_bf = ct.astype(bn_f32, ct.bfloat16)

    def _bf16_add(a, b):
        return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32),
                         ct.bfloat16)

    # Twelve-way residual bf16 add (sequential — matches Triton chain).
    local = c * HW + spatial
    v0 = ct.gather(r0_ptr, n * (512 * HW) + local)
    v1 = ct.gather(r1_ptr, n * (480 * HW) + local)
    residual = _bf16_add(v0, v1)
    v2 = ct.gather(r2_ptr, n * (448 * HW) + local)
    residual = _bf16_add(residual, v2)
    v3 = ct.gather(r3_ptr, n * (416 * HW) + local)
    residual = _bf16_add(residual, v3)
    v4 = ct.gather(r4_ptr, n * (384 * HW) + local)
    residual = _bf16_add(residual, v4)
    v5 = ct.gather(r5_ptr, n * (352 * HW) + local)
    residual = _bf16_add(residual, v5)
    v6 = ct.gather(r6_ptr, n * (320 * HW) + local)
    residual = _bf16_add(residual, v6)
    v7 = ct.gather(r7_ptr, n * (288 * HW) + local)
    residual = _bf16_add(residual, v7)
    v8 = ct.gather(r8_ptr, n * (256 * HW) + local)
    residual = _bf16_add(residual, v8)
    v9 = ct.gather(r9_ptr, n * (224 * HW) + local)
    residual = _bf16_add(residual, v9)
    v10 = ct.gather(r10_ptr, n * (192 * HW) + local)
    residual = _bf16_add(residual, v10)
    v11 = ct.gather(r11_ptr, n * (160 * HW) + local)
    residual = _bf16_add(residual, v11)

    pool_parent = _bf16_add(residual, bn_bf)
    pool_value_f = ct.astype(pool_parent, ct.float32) * 0.25
    pool_value = ct.astype(pool_value_f, ct.bfloat16)

    h = spatial // W
    w_col = spatial - h * W
    out_base = n * (C * OUT_HW) + c * OUT_HW + h * (2 * OUT_W) + w_col * 2

    # Per-channel scalar outputs.
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    scale_grad_val = sum_centered * ct.reshape(invstd, ())
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(scale_grad_val, (1,)))

    # 2x2 nearest-neighbor pool upsample: four positions per active row.
    ct.scatter(pool_out_ptr, out_base, pool_value, mask=active)
    ct.scatter(pool_out_ptr, out_base + 1, pool_value, mask=active)
    ct.scatter(pool_out_ptr, out_base + OUT_W, pool_value, mask=active)
    ct.scatter(pool_out_ptr, out_base + OUT_W + 1, pool_value, mask=active)


@oracle_impl(hardware="B200", point="571e1d6c", BLOCK_R=4096)
def oracle_forward(inputs, *, BLOCK_R: int):
    (a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
     a12, a13, a14, a15, a16, a17, a18, _a19) = inputs
    device = a12.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    pool_out = torch.empty_strided(
        (N, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    def _flat(t):
        size = 1
        for s in t.shape:
            size *= s
        return t.contiguous().view(size)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _densenet_pool_kernel,
        (_flat(a0), _flat(a1), _flat(a2), _flat(a3),
         _flat(a4), _flat(a5), _flat(a6), _flat(a7),
         _flat(a8), _flat(a9), _flat(a10), _flat(a11),
         _flat(a12), a13.contiguous().view(1), _flat(a14), _flat(a15),
         a16.contiguous().view(C),
         a17.contiguous().view(C),
         a18.contiguous().view(C),
         sum_out, scale_grad, pool_out.view(-1),
         BLOCK_R),
    )

    return sum_out, scale_grad, pool_out
