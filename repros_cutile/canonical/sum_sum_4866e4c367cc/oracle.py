"""cuTile port of sum_sum_4866e4c367cc: DenseNet BN backward tail (768ch).

Mirrors Triton's `_fused_channel_kernel` which does per-channel reductions
(`tl.sum(..., axis=0)` on where-values and product) inside the kernel via one
program per channel, then computes the BN-backward dense output and residual
tail-add all in one kernel. cuTile computes the same reductions with
`ct.sum(...)` inside the kernel.

Grid `(C,)` with BLOCK_K covering N*HW rows (784 for H=W=14 -> BLOCK=1024;
196 for H=W=7 -> BLOCK=256).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 768
N = 4
SLICE_START = 736
SLICE_C = 32
SCALE = 0.0012755102040816326


@ct.kernel
def _fused_channel_kernel(
    r0_ptr, r1_ptr, r2_ptr, r3_ptr, r4_ptr, r5_ptr, r6_ptr, r7_ptr,
    mask_ptr,           # bf16 [N,C,H,W] contig flat
    fill_ptr,           # bf16 scalar
    source_ptr,         # bf16 [N,C,H,W] contig flat
    centered_source_ptr, # f32 [N,C,H,W] contig flat
    mean_ptr,           # f32 [C]
    invstd_ptr,         # f32 [C]
    weight_ptr,         # f32 [C]
    sum_out_ptr,        # f32 [C]
    vec_out_ptr,        # f32 [C]
    dense_out_ptr,      # bf16 [N,C,H,W] contig flat
    tail_out_ptr,       # bf16 [N, SLICE_C, H, W] contig flat
    HW: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    total_spatial = N * HW
    rows = ct.arange(BLOCK_K, dtype=ct.int32)
    active = rows < total_spatial
    n = rows // HW
    spatial = rows - n * HW
    offsets = n * (C * HW) + c * HW + spatial

    mask_bf = ct.gather(mask_ptr, offsets)
    mask_value = ct.astype(mask_bf, ct.float32)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_value = ct.astype(fill_scalar, ct.float32)
    source_bf = ct.gather(source_ptr, offsets)
    source_value = ct.astype(source_bf, ct.float32)
    where_value = ct.where(mask_value <= 0.0,
                           ct.broadcast_to(fill_value, (BLOCK_K,)),
                           source_value)
    where_active = ct.where(active, where_value, 0.0)

    centered_source = ct.gather(centered_source_ptr, offsets)
    mean_scalar = ct.gather(mean_ptr, ct.broadcast_to(c, (1,)))
    mean = ct.broadcast_to(mean_scalar, (BLOCK_K,))
    centered = centered_source - mean
    product = ct.where(active, where_value * centered, 0.0)

    sum1 = ct.sum(where_active)  # scalar
    sum2 = ct.sum(product)       # scalar

    invstd_scalar = ct.gather(invstd_ptr, ct.broadcast_to(c, (1,)))
    weight_scalar = ct.gather(weight_ptr, ct.broadcast_to(c, (1,)))
    invstd = ct.reshape(invstd_scalar, (1,))
    weight = ct.reshape(weight_scalar, (1,))

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum1, (1,)))
    vec_val = sum2 * ct.reshape(invstd, ())
    ct.store(vec_out_ptr, index=(c,), tile=ct.reshape(vec_val, (1,)))

    mean_term = sum1 * SCALE
    sum2_scaled = sum2 * SCALE
    invstd_sq = invstd * invstd
    variance_term = sum2_scaled * invstd_sq
    output_scale = invstd * weight

    variance_bc = ct.broadcast_to(variance_term, (BLOCK_K,))
    mean_bc = ct.broadcast_to(mean_term, (BLOCK_K,))
    scale_bc = ct.broadcast_to(output_scale, (BLOCK_K,))

    centered_scaled = centered * variance_bc
    sub1 = where_value - centered_scaled
    sub2 = sub1 - mean_bc
    dense = sub2 * scale_bc
    dense_bf = ct.astype(dense, ct.bfloat16)
    ct.scatter(dense_out_ptr, offsets, dense_bf, mask=active)

    # Tail add for c in [SLICE_START, C).
    in_slice = c >= SLICE_START
    if in_slice:
        slice_c_val = c - SLICE_START
        tail_offsets = n * (SLICE_C * HW) + slice_c_val * HW + spatial

        v0 = ct.gather(r0_ptr, n * (1024 * HW) + c * HW + spatial)
        v1 = ct.gather(r1_ptr, n * (992 * HW) + c * HW + spatial)
        v2 = ct.gather(r2_ptr, n * (960 * HW) + c * HW + spatial)
        v3 = ct.gather(r3_ptr, n * (928 * HW) + c * HW + spatial)
        v4 = ct.gather(r4_ptr, n * (896 * HW) + c * HW + spatial)
        v5 = ct.gather(r5_ptr, n * (864 * HW) + c * HW + spatial)
        v6 = ct.gather(r6_ptr, n * (832 * HW) + c * HW + spatial)
        v7 = ct.gather(r7_ptr, n * (800 * HW) + c * HW + spatial)

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
        tail_value = _bf16_add(residual, dense_bf)
        ct.scatter(tail_out_ptr, tail_offsets, tail_value, mask=active)


@oracle_impl(hardware="B200", point="50808975", BLOCK_K=1024)
@oracle_impl(hardware="B200", point="13ede9b8", BLOCK_K=256)
def oracle_forward(inputs, *, BLOCK_K: int):
    (
        r0, r1, r2, r3, r4, r5, r6, r7,
        mask, fill, source, centered_source, mean, invstd, weight,
    ) = inputs
    height = int(mask.shape[2])
    width = int(mask.shape[3])
    hw = height * width
    device = mask.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    vec_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, height, width),
        (C * hw, hw, width, 1),
        device=device, dtype=torch.bfloat16,
    )
    tail_out = torch.empty_strided(
        (N, SLICE_C, height, width),
        (SLICE_C * hw, hw, width, 1),
        device=device, dtype=torch.bfloat16,
    )

    def _flat(t):
        size = 1
        for s in t.shape:
            size *= s
        return t.contiguous().view(size)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _fused_channel_kernel,
        (_flat(r0), _flat(r1), _flat(r2), _flat(r3),
         _flat(r4), _flat(r5), _flat(r6), _flat(r7),
         _flat(mask), fill.view(1), _flat(source), _flat(centered_source),
         mean.view(C).contiguous(), invstd.view(C).contiguous(),
         weight.view(C).contiguous(),
         sum_out, vec_out,
         dense_out.view(-1), tail_out.view(-1),
         hw, BLOCK_K),
    )
    return sum_out, vec_out, dense_out, tail_out
