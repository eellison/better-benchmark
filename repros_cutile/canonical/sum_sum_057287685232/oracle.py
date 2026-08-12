"""cuTile port of sum_sum_057287685232 (SCHEDULER_FUSION): DenseNet
transition-backward tail with BN backward + 2x2 avg_pool_backward.

For each channel c, reduce across BLOCK_R = N*HW=8192 rows to compute sum_1
and sum_2, apply the BN backward epilogue, add sliced residual, and write the
2x2 broadcast pool_backward output. cuTile's default fp32 arithmetic is
IEEE round-to-nearest-even, so the Triton inline PTX becomes plain
arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 80
H = 8
W = 8
HW = H * W
OUT_H = 16
OUT_W = 16
OUT_HW = OUT_H * OUT_W
R = N * HW
SCALE = 0.0001220703125


@ct.kernel
def _densenet_pool_kernel(
    residual_ptr,          # bf16 [N, 96, 8, 8]
    mask_ptr,              # bf16 [N, 80, 8, 8]
    fill_ptr,              # bf16 []
    source_ptr,            # bf16 [N, 80, 8, 8]
    centered_source_ptr,   # bf16 [N, 80, 8, 8]
    mean_ptr,              # f32 [1, 80, 1, 1]
    invstd_ptr,            # f32 [80]
    weight_ptr,            # f32 [80]
    sum_out_ptr,           # f32 [80]
    scale_grad_ptr,        # f32 [80]
    pool_out_ptr,          # bf16 [N, 80, 16, 16]
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    n = rows // 64
    spatial = rows - n * 64
    # base = n * (80 * 64) + c * 64 + spatial
    base = n * (80 * 64) + c * 64 + spatial

    mask_value = ct.astype(ct.gather(mask_ptr, (base,)), ct.float32)
    fill_value = ct.astype(ct.load(fill_ptr, index=(0,), shape=(1,)), ct.float32)
    fill_bcast = ct.reshape(fill_value, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    source_value = ct.astype(ct.gather(source_ptr, (base,)), ct.float32)
    zero_f = ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    where_f32 = ct.where(mask_value <= zero_f, fill_bcast, source_value)
    # Simulate the bf16 rounding boundary from the where cast
    where_bf16 = ct.astype(where_f32, ct.bfloat16)
    where_f32 = ct.astype(where_bf16, ct.float32)

    centered_source = ct.astype(ct.gather(centered_source_ptr, (base,)), ct.float32)
    mean_scalar = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    mean_bcast = ct.reshape(mean_scalar, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    centered = centered_source - mean_bcast

    product = where_f32 * centered
    sum_where = ct.sum(where_f32)
    sum_centered = ct.sum(product)

    invstd_scalar = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_scalar = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd = ct.astype(invstd_scalar, ct.float32)
    weight = ct.astype(weight_scalar, ct.float32)
    mean_term = sum_where * SCALE
    dot_scaled = sum_centered * SCALE
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight

    variance_bc = ct.reshape(variance_term, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    mean_term_bc = ct.reshape(mean_term, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    scale_bc = ct.reshape(output_scale, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)

    after_variance = where_f32 - centered * variance_bc
    after_mean = after_variance - mean_term_bc
    bn_bf16 = ct.astype(after_mean * scale_bc, ct.bfloat16)

    # residual: n * 96 * 64 + (c + 16) * 64 + spatial
    residual_offset = n * (96 * 64) + (c + 16) * 64 + spatial
    residual = ct.gather(residual_ptr, (residual_offset,))
    pool_parent = ct.astype(residual, ct.float32) + ct.astype(bn_bf16, ct.float32)
    pool_parent_bf16 = ct.astype(pool_parent, ct.bfloat16)
    pool_value = ct.astype(ct.astype(pool_parent_bf16, ct.float32) * 0.25, ct.bfloat16)

    h = spatial // 8
    w = spatial - h * 8
    # out_base = n * (80 * 256) + c * 256 + h * 32 + w * 2
    out_base = n * (80 * 256) + c * 256 + h * 32 + w * 2

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    grad_val = sum_centered * invstd
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(grad_val, (1,)))

    # Broadcast pool_value into a 2x2 block via scatter
    ct.scatter(pool_out_ptr, (out_base,), pool_value)
    ct.scatter(pool_out_ptr, (out_base + 1,), pool_value)
    ct.scatter(pool_out_ptr, (out_base + 16,), pool_value)
    ct.scatter(pool_out_ptr, (out_base + 17,), pool_value)


@oracle_impl(hardware="B200", point="529a48b9", BLOCK_R=8192)
def oracle_forward(inputs, *, BLOCK_R: int):
    (
        arg0_1,   # bf16 [128, 96, 8, 8]
        arg1_1,   # bf16 [128, 80, 8, 8]
        arg2_1,   # bf16 []
        arg3_1,   # bf16 [128, 80, 8, 8]
        arg4_1,   # bf16 [128, 80, 8, 8]
        arg5_1,   # f32 [1, 80, 1, 1]
        arg6_1,   # f32 [80]
        arg7_1,   # f32 [80]
        _arg8_1,  # bf16 [128, 80, 16, 16] (not used — pool_backward output shape)
    ) = inputs
    device = arg0_1.device

    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    pool_out = torch.empty_strided(
        (N, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # Flatten arg0 (residual) to 1D for gather
    residual_flat = arg0_1.view(-1)
    mask_flat = arg1_1.view(-1)
    source_flat = arg3_1.view(-1)
    centered_source_flat = arg4_1.view(-1)
    fill_flat = arg2_1.view(1)
    # arg5 shape (1, 80, 1, 1) — view as (80,)
    mean_flat = arg5_1.view(C)
    pool_flat = pool_out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _densenet_pool_kernel,
        (residual_flat, mask_flat, fill_flat, source_flat, centered_source_flat,
         mean_flat, arg6_1, arg7_1, sum_out, scale_grad, pool_flat, BLOCK_R),
    )
    return sum_out, scale_grad, pool_out
