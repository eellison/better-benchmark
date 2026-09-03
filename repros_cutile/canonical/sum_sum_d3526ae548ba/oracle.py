"""cuTile port of sum_sum_d3526ae548ba (SCHEDULER_FUSION): densenet121 BN
backward + residual slice add.

Matches Triton: per-channel program iterates R spatial elements from
NCHW-contiguous arg tensors via gather (no pad copies).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


C = 416
HW = 784  # 28*28
N = 4
R = N * HW  # 3136
BLOCK_R = 4096  # next pow2 >= R
SLICE_START = 384
SLICE_C = 32
SCALE = 0.00031887755102040814


@ct.kernel
def _dense_kernel(
    mask_input_ptr,       # bf16 flat NCHW
    fill_ptr,             # bf16 [1]
    source_ptr,           # bf16 flat NCHW
    centered_source_ptr,  # bf16 flat NCHW
    mean_ptr,             # f32 [C]
    invstd_ptr,           # f32 [C]
    weight_ptr,           # f32 [C]
    sum_out_ptr,          # f32 [C]
    scale_grad_ptr,       # f32 [C]
    dense_out_ptr,        # bf16 flat NCHW
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    R_C: ct.Constant[int],
    BLOCK_R_C: ct.Constant[int],
    SCALE_C: ct.Constant[float],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R_C, dtype=ct.int32)
    active = rows < R_C
    n = rows // HW_C
    spatial = rows - n * HW_C
    offsets = n * (C_C * HW_C) + c * HW_C + spatial

    zero_bf = ct.zeros((BLOCK_R_C,), dtype=ct.bfloat16)
    mask_value = ct.gather(mask_input_ptr, (offsets,), mask=active)
    fill_scalar = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bc = ct.broadcast_to(fill_scalar, (BLOCK_R_C,))
    source_value = ct.gather(source_ptr, (offsets,), mask=active)
    where_bf16 = ct.where(mask_value <= zero_bf, fill_bc, source_value)
    where_f = ct.astype(where_bf16, ct.float32)
    where_f = ct.where(active, where_f, ct.zeros((BLOCK_R_C,), dtype=ct.float32))

    cs_bf = ct.gather(centered_source_ptr, (offsets,), mask=active)
    centered_source = ct.astype(cs_bf, ct.float32)
    mean_v = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_bc = ct.broadcast_to(mean_v, (BLOCK_R_C,))
    centered = centered_source - mean_bc
    centered = ct.where(active, centered, ct.zeros((BLOCK_R_C,), dtype=ct.float32))

    product = where_f * centered
    sum_where = ct.sum(where_f)
    sum_centered = ct.sum(product)

    invstd_v = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_v = ct.load(weight_ptr, index=(c,), shape=(1,))
    mean_term = sum_where * SCALE_C
    dot_scaled = sum_centered * SCALE_C
    invstd_sq = invstd_v * invstd_v
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd_v * weight_v

    mean_term_bc = ct.broadcast_to(ct.reshape(mean_term, (1,)), (BLOCK_R_C,))
    variance_term_bc = ct.broadcast_to(variance_term, (BLOCK_R_C,))
    output_scale_bc = ct.broadcast_to(output_scale, (BLOCK_R_C,))
    after_variance = where_f - centered * variance_term_bc
    after_mean = after_variance - mean_term_bc
    dense_bf16 = ct.astype(after_mean * output_scale_bc, ct.bfloat16)

    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_where, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(sum_centered * invstd_v, (1,)))
    ct.scatter(dense_out_ptr, (offsets,), dense_bf16, mask=active)


@ct.kernel
def _slice_add_kernel(
    r0_ptr,          # bf16 flat, r0 has 512 channels
    r1_ptr,          # bf16 flat, r1 has 480 channels
    r2_ptr,          # bf16 flat, r2 has 448 channels
    dense_out_ptr,   # bf16 flat, C=416 channels
    add_out_ptr,     # bf16 flat, SLICE_C=32 channels
    C_C: ct.Constant[int],
    HW_C: ct.Constant[int],
    R_C: ct.Constant[int],
    BLOCK_R_C: ct.Constant[int],
    SLICE_START_C: ct.Constant[int],
    R0_STRIDE_C: ct.Constant[int],
    R1_STRIDE_C: ct.Constant[int],
    R2_STRIDE_C: ct.Constant[int],
    C_STRIDE: ct.Constant[int],  # C * HW
    SLICE_C_C: ct.Constant[int],
):
    slice_c = ct.bid(0)
    channel = slice_c + SLICE_START_C

    rows = ct.arange(BLOCK_R_C, dtype=ct.int32)
    active = rows < R_C
    n = rows // HW_C
    spatial = rows - n * HW_C

    dense_offsets = n * C_STRIDE + channel * HW_C + spatial
    r0_offsets = n * R0_STRIDE_C + channel * HW_C + spatial
    r1_offsets = n * R1_STRIDE_C + channel * HW_C + spatial
    r2_offsets = n * R2_STRIDE_C + channel * HW_C + spatial
    add_offsets = n * (SLICE_C_C * HW_C) + slice_c * HW_C + spatial

    dense_bf = ct.gather(dense_out_ptr, (dense_offsets,), mask=active)
    v0 = ct.gather(r0_ptr, (r0_offsets,), mask=active)
    v1 = ct.gather(r1_ptr, (r1_offsets,), mask=active)
    v2 = ct.gather(r2_ptr, (r2_offsets,), mask=active)

    residual = ct.astype(
        ct.astype(v0, ct.float32) + ct.astype(v1, ct.float32), ct.bfloat16
    )
    residual = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(v2, ct.float32), ct.bfloat16
    )
    add_value = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(dense_bf, ct.float32),
        ct.bfloat16,
    )
    ct.scatter(add_out_ptr, (add_offsets,), add_value, mask=active)


@oracle_impl(hardware="B200", point="7eb90cb4", BLOCK_R=4096)
def oracle_forward(inputs, *, BLOCK_R):
    (
        r0, r1, r2, mask_input, fill, source, centered_source,
        mean, invstd, weight,
    ) = inputs
    device = source.device

    mask_flat = mask_input.view(-1)
    source_flat = source.view(-1)
    cs_flat = centered_source.view(-1)
    r0_flat = r0.view(-1)
    r1_flat = r1.view(-1)
    r2_flat = r2.view(-1)

    dense_out = torch.empty_strided(
        (N, C, 28, 28),
        (C * HW, HW, 28, 1),
        device=device, dtype=torch.bfloat16,
    )
    dense_flat = dense_out.view(-1)

    add_out = torch.empty_strided(
        (N, SLICE_C, 28, 28),
        (SLICE_C * HW, HW, 28, 1),
        device=device, dtype=torch.bfloat16,
    )
    add_flat = add_out.view(-1)

    mean_flat = mean.view(C)
    invstd_flat = invstd.view(C)
    weight_flat = weight.view(C)
    fill_flat = fill.view(1)

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)

    r0_c = int(r0.shape[1])  # 512
    r1_c = int(r1.shape[1])  # 480
    r2_c = int(r2.shape[1])  # 448

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _dense_kernel,
        (mask_flat, fill_flat, source_flat, cs_flat, mean_flat,
         invstd_flat, weight_flat, sum_out, scale_grad, dense_flat,
         C, HW, R, BLOCK_R, SCALE),
    )
    ct.launch(
        stream,
        (SLICE_C, 1, 1),
        _slice_add_kernel,
        (r0_flat, r1_flat, r2_flat, dense_flat, add_flat,
         C, HW, R, BLOCK_R, SLICE_START,
         r0_c * HW, r1_c * HW, r2_c * HW, C * HW, SLICE_C),
    )

    return sum_out, scale_grad, dense_out, add_out
