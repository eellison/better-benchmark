"""cuTile port of sum_sum_47b68e6603ba (SCHEDULER_FUSION): DenseNet
BN-backward tail. Per-channel c reduce across BLOCK_R = N*HW = 2048 rows.

cuTile's default fp32 arithmetic is IEEE round-to-nearest-even, so the
Triton inline PTX becomes plain arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 168
INPUT_C = 184
SLICE_OFFSET = 16
SLICE_C = 16
H = 4
W = 4
HW = H * W
R = N * HW   # 2048
SCALE = 0.00048828125


@ct.kernel
def _bn_tail_kernel(
    residual_ptr,          # bf16 [N, 184, 4, 4]
    mask_ptr,              # bf16 [N, 168, 4, 4]
    fill_ptr,              # bf16 []
    source_ptr,            # bf16 [N, 168, 4, 4]
    centered_source_ptr,   # bf16 [N, 168, 4, 4]
    mean_ptr,              # f32 [168]
    invstd_ptr,            # f32 [168]
    weight_ptr,            # f32 [168]
    sum_out_ptr,           # f32 [168]
    scale_grad_ptr,        # f32 [168]
    add_out_ptr,           # bf16 [N, 168, 4, 4]
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    n = rows // 16
    spatial = rows - n * 16

    compact_offsets = n * (168 * 16) + c * 16 + spatial
    residual_offsets = n * (184 * 16) + (c + 16) * 16 + spatial

    gate = ct.astype(ct.gather(mask_ptr, (compact_offsets,)), ct.float32)
    source = ct.astype(ct.gather(source_ptr, (compact_offsets,)), ct.float32)
    fill_scalar = ct.astype(ct.load(fill_ptr, index=(0,), shape=(1,)), ct.float32)
    fill_bc = ct.reshape(fill_scalar, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    zero_f = ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    selected_f32 = ct.where(gate <= zero_f, fill_bc, source)
    # bf16 rounding boundary
    selected_bf16 = ct.astype(selected_f32, ct.bfloat16)
    selected = ct.astype(selected_bf16, ct.float32)

    activation = ct.astype(ct.gather(centered_source_ptr, (compact_offsets,)), ct.float32)
    mean_scalar = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean = ct.astype(mean_scalar, ct.float32)
    mean_bc = ct.reshape(mean, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    centered = activation - mean_bc

    dot_terms = selected * centered
    sum_value = ct.sum(selected)
    dot_value = ct.sum(dot_terms)

    invstd = ct.astype(ct.load(invstd_ptr, index=(c,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_value, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(dot_value * invstd, (1,)))

    mean_term = sum_value * SCALE
    dot_mean = dot_value * SCALE
    invstd_sq = invstd * invstd
    correction_scale = dot_mean * invstd_sq
    output_scale = invstd * weight

    correction_bc = ct.reshape(correction_scale, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    mean_term_bc = ct.reshape(mean_term, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)
    output_scale_bc = ct.reshape(output_scale, (1,)) + ct.full(shape=(BLOCK_R,), fill_value=0.0, dtype=ct.float32)

    correction = centered * correction_bc
    after_correction = selected - correction
    after_mean = after_correction - mean_term_bc
    grad_bf16 = ct.astype(after_mean * output_scale_bc, ct.bfloat16)

    residual = ct.gather(residual_ptr, (residual_offsets,))
    add_value = ct.astype(ct.astype(residual, ct.float32) + ct.astype(grad_bf16, ct.float32),
                          ct.bfloat16)
    ct.scatter(add_out_ptr, (compact_offsets,), add_value)


@oracle_impl(hardware="B200", point="6bc30484", BLOCK_R=2048)
def oracle_forward(inputs, *, BLOCK_R: int):
    residual, mask, fill, source, centered_source, mean, invstd, weight = inputs

    sum_out = torch.empty((C,), device=mask.device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=mask.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=mask.device, dtype=torch.bfloat16,
    )

    residual_flat = residual.view(-1)
    mask_flat = mask.view(-1)
    source_flat = source.view(-1)
    centered_source_flat = centered_source.view(-1)
    fill_flat = fill.view(1)
    mean_flat = mean.view(C)
    add_flat = add_out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _bn_tail_kernel,
        (residual_flat, mask_flat, fill_flat, source_flat, centered_source_flat,
         mean_flat, invstd, weight, sum_out, scale_grad, add_flat, BLOCK_R),
    )
    slice_out = torch.as_strided(add_out, (N, SLICE_C, H, W), (C * HW, HW, W, 1))
    return sum_out, scale_grad, add_out, slice_out
