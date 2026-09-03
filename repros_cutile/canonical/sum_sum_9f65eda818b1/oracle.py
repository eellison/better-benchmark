"""cuTile port of sum_sum_9f65eda818b1: phlippe DenseNet BN backward tail.

For each channel (160), compute over K_TOTAL=8192 elements:
  - selected = where(mask <= 0, fill, source)  [bf16]
  - centered = centered_source - mean[c]
  - sum(selected), sum(selected * centered) * invstd -> f32 side outputs
  - BN-backward epilogue: grad = (selected - centered * (dot*scale*invstd^2) - sum*scale) * invstd * weight
  - add_out = bf16(residual[..., c+16, :, :] + grad_bf16)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 160
INPUT_C = 176
SLICE_OFFSET = 16
SLICE_C = 16
H = 8
W = 8
HW = H * W  # 64
K_TOTAL = N * HW  # 8192
SCALE = 0.0001220703125  # 1/K_TOTAL


@ct.kernel
def _bn_tail_kernel(
    residual_ptr,          # bf16 (N, INPUT_C, H, W) flat
    mask_ptr,              # bf16 (N, C, H, W)
    fill_ptr,              # bf16 scalar
    source_ptr,            # bf16 (N, C, H, W)
    centered_source_ptr,   # bf16 (N, C, H, W)
    mean_ptr,              # f32 (C,) flattened
    invstd_ptr,            # f32 (C,)
    weight_ptr,            # f32 (C,)
    sum_out_ptr,           # f32 (C,)
    scale_grad_ptr,        # f32 (C,)
    add_out_ptr,           # bf16 (N, C, H, W)
    C_: ct.Constant[int],
    INPUT_C_: ct.Constant[int],
    SLICE_OFFSET_: ct.Constant[int],
    HW_: ct.Constant[int],
    K_TOTAL_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    k = ct.arange(BLOCK_K, dtype=ct.int32)
    active = k < K_TOTAL_
    n = k // HW_
    spatial = k - n * HW_
    compact_offs = n * (C_ * HW_) + c * HW_ + spatial
    residual_offs = n * (INPUT_C_ * HW_) + (c + SLICE_OFFSET_) * HW_ + spatial

    gate = ct.gather(mask_ptr, compact_offs, mask=active, padding_value=0)
    source = ct.gather(source_ptr, compact_offs, mask=active, padding_value=0)
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    zero_bf = ct.zeros((BLOCK_K,), dtype=ct.bfloat16)
    fill_broad = ct.broadcast_to(ct.reshape(fill, (1,)), (BLOCK_K,))
    selected_bf = ct.where(gate <= zero_bf, fill_broad, source)
    selected = ct.astype(selected_bf, ct.float32)

    activation = ct.astype(
        ct.gather(centered_source_ptr, compact_offs, mask=active, padding_value=0),
        ct.float32)
    mean = ct.astype(
        ct.gather(mean_ptr, ct.full((1,), c, dtype=ct.int32)), ct.float32)
    mean_broad = ct.broadcast_to(mean, (BLOCK_K,))
    centered = activation - mean_broad
    dot_terms = selected * centered

    zero_f = ct.zeros((BLOCK_K,), dtype=ct.float32)
    sum_value = ct.sum(ct.where(active, selected, zero_f))
    dot_value = ct.sum(ct.where(active, dot_terms, zero_f))
    invstd = ct.astype(
        ct.gather(invstd_ptr, ct.full((1,), c, dtype=ct.int32)), ct.float32)
    weight = ct.astype(
        ct.gather(weight_ptr, ct.full((1,), c, dtype=ct.int32)), ct.float32)

    ct.scatter(sum_out_ptr, ct.full((1,), c, dtype=ct.int32),
               ct.reshape(sum_value, (1,)))
    ct.scatter(scale_grad_ptr, ct.full((1,), c, dtype=ct.int32),
               ct.reshape(dot_value * invstd, (1,)))

    mean_term = sum_value * SCALE_
    dot_mean = dot_value * SCALE_
    invstd_sq = invstd * invstd
    correction_scale = dot_mean * invstd_sq
    output_scale = invstd * weight
    # centered * correction_scale (broadcast: correction_scale is (1,), centered (BLOCK_K,))
    cs_broad = ct.broadcast_to(correction_scale, (BLOCK_K,))
    correction = centered * cs_broad
    mt_broad = ct.broadcast_to(mean_term, (BLOCK_K,))
    os_broad = ct.broadcast_to(output_scale, (BLOCK_K,))
    grad = (selected - correction - mt_broad) * os_broad
    grad_bf = ct.astype(grad, ct.bfloat16)

    residual = ct.gather(residual_ptr, residual_offs, mask=active, padding_value=0)
    add_val = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(grad_bf, ct.float32),
        ct.bfloat16)
    ct.scatter(add_out_ptr, compact_offs, add_val, mask=active)


@oracle_impl(hardware="B200", point="87b141c7")
def oracle_forward(inputs):
    residual, mask, fill, source, centered_source, mean, invstd, weight = inputs
    device = mask.device
    BLOCK_K = 8192

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _bn_tail_kernel,
        (
            residual.reshape(-1), mask.reshape(-1), fill.view(1),
            source.reshape(-1), centered_source.reshape(-1),
            mean.reshape(-1), invstd.reshape(-1), weight.reshape(-1),
            sum_out, scale_grad, add_out.reshape(-1),
            C, INPUT_C, SLICE_OFFSET, HW, K_TOTAL, SCALE, BLOCK_K,
        ),
    )
    slice_out = torch.as_strided(
        add_out,
        (N, SLICE_C, H, W),
        (C * HW, HW, W, 1),
    )
    return sum_out, scale_grad, add_out, slice_out
