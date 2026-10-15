"""cuTile port of sum_sum_b58d306002be: DenseNet BN-backward + residual add.

Per-channel kernel (grid = C):
  - reduces sum(selected) and sum(selected * centered) over the full [N, HW]
    axis, storing the per-channel scalars,
  - broadcasts the scalars back and writes the bf16 add output with the
    residual slice (channels [SLICE_START, SLICE_START + C)).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 104
INPUT_C = 120
SLICE_START = 16
H = 4
W = 4
HW = H * W
K_TOTAL = N * HW
SCALE = 0.00048828125


@ct.kernel
def _bn_tail_kernel(
    residual_ptr,        # bf16 [N, INPUT_C, H, W]
    mask_ptr,            # bf16 [N, C, H, W]
    fill_ptr,            # bf16 [1]
    rhs_ptr,             # bf16 [N, C, H, W]
    activation_ptr,      # bf16 [N, C, H, W]
    mean_ptr,            # f32  [C]
    invstd_ptr,          # f32  [C]
    weight_ptr,          # f32  [C]
    sum_out_ptr,         # f32  [C]
    scaled_dot_out_ptr,  # f32  [C]
    add_out_ptr,         # bf16 [N, C, H, W]
    SLICE_START_: ct.Constant[int],
    HW_: ct.Constant[int],
    K_TOTAL_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK_K: ct.Constant[int],
):
    c = ct.bid(0)
    k = ct.arange(BLOCK_K, dtype=ct.int32)
    active = k < K_TOTAL_

    n = k // HW_
    hw = k - n * HW_
    h = hw // W
    w = hw - h * W
    c_t = ct.full((BLOCK_K,), c, dtype=ct.int32)
    c_res_t = ct.full((BLOCK_K,), c + SLICE_START_, dtype=ct.int32)

    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))

    mask_v = ct.gather(mask_ptr, (n, c_t, h, w), mask=active,
                       padding_value=ct.bfloat16(0.0))
    rhs_v = ct.gather(rhs_ptr, (n, c_t, h, w), mask=active,
                      padding_value=ct.bfloat16(0.0))
    act_v = ct.gather(activation_ptr, (n, c_t, h, w), mask=active,
                      padding_value=ct.bfloat16(0.0))
    residual_v = ct.gather(residual_ptr, (n, c_res_t, h, w), mask=active,
                           padding_value=ct.bfloat16(0.0))

    fill_v = ct.full((BLOCK_K,), 0.0, dtype=ct.bfloat16) + ct.reshape(fill, (1,))
    selected_bf = ct.where(mask_v <= ct.astype(0.0, ct.bfloat16), fill_v, rhs_v)
    selected = ct.astype(selected_bf, ct.float32)

    mean_f = ct.astype(mean, ct.float32)
    invstd_f = ct.astype(invstd, ct.float32)
    weight_f = ct.astype(weight, ct.float32)

    centered = ct.astype(act_v, ct.float32) - ct.broadcast_to(mean_f, (BLOCK_K,))
    dot_terms = selected * centered

    sel_masked = ct.where(active, selected, 0.0)
    dot_masked = ct.where(active, dot_terms, 0.0)
    sum_val = ct.sum(sel_masked)
    dot_val = ct.sum(dot_masked)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(sum_val, (1,)))
    scaled_dot = dot_val * invstd_f
    ct.store(scaled_dot_out_ptr, index=(c,), tile=ct.reshape(scaled_dot, (1,)))

    mean_term = sum_val * SCALE_
    dot_mean = dot_val * SCALE_
    invstd_sq = invstd_f * invstd_f
    variance_scale = ct.reshape(dot_mean * invstd_sq, (1,))
    output_scale = ct.reshape(invstd_f * weight_f, (1,))

    var_scale_bcast = ct.broadcast_to(variance_scale, (BLOCK_K,))
    out_scale_bcast = ct.broadcast_to(output_scale, (BLOCK_K,))
    mean_term_bcast = ct.full((BLOCK_K,), 0.0, dtype=ct.float32) + ct.reshape(mean_term, (1,))

    correction = centered * var_scale_bcast
    after_correction = selected - correction
    after_mean = after_correction - mean_term_bcast
    grad_bf = ct.astype(after_mean * out_scale_bcast, ct.bfloat16)

    add_val = ct.astype(
        ct.astype(residual_v, ct.float32) + ct.astype(grad_bf, ct.float32),
        ct.bfloat16,
    )
    ct.scatter(add_out_ptr, (n, c_t, h, w), add_val, mask=active)


@oracle_impl(hardware="B200", point="4df6a9d1", BLOCK_K=2048)
def oracle_forward(inputs, *, BLOCK_K: int):
    residual, mask, fill, rhs, activation, mean, invstd, weight = inputs

    device = mask.device
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    fill_1d = fill.view(1)
    mean_1d = mean.view(C)

    # BLOCK_K must be at least K_TOTAL=2048; use 2048.
    block_k = K_TOTAL if BLOCK_K < K_TOTAL else BLOCK_K
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_tail_kernel,
        (residual, mask, fill_1d, rhs, activation, mean_1d, invstd, weight,
         sum_out, scaled_dot_out, add_out,
         SLICE_START, HW, K_TOTAL, SCALE, block_k),
    )

    slice_out = torch.as_strided(
        add_out,
        (N, SLICE_START, H, W),
        (C * HW, HW, W, 1),
    )
    return sum_out, scaled_dot_out, add_out, slice_out
