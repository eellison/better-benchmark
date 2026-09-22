"""cuTile port of sum_sum_0c9cec267360: DenseNet BN backward + masked producer.

One channel-resident cuTile program per output channel — mirrors Triton's
1-kernel structure with BLOCK_N=128, BLOCK_HW=16. Shares the
`where(mask <= 0, fill, source)` producer across sum_1 = sum(where.f32)
and sum_2 = sum(where * centered), then emits the BN-backward add_out.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 136
INPUT_C = 152
SLICE_OFFSET = 16
SLICE_C = 16
H = 4
W = 4
HW = H * W
SCALE = 0.00048828125


@ct.kernel
def _bn_tail_kernel(
    residual_ptr,        # bf16 (N, INPUT_C, H, W) contiguous
    mask_ptr,            # bf16 (N, C, H, W) contiguous
    fill_ptr,            # bf16 (1,)
    source_ptr,          # bf16 (N, C, H, W)
    centered_source_ptr, # bf16 (N, C, H, W)
    mean_ptr,            # f32 (C,)
    invstd_ptr,          # f32 (C,)
    weight_ptr,          # f32 (C,)
    sum_out_ptr,         # f32 (C,)
    scale_grad_ptr,      # f32 (C,)
    add_out_ptr,         # bf16 (N, C, H, W)
    BLOCK_N_: ct.Constant[int],
    BLOCK_HW_: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    c = ct.bid(0)

    # Load the (N, 1, HW) channel-c strip. BLOCK_N=128 covers all N, BLOCK_HW=16 covers HW=16.
    mask_bf = ct.load(mask_ptr, index=(0, c, 0, 0), shape=(BLOCK_N_, 1, H, W))
    source_bf = ct.load(source_ptr, index=(0, c, 0, 0), shape=(BLOCK_N_, 1, H, W))
    centered_bf = ct.load(centered_source_ptr, index=(0, c, 0, 0), shape=(BLOCK_N_, 1, H, W))
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))
    fill_bcast = ct.reshape(fill_tile, (1, 1, 1, 1))

    zero_bf = ct.zeros((BLOCK_N_, 1, H, W), dtype=ct.bfloat16)
    where_bf = ct.where(mask_bf <= zero_bf, fill_bcast, source_bf)
    where_f = ct.astype(where_bf, ct.float32)

    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_bcast = ct.reshape(mean_1, (1, 1, 1, 1))
    centered_f = ct.astype(centered_bf, ct.float32) - mean_bcast
    dot = where_f * centered_f

    sum_value = ct.sum(where_f)  # 0-d
    dot_value = ct.sum(dot)      # 0-d

    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))

    sum_tile = ct.full((1,), sum_value, dtype=ct.float32)
    dot_tile = ct.full((1,), dot_value, dtype=ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=sum_tile)
    ct.store(scale_grad_ptr, index=(c,), tile=dot_tile * invstd_1)

    mean_term = sum_tile * SCALE_
    dot_scaled = dot_tile * SCALE_
    invstd_sq = invstd_1 * invstd_1
    correction_scale = ct.reshape(dot_scaled * invstd_sq, (1, 1, 1, 1))
    output_scale = ct.reshape(invstd_1 * weight_1, (1, 1, 1, 1))
    mean_term_bc = ct.reshape(mean_term, (1, 1, 1, 1))

    correction = centered_f * correction_scale
    after_corr = where_f - correction
    centered_grad = after_corr - mean_term_bc
    grad_f = centered_grad * output_scale
    grad_bf = ct.astype(grad_f, ct.bfloat16)

    # add_out = residual[:, c+SLICE_OFFSET, :, :] + grad_bf, rounded to bf16.
    residual_c = ct.load(
        residual_ptr, index=(0, c + SLICE_OFFSET, 0, 0),
        shape=(BLOCK_N_, 1, H, W),
    )
    added_f = ct.astype(residual_c, ct.float32) + ct.astype(grad_bf, ct.float32)
    added_bf = ct.astype(added_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(0, c, 0, 0), tile=added_bf)


@oracle_impl(hardware="B200", point="756f1cd5", BLOCK_N=128, BLOCK_HW=16)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_HW: int):
    residual, mask, fill, source, centered_source, mean, invstd, weight = inputs
    device = residual.device

    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (N, C, H, W), (C * HW, HW, W, 1),
        device=device, dtype=torch.bfloat16,
    )

    mean_1d = mean.view(C)
    invstd_1d = invstd.view(C)
    weight_1d = weight.view(C)
    fill_1d = fill.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C, 1, 1), _bn_tail_kernel,
        (residual, mask, fill_1d, source, centered_source,
         mean_1d, invstd_1d, weight_1d,
         sum_out, scale_grad, add_out,
         BLOCK_N, BLOCK_HW, SCALE),
    )
    slice_out = torch.as_strided(add_out, (N, SLICE_C, H, W), (C * HW, HW, W, 1))
    return sum_out, scale_grad, add_out, slice_out
