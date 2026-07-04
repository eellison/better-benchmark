"""cuTile port of sum_sum_77f6be69be60: ShuffleNet ReLU-gated BN-backward.
3-kernel plan:
  1) partial reduce: per (hw, c) sums selected / dot over batch axis.
  2) finalize: sum-over-hw partials to produce per-channel outputs and stats.
  3) epilogue: compute dense output tensor.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 1024
H = 7
W = 7
HW = H * W  # 49
NHW = N * HW
TOTAL = N * C * HW
INV_HW = 0.02040816326530612
INV_NHW = 0.00015943877551020407


@ct.kernel
def _partial_reduce_kernel(
    grad_ptr,      # bf16 [128, 1024]  (in element order)
    bn_ptr,        # bf16 flat NHWC [N * H * W * C]
    mean_ptr,      # f32 [C]
    invstd_ptr,    # f32 [C]
    weight_ptr,    # f32 [C]
    bias_ptr,      # f32 [C]
    partials_ptr,  # f32 flat [2 * HW * C]
    full_ptr,      # bf16 scalar
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    hw = ct.bid(1)
    c_vec = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask_1d = c_vec < 1024

    n_1d = ct.arange(128, dtype=ct.int32)
    zero_bool_2d = ct.full((128, BLOCK_C), False, dtype=ct.bool_)
    c_mask_2d = ct.reshape(c_mask_1d, (1, BLOCK_C)) | zero_bool_2d
    zero_i32_2d = ct.zeros((128, BLOCK_C), dtype=ct.int32)
    c_2d = ct.reshape(c_vec, (1, BLOCK_C)) + zero_i32_2d
    n_2d = ct.reshape(n_1d, (128, 1)) + zero_i32_2d

    # offsets: NHWC layout, flat = n * (H*W*C) + hw * C + c
    offsets = n_2d * (H * W * C) + hw * C + c_2d

    bn = ct.astype(ct.gather(bn_ptr, (offsets,), mask=c_mask_2d, padding_value=ct.bfloat16(0.0)), ct.float32)
    mean = ct.gather(mean_ptr, (c_2d,), mask=c_mask_2d, padding_value=ct.float32(0.0))
    invstd = ct.gather(invstd_ptr, (c_2d,), mask=c_mask_2d, padding_value=ct.float32(0.0))
    weight = ct.gather(weight_ptr, (c_2d,), mask=c_mask_2d, padding_value=ct.float32(0.0))
    bias = ct.gather(bias_ptr, (c_2d,), mask=c_mask_2d, padding_value=ct.float32(0.0))

    centered = bn - mean
    affine_bf16 = ct.astype(((centered * invstd) * weight) + bias, ct.bfloat16)

    # source_n = offsets // 50176 = n (since 50176 = 1024*49)
    # source_bf16 = (grad[source_n, c] * INV_HW) as bf16
    source_offsets = n_2d * 1024 + c_2d
    grad_val = ct.astype(ct.gather(grad_ptr, (source_offsets,), mask=c_mask_2d, padding_value=ct.bfloat16(0.0)), ct.float32)
    source_bf16 = ct.astype(grad_val * 0.02040816326530612, ct.bfloat16)
    zero_bf16 = ct.astype(
        ct.astype(source_bf16, ct.float32) * 0.0,
        ct.bfloat16,
    )
    selected = ct.astype(
        ct.where(affine_bf16 <= ct.astype(ct.full((128, BLOCK_C), 0.0, dtype=ct.float32), ct.bfloat16),
                  zero_bf16, source_bf16),
        ct.float32,
    )
    zero_f_2d = ct.zeros((128, BLOCK_C), dtype=ct.float32)
    selected = ct.where(c_mask_2d, selected, zero_f_2d)
    dot = ct.where(c_mask_2d, selected * centered, zero_f_2d)

    sel_sum = ct.sum(selected, axis=0)
    dot_sum = ct.sum(dot, axis=0)

    out_offsets = hw * 1024 + c_vec
    ct.scatter(partials_ptr, (out_offsets,), sel_sum, mask=c_mask_1d)
    ct.scatter(partials_ptr, (out_offsets + 50176,), dot_sum, mask=c_mask_1d)

    # Full scalar: bf16(0.0). Store from block (0, 0) once.
    if c_block == 0:
        if hw == 0:
            ct.store(full_ptr, index=(0,), tile=ct.full((1,), 0.0, dtype=ct.bfloat16))


@ct.kernel
def _finalize_kernel(
    partials_ptr,      # f32 flat
    invstd_ptr,        # f32 [C]
    sum_out_ptr,       # f32 [C]
    scale_grad_ptr,    # f32 [C]
    stats_ptr,         # f32 [2, C]
    BLOCK_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c_block = ct.bid(0)
    c_vec = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask_1d = c_vec < 1024

    hw_1d = ct.arange(BLOCK_HW, dtype=ct.int32)
    zero_bool_2d = ct.full((BLOCK_C, BLOCK_HW), False, dtype=ct.bool_)
    c_mask_2d = ct.reshape(c_mask_1d, (BLOCK_C, 1)) | zero_bool_2d
    hw_mask_2d = ct.reshape(hw_1d < 49, (1, BLOCK_HW)) | zero_bool_2d
    mask = c_mask_2d & hw_mask_2d
    zero_i32_2d = ct.zeros((BLOCK_C, BLOCK_HW), dtype=ct.int32)
    c_2d = ct.reshape(c_vec, (BLOCK_C, 1)) + zero_i32_2d
    hw_2d = ct.reshape(hw_1d, (1, BLOCK_HW)) + zero_i32_2d
    offsets = hw_2d * 1024 + c_2d

    sel_tile = ct.gather(partials_ptr, (offsets,), mask=mask, padding_value=ct.float32(0.0))
    dot_tile = ct.gather(partials_ptr, (offsets + 50176,), mask=mask, padding_value=ct.float32(0.0))
    sum_selected = ct.sum(sel_tile, axis=1)
    sum_dot = ct.sum(dot_tile, axis=1)

    invstd = ct.gather(invstd_ptr, (c_vec,), mask=c_mask_1d, padding_value=ct.float32(0.0))

    ct.scatter(sum_out_ptr, (c_vec,), sum_selected, mask=c_mask_1d)
    ct.scatter(scale_grad_ptr, (c_vec,), sum_dot * invstd, mask=c_mask_1d)
    ct.scatter(stats_ptr, (c_vec,), sum_selected * INV_NHW, mask=c_mask_1d)
    ct.scatter(stats_ptr, (c_vec + 1024,), (sum_dot * INV_NHW) * (invstd * invstd), mask=c_mask_1d)


@ct.kernel
def _epilogue_kernel(
    grad_ptr,     # bf16 [128, 1024]
    bn_ptr,       # bf16 flat NHWC
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    stats_ptr,    # f32 flat [2 * C]
    out_ptr,      # bf16 flat NHWC
    BLOCK: ct.Constant[int],
):
    offsets = ct.bid(0) * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = offsets < TOTAL
    c = offsets - (offsets // 1024) * 1024
    n = offsets // 50176

    bn = ct.astype(ct.gather(bn_ptr, (offsets,), mask=active, padding_value=ct.bfloat16(0.0)), ct.float32)
    mean = ct.gather(mean_ptr, (c,), mask=active, padding_value=ct.float32(0.0))
    invstd = ct.gather(invstd_ptr, (c,), mask=active, padding_value=ct.float32(0.0))
    weight = ct.gather(weight_ptr, (c,), mask=active, padding_value=ct.float32(0.0))
    bias = ct.gather(bias_ptr, (c,), mask=active, padding_value=ct.float32(0.0))

    centered = bn - mean
    affine_bf16 = ct.astype(((centered * invstd) * weight) + bias, ct.bfloat16)
    source_offsets = n * 1024 + c
    grad_val = ct.astype(ct.gather(grad_ptr, (source_offsets,), mask=active, padding_value=ct.bfloat16(0.0)), ct.float32)
    source_bf16 = ct.astype(grad_val * 0.02040816326530612, ct.bfloat16)
    zero_bf16 = ct.astype(ct.astype(source_bf16, ct.float32) * 0.0, ct.bfloat16)
    selected = ct.astype(
        ct.where(affine_bf16 <= ct.astype(ct.zeros((BLOCK,), dtype=ct.float32), ct.bfloat16),
                  zero_bf16, source_bf16),
        ct.float32,
    )

    mean_term = ct.gather(stats_ptr, (c,), mask=active, padding_value=ct.float32(0.0))
    centered_term = ct.gather(stats_ptr, (c + 1024,), mask=active, padding_value=ct.float32(0.0))
    out = ((selected - centered * centered_term) - mean_term) * (invstd * weight)
    ct.scatter(out_ptr, (offsets,), ct.astype(out, ct.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="27bd32b1")
def oracle_forward(inputs):
    grad, bn_input, mean, invstd, weight, bias, _shape0 = inputs
    full = torch.empty((), device=grad.device, dtype=torch.bfloat16)
    sum_out = torch.empty((C,), device=grad.device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=grad.device, dtype=torch.float32)
    # Dense out (NHWC layout)
    dense_nhwc = torch.empty((N, H, W, C), device=grad.device, dtype=torch.bfloat16)
    dense = dense_nhwc.permute(0, 3, 1, 2)  # (N, C, H, W) with channels-last strides
    partials = torch.empty((2, HW, C), device=grad.device, dtype=torch.float32)
    stats = torch.empty((2, C), device=grad.device, dtype=torch.float32)

    # bn_input is channels-last (NHWC in memory)
    bn_nhwc = bn_input.permute(0, 2, 3, 1)  # (N, H, W, C) with row-major strides
    bn_flat = bn_nhwc.reshape(-1)

    stream = torch.cuda.current_stream()

    BLOCK_C = 8
    ct.launch(
        stream,
        ((C + BLOCK_C - 1) // BLOCK_C, HW, 1),
        _partial_reduce_kernel,
        (
            grad.reshape(-1), bn_flat,
            mean.view(-1), invstd.view(-1), weight, bias,
            partials.view(-1),
            full.view(1),
            BLOCK_C,
        ),
    )

    ct.launch(
        stream,
        ((C + 16 - 1) // 16, 1, 1),
        _finalize_kernel,
        (
            partials.view(-1),
            invstd.view(-1),
            sum_out, scale_grad,
            stats.view(-1),
            16, 64,
        ),
    )

    BLOCK = 1024
    ct.launch(
        stream,
        ((TOTAL + BLOCK - 1) // BLOCK, 1, 1),
        _epilogue_kernel,
        (
            grad.reshape(-1), bn_flat,
            mean.view(-1), invstd.view(-1), weight, bias,
            stats.view(-1),
            dense_nhwc.view(-1),
            BLOCK,
        ),
    )

    return full, sum_out, scale_grad, dense
