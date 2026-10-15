"""cuTile port of sum_sum_e62c5b3ab99a: DenseNet bf16 BN-backward tail with
slice-add residual over the last 32 channels. Single per-channel kernel:
compute sum_where, sum_where*centered, dense gradient, and slice residual add.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 352
H = 28
W = 28
HW = H * W  # 784
R = N * HW  # 3136
SLICE_START = 320
SLICE_C = 32
SCALE = 0.00031887755102040814


@ct.kernel
def _tail_kernel(
    r0_ptr,           # bf16 [N * 512 * H * W] flat
    r1_ptr,           # bf16 [N * 480 * H * W] flat
    r2_ptr,           # bf16 [N * 448 * H * W] flat
    r3_ptr,           # bf16 [N * 416 * H * W] flat
    r4_ptr,           # bf16 [N * 384 * H * W] flat
    mask_input_ptr,   # bf16 [N * C * H * W] flat
    fill_ptr,         # bf16 [1]
    source_ptr,       # bf16 [N * C * H * W] flat
    centered_source_ptr,  # bf16 [N * C * H * W] flat
    mean_ptr,         # f32 [C]
    invstd_ptr,       # f32 [C]
    weight_ptr,       # f32 [C]
    sum_out_ptr,      # f32 [C]
    scale_grad_ptr,   # f32 [C]
    dense_out_ptr,    # bf16 [N * C * H * W] flat
    add_out_ptr,      # bf16 [N * SLICE_C * H * W] flat
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    R_: ct.Constant[int],
    SLICE_START_: ct.Constant[int],
    SLICE_C_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)  # channel

    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R_
    n_idx = rows // HW_
    spatial = rows - n_idx * HW_

    dense_flat_offsets = n_idx * (C_ * HW_) + c * HW_ + spatial

    zero_bf = ct.full((BLOCK_R,), 0.0, dtype=ct.bfloat16)
    zero_f = ct.full((BLOCK_R,), 0.0, dtype=ct.float32)

    mask_value = ct.gather(mask_input_ptr, (dense_flat_offsets,),
                            mask=active,
                            padding_value=ct.bfloat16(0.0))
    fill_value = ct.load(fill_ptr, index=(0,), shape=(1,))
    # Broadcast fill (bf16, shape (1,)) to (BLOCK_R,) via addition to zero tile
    fill_broadcast = zero_bf + fill_value
    source_value = ct.gather(source_ptr, (dense_flat_offsets,),
                              mask=active,
                              padding_value=ct.bfloat16(0.0))

    where_bf16 = ct.where(mask_value <= zero_bf, fill_broadcast, source_value)
    where_f32 = ct.where(active, ct.astype(where_bf16, ct.float32), zero_f)

    centered_source = ct.gather(centered_source_ptr, (dense_flat_offsets,),
                                  mask=active,
                                  padding_value=ct.bfloat16(0.0))
    centered_source_f = ct.astype(centered_source, ct.float32)
    # mean is 1D [C]
    mean_val = ct.load(mean_ptr, index=(c,), shape=(1,))
    mean_broadcast = zero_f + ct.astype(mean_val, ct.float32)
    centered = ct.where(active, centered_source_f - mean_broadcast, zero_f)

    product = where_f32 * centered
    sum_where = ct.sum(where_f32)
    sum_centered = ct.sum(product)

    invstd_v = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_v = ct.load(weight_ptr, index=(c,), shape=(1,))
    invstd_f = ct.astype(invstd_v, ct.float32)
    weight_f = ct.astype(weight_v, ct.float32)

    mean_term = sum_where * SCALE_
    prod_scaled = sum_centered * SCALE_
    invstd_sq = invstd_f * invstd_f
    variance_term = prod_scaled * invstd_sq
    output_scale = invstd_f * weight_f

    variance_bcast = zero_f + variance_term
    mean_term_bcast = zero_f + mean_term
    output_scale_bcast = zero_f + output_scale

    corrected = where_f32 - (centered * variance_bcast)
    centered_grad = corrected - mean_term_bcast
    dense_bf16 = ct.astype(centered_grad * output_scale_bcast, ct.bfloat16)

    # Scalar reductions to per-channel outputs: shape (1,)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + sum_where, (1,)))
    ct.store(scale_grad_ptr, index=(c,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + sum_centered * invstd_f, (1,)))

    ct.scatter(dense_out_ptr, (dense_flat_offsets,), dense_bf16, mask=active)

    # Slice add: only for c >= SLICE_START
    slice_c = c - SLICE_START_
    add_offsets = n_idx * (SLICE_C_ * HW_) + slice_c * HW_ + spatial
    in_slice_scalar = c >= SLICE_START_
    in_slice_bcast = (ct.full((BLOCK_R,), False, dtype=ct.bool_)) | in_slice_scalar
    add_mask = active & in_slice_bcast

    v0_off = n_idx * (512 * HW_) + c * HW_ + spatial
    v1_off = n_idx * (480 * HW_) + c * HW_ + spatial
    v2_off = n_idx * (448 * HW_) + c * HW_ + spatial
    v3_off = n_idx * (416 * HW_) + c * HW_ + spatial
    v4_off = n_idx * (384 * HW_) + c * HW_ + spatial

    v0 = ct.gather(r0_ptr, (v0_off,), mask=add_mask, padding_value=ct.bfloat16(0.0))
    v1 = ct.gather(r1_ptr, (v1_off,), mask=add_mask, padding_value=ct.bfloat16(0.0))
    v2 = ct.gather(r2_ptr, (v2_off,), mask=add_mask, padding_value=ct.bfloat16(0.0))
    v3 = ct.gather(r3_ptr, (v3_off,), mask=add_mask, padding_value=ct.bfloat16(0.0))
    v4 = ct.gather(r4_ptr, (v4_off,), mask=add_mask, padding_value=ct.bfloat16(0.0))

    def _bf16_add(a, b):
        return ct.astype(ct.astype(a, ct.float32) + ct.astype(b, ct.float32), ct.bfloat16)

    residual = _bf16_add(v0, v1)
    residual = _bf16_add(residual, v2)
    residual = _bf16_add(residual, v3)
    residual = _bf16_add(residual, v4)
    add_value = _bf16_add(residual, dense_bf16)

    ct.scatter(add_out_ptr, (add_offsets,), add_value, mask=add_mask)


@oracle_impl(
    hardware="B200",
    point="914547b5",
    BLOCK_R=4096,
)
def oracle_forward(inputs, *, BLOCK_R: int):
    (
        r0,
        r1,
        r2,
        r3,
        r4,
        mask_input,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
    ) = inputs

    sum_out = torch.empty_strided((C,), (1,), device=source.device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=source.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=source.device,
        dtype=torch.bfloat16,
    )
    add_out = torch.empty_strided(
        (N, SLICE_C, H, W),
        (SLICE_C * HW, HW, W, 1),
        device=source.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    # Flatten all tensors to 1D for gather.
    fill_view = fill.view(1)
    ct.launch(
        stream,
        (C, 1, 1),
        _tail_kernel,
        (
            r0.reshape(-1), r1.reshape(-1), r2.reshape(-1), r3.reshape(-1), r4.reshape(-1),
            mask_input.reshape(-1), fill_view,
            source.reshape(-1), centered_source.reshape(-1),
            mean.view(-1), invstd, weight,
            sum_out, scale_grad,
            dense_out.view(-1), add_out.view(-1),
            C, HW, R,
            SLICE_START, SLICE_C, SCALE,
            BLOCK_R,
        ),
    )
    return sum_out, scale_grad, dense_out, add_out
