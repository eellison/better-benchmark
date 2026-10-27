"""cuTile port of sum_sum_868cea3d5857: EfficientNet crop + BN-affine + SiLU
gradient + BN-backward channel reductions. Multi-kernel plan:
  1) partial spatial reduce (per H*W position, sum over N per channel)
  2) finalize channel reduce (sum over HW to get [C])
  3) epilogue (dense output)
The kernel operates in channels-last (N,H,W,C) memory order via gather with
flattened indices.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BN_SCALE = 2.4912308673469386e-06


@ct.kernel
def _partial_spatial_reduce_kernel(
    crop_ptr,        # bf16 flat [N * C * IN_H * IN_W]
    activation_ptr,  # bf16 flat [N * C * H * W]
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    partials_ptr,    # f32 flat [2 * total_x] where total_x = C*H*W
    N_: ct.Constant[int],
    C_: ct.Constant[int],
    H_: ct.Constant[int],
    W_: ct.Constant[int],
    IN_H_: ct.Constant[int],
    IN_W_: ct.Constant[int],
    X_BLOCK: ct.Constant[int],
    N_BLOCK: ct.Constant[int],
):
    x_offsets = ct.bid(0) * X_BLOCK + ct.arange(X_BLOCK, dtype=ct.int32)
    n_offsets = ct.arange(N_BLOCK, dtype=ct.int32)
    total_x = C_ * H_ * W_
    x_mask = x_offsets < total_x

    # 1D per-position derivations
    c_1d = x_offsets - (x_offsets // C_) * C_
    w_1d = (x_offsets // C_) - ((x_offsets // C_) // W_) * W_
    h_1d = (x_offsets // (C_ * W_)) - ((x_offsets // (C_ * W_)) // H_) * H_

    # Broadcast to (X_BLOCK, N_BLOCK)
    x_2d_c = ct.reshape(c_1d, (X_BLOCK, 1))
    x_2d_w = ct.reshape(w_1d, (X_BLOCK, 1))
    x_2d_h = ct.reshape(h_1d, (X_BLOCK, 1))
    n_2d = ct.reshape(n_offsets, (1, N_BLOCK))

    # For broadcasting, add zeros
    zero_i32_2d = ct.zeros((X_BLOCK, N_BLOCK), dtype=ct.int32)
    c_2d = x_2d_c + zero_i32_2d
    w_2d = x_2d_w + zero_i32_2d
    h_2d = x_2d_h + zero_i32_2d
    n_2d_full = n_2d + zero_i32_2d

    x_mask_2d = ct.reshape(x_mask, (X_BLOCK, 1))
    n_mask_2d = ct.reshape(n_offsets < N_, (1, N_BLOCK))
    zero_bool_2d = ct.full((X_BLOCK, N_BLOCK), False, dtype=ct.bool_)
    mask = (x_mask_2d | zero_bool_2d) & (n_mask_2d | zero_bool_2d)

    activation_offset = n_2d_full * (C_ * H_ * W_) + h_2d * W_ * C_ + w_2d * C_ + c_2d
    crop_offset = n_2d_full * (C_ * IN_H_ * IN_W_) + (h_2d + 1) * IN_W_ * C_ + (w_2d + 1) * C_ + c_2d

    zero_f = ct.full((X_BLOCK, N_BLOCK), 0.0, dtype=ct.float32)

    activation = ct.astype(
        ct.gather(activation_ptr, (activation_offset,), mask=mask, padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    crop = ct.astype(
        ct.gather(crop_ptr, (crop_offset,), mask=mask, padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )

    # Load channel-vector inputs by gather with c_2d indices.
    mean = ct.gather(mean_ptr, (c_2d,), mask=mask, padding_value=ct.float32(0.0))
    invstd = ct.gather(invstd_ptr, (c_2d,), mask=mask, padding_value=ct.float32(0.0))
    weight = ct.gather(weight_ptr, (c_2d,), mask=mask, padding_value=ct.float32(0.0))
    bias = ct.gather(bias_ptr, (c_2d,), mask=mask, padding_value=ct.float32(0.0))

    centered = activation - mean
    affine = ((centered * invstd) * weight) + bias
    affine = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    one = ct.full((X_BLOCK, N_BLOCK), 1.0, dtype=ct.float32)
    sigmoid = 1.0 / (one + ct.exp(-affine))
    producer = (crop * sigmoid) * ((affine * (one - sigmoid)) + one)
    producer = ct.astype(ct.astype(producer, ct.bfloat16), ct.float32)
    producer = ct.where(mask, producer, zero_f)

    sum0 = ct.sum(producer, axis=1)  # (X_BLOCK,)
    sum1 = ct.sum(producer * centered, axis=1)

    ct.scatter(partials_ptr, (x_offsets,), sum0, mask=x_mask)
    x_offsets_shifted = x_offsets + total_x
    ct.scatter(partials_ptr, (x_offsets_shifted,), sum1, mask=x_mask)


@ct.kernel
def _finalize_channel_reduce_kernel(
    partials_ptr,    # f32 flat [2 * C * HW]
    stats_ptr,       # f32 [2 * C]
    sum_out_ptr,     # f32 [C]
    scaled_sum_out_ptr,  # f32 [C]
    invstd_ptr,      # f32 [C]
    C_: ct.Constant[int],
    HW_: ct.Constant[int],
    SCALE_: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    channel = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32)
    mask = offsets < HW_
    partial_base = offsets * C_ + channel

    sum0_tile = ct.gather(partials_ptr, (partial_base,), mask=mask, padding_value=ct.float32(0.0))
    sum1_offsets = C_ * HW_ + partial_base
    sum1_tile = ct.gather(partials_ptr, (sum1_offsets,), mask=mask, padding_value=ct.float32(0.0))

    sum0 = ct.sum(sum0_tile)
    sum1 = ct.sum(sum1_tile)
    invstd = ct.astype(ct.load(invstd_ptr, index=(channel,), shape=(1,)), ct.float32)
    invstd_sq = invstd * invstd

    ct.store(sum_out_ptr, index=(channel,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + sum0, (1,)))
    ct.store(scaled_sum_out_ptr, index=(channel,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + sum1 * invstd, (1,)))
    # stats is a flat [2*C] tensor
    ct.store(stats_ptr, index=(channel,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + sum0 * SCALE_, (1,)))
    ct.store(stats_ptr, index=(C_ + channel,), tile=ct.reshape(ct.full((1,), 0.0, dtype=ct.float32) + (sum1 * SCALE_) * invstd_sq, (1,)))


@ct.kernel
def _epilogue_kernel(
    crop_ptr,        # bf16 flat
    activation_ptr,  # bf16 flat
    mean_ptr,        # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    stats_ptr,       # f32 [2 * C]
    out_ptr,         # bf16 flat [N * C * H * W]
    N_: ct.Constant[int],
    C_: ct.Constant[int],
    H_: ct.Constant[int],
    W_: ct.Constant[int],
    IN_H_: ct.Constant[int],
    IN_W_: ct.Constant[int],
    NUMEL_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    offsets = ct.bid(0) * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    mask = offsets < NUMEL_

    c = offsets - (offsets // C_) * C_
    w = (offsets // C_) - ((offsets // C_) // W_) * W_
    h = (offsets // (C_ * W_)) - ((offsets // (C_ * W_)) // H_) * H_
    n = offsets // (C_ * H_ * W_)

    crop_offset = n * (C_ * IN_H_ * IN_W_) + (h + 1) * IN_W_ * C_ + (w + 1) * C_ + c

    activation = ct.astype(
        ct.gather(activation_ptr, (offsets,), mask=mask, padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    crop = ct.astype(
        ct.gather(crop_ptr, (crop_offset,), mask=mask, padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )

    mean = ct.astype(ct.gather(mean_ptr, (c,), mask=mask, padding_value=ct.float32(0.0)), ct.float32)
    invstd = ct.astype(ct.gather(invstd_ptr, (c,), mask=mask, padding_value=ct.float32(0.0)), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, (c,), mask=mask, padding_value=ct.float32(0.0)), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, (c,), mask=mask, padding_value=ct.float32(0.0)), ct.float32)
    mean_term = ct.astype(ct.gather(stats_ptr, (c,), mask=mask, padding_value=ct.float32(0.0)), ct.float32)
    centered_term = ct.astype(ct.gather(stats_ptr, (C_ + c,), mask=mask, padding_value=ct.float32(0.0)), ct.float32)

    centered = activation - mean
    affine = ((centered * invstd) * weight) + bias
    affine = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    one = ct.full((BLOCK,), 1.0, dtype=ct.float32)
    sigmoid = 1.0 / (one + ct.exp(-affine))
    producer = (crop * sigmoid) * ((affine * (one - sigmoid)) + one)
    producer = ct.astype(ct.astype(producer, ct.bfloat16), ct.float32)
    out = ((producer - (centered * centered_term)) - mean_term) * (invstd * weight)
    out_bf16 = ct.astype(out, ct.bfloat16)
    ct.scatter(out_ptr, (offsets,), out_bf16, mask=mask)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="dc48d65c", X_BLOCK=32, HW_BLOCK=4096, EPILOGUE_BLOCK=1024)
@oracle_impl(hardware="B200", point="2bf470ea", X_BLOCK=32, HW_BLOCK=256, EPILOGUE_BLOCK=1024)
def oracle_forward(inputs, *, X_BLOCK, HW_BLOCK, EPILOGUE_BLOCK):
    crop, activation, mean, invstd, weight, bias, _shape_param_0 = inputs
    n = int(activation.shape[0])
    c = int(activation.shape[1])
    h = int(activation.shape[2])
    w = int(activation.shape[3])
    in_h = int(crop.shape[2])
    in_w = int(crop.shape[3])
    numel = n * c * h * w
    total_x = c * h * w

    sum_out = torch.empty_like(weight)
    scaled_sum_out = torch.empty_like(weight)
    # Allocate a contiguous NHWC tensor; will permute to NCHW output.
    tensor_out_nhwc = torch.empty((n, h, w, c), device=activation.device, dtype=torch.bfloat16)
    tensor_out = tensor_out_nhwc.permute(0, 3, 1, 2)  # NCHW view of the NHWC buffer
    partials = torch.empty((2, total_x), device=activation.device, dtype=torch.float32)
    stats = torch.empty((2, c), device=activation.device, dtype=torch.float32)

    # Reshape mean/invstd from [1, C, 1, 1] to [C].
    mean_1d = mean.view(-1)
    invstd_1d = invstd.view(-1)

    stream = torch.cuda.current_stream()

    # activation/crop are channels-last (NHWC in memory). Get NHWC contiguous view.
    # activation.permute(0, 2, 3, 1).contiguous() would copy; instead use the
    # underlying storage via a permute + view.
    activation_nhwc = activation.permute(0, 2, 3, 1)  # (N, H, W, C) with strides matching row-major
    crop_nhwc = crop.permute(0, 2, 3, 1)              # (N, IN_H, IN_W, C)
    activation_flat = activation_nhwc.reshape(-1)
    crop_flat = crop_nhwc.reshape(-1)

    # Kernel 1: partial spatial reduce
    x_grid = (total_x + X_BLOCK - 1) // X_BLOCK
    ct.launch(
        stream,
        (x_grid, 1, 1),
        _partial_spatial_reduce_kernel,
        (
            crop_flat, activation_flat,
            mean_1d, invstd_1d, weight, bias,
            partials.view(-1),
            n, c, h, w, in_h, in_w,
            X_BLOCK, n,
        ),
    )

    # Kernel 2: finalize channel reduce. HW may be non-pow2 (e.g. 56*56=3136),
    # so round BLOCK up to next pow2 and mask.
    hw = h * w
    hw_pad = _next_pow2(hw)
    ct.launch(
        stream,
        (c, 1, 1),
        _finalize_channel_reduce_kernel,
        (
            partials.view(-1), stats.view(-1),
            sum_out, scaled_sum_out,
            invstd_1d,
            c, hw, BN_SCALE,
            hw_pad,
        ),
    )

    # Kernel 3: epilogue
    epilogue_grid = (numel + EPILOGUE_BLOCK - 1) // EPILOGUE_BLOCK
    ct.launch(
        stream,
        (epilogue_grid, 1, 1),
        _epilogue_kernel,
        (
            crop_flat, activation_flat,
            mean_1d, invstd_1d, weight, bias,
            stats.view(-1),
            tensor_out_nhwc.view(-1),
            n, c, h, w, in_h, in_w,
            numel, EPILOGUE_BLOCK,
        ),
    )

    return sum_out, scaled_sum_out, tensor_out
