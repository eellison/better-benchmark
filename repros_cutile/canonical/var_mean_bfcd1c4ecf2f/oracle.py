"""cuTile port of var_mean_bfcd1c4ecf2f (NEW_PATTERN): GroupNorm residual ReLU.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _groupnorm_residual_relu_kernel(
    x_ptr,           # bf16 [TOTAL_GROUPS * GROUP_ELEMS]
    weight_ptr,      # bf16 [CHANNELS]
    bias_ptr,        # bf16 [CHANNELS]
    residual_ptr,    # bf16 [TOTAL_GROUPS * GROUP_ELEMS]
    mean_ptr,        # f32 [TOTAL_GROUPS]
    rsqrt_ptr,       # f32 [TOTAL_GROUPS]
    relu_ptr,        # f32 [TOTAL_GROUPS * GROUP_ELEMS]
    bf16_ptr,        # bf16 [TOTAL_GROUPS * GROUP_ELEMS]
    mask_ptr,        # bool [TOTAL_GROUPS * GROUP_ELEMS]
    TOTAL_GROUPS: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    HW: ct.Constant[int],
    GROUP_ELEMS: ct.Constant[int],
    GROUPS_C: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    row_block = ct.bid(0)
    rows = ct.arange(BLOCK_M, dtype=ct.int64) + row_block * BLOCK_M
    elems = ct.arange(BLOCK_K, dtype=ct.int64)
    rows_2d = ct.reshape(rows, (BLOCK_M, 1))
    elems_2d = ct.reshape(elems, (1, BLOCK_K))
    rows_broad = ct.broadcast_to(rows_2d, (BLOCK_M, BLOCK_K))
    elems_broad = ct.broadcast_to(elems_2d, (BLOCK_M, BLOCK_K))
    row_mask = rows < TOTAL_GROUPS
    row_mask_2d = ct.reshape(row_mask, (BLOCK_M, 1))
    row_mask_broad = ct.broadcast_to(row_mask_2d, (BLOCK_M, BLOCK_K))
    elem_mask = elems_broad < GROUP_ELEMS
    mask = row_mask_broad & elem_mask

    offsets = rows_broad * GROUP_ELEMS + elems_broad
    zero64 = ct.zeros((BLOCK_M, BLOCK_K), dtype=ct.int64)
    safe = ct.where(mask, offsets, zero64)
    x_bf = ct.gather(x_ptr, safe)
    x = ct.astype(x_bf, ct.float32)

    zero_f = ct.full((BLOCK_M, BLOCK_K), 0.0, dtype=ct.float32)
    x_masked = ct.where(mask, x, zero_f)
    inv_count = 1.0 / float(GROUP_ELEMS)
    mean = ct.sum(x_masked, axis=1) * inv_count
    mean_2d = ct.reshape(mean, (BLOCK_M, 1))
    mean_broad = ct.broadcast_to(mean_2d, (BLOCK_M, BLOCK_K))
    centered = x - mean_broad
    centered_m = ct.where(mask, centered, zero_f)
    variance = ct.sum(centered_m * centered_m, axis=1) * inv_count
    invstd = ct.rsqrt(variance + EPS)
    invstd_2d = ct.reshape(invstd, (BLOCK_M, 1))
    invstd_broad = ct.broadcast_to(invstd_2d, (BLOCK_M, BLOCK_K))
    normalized = centered * invstd_broad

    channels_per_group = CHANNELS // GROUPS_C
    group = rows - (rows // GROUPS_C) * GROUPS_C
    group_2d = ct.reshape(group, (BLOCK_M, 1))
    group_broad = ct.broadcast_to(group_2d, (BLOCK_M, BLOCK_K))
    channel = group_broad * channels_per_group + elems_broad // HW
    safe_ch = ct.where(mask, channel, zero64)
    weight_bf = ct.gather(weight_ptr, safe_ch)
    bias_bf = ct.gather(bias_ptr, safe_ch)
    weight = ct.astype(weight_bf, ct.float32)
    bias = ct.astype(bias_bf, ct.float32)

    residual_bf = ct.gather(residual_ptr, safe)
    residual = ct.astype(residual_bf, ct.float32)

    affine = normalized * weight + bias
    summed = affine + residual
    is_nan = summed != summed
    positive = summed > 0.0
    relu = ct.where(positive | is_nan, summed, zero_f)

    # Store side outputs
    row_1d_zero = ct.zeros((BLOCK_M,), dtype=ct.int64)
    safe_rows = ct.where(row_mask, rows, row_1d_zero)
    ct.scatter(mean_ptr, safe_rows, mean, mask=row_mask)
    ct.scatter(rsqrt_ptr, safe_rows, invstd, mask=row_mask)
    ct.scatter(relu_ptr, safe, relu, mask=mask)
    ct.scatter(bf16_ptr, safe, ct.astype(relu, ct.bfloat16), mask=mask)
    # relu <= 0.0
    le_val = relu <= 0.0
    ct.scatter(mask_ptr, safe, le_val, mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="a488e261", BLOCK_M=4, BLOCK_K=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_K: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    group_shape = _as_shape(shape0)
    out_shape = _as_shape(shape1)
    batch = int(arg0_1.shape[0])
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    groups = int(group_shape[1])
    group_elems = int(group_shape[2]) * int(group_shape[3])
    total_groups = batch * groups

    mean = torch.empty_strided((batch, groups), (groups, 1),
                                device=arg0_1.device, dtype=torch.float32)
    rsqrt = torch.empty_strided((batch, groups), (groups, 1),
                                 device=arg0_1.device, dtype=torch.float32)
    relu = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                device=arg0_1.device, dtype=torch.float32)
    bf16 = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                                device=arg0_1.device, dtype=torch.bfloat16)
    le = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                              device=arg0_1.device, dtype=torch.bool)

    x_flat = arg0_1.reshape(-1)
    residual_flat = arg3_1.reshape(-1)
    mean_flat = mean.reshape(-1)
    rsqrt_flat = rsqrt.reshape(-1)
    relu_flat = relu.reshape(-1)
    bf16_flat = bf16.reshape(-1)
    le_flat = le.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((total_groups + BLOCK_M - 1) // BLOCK_M, 1, 1),
        _groupnorm_residual_relu_kernel,
        (x_flat, arg1_1, arg2_1, residual_flat, mean_flat, rsqrt_flat,
         relu_flat, bf16_flat, le_flat,
         total_groups, channels, hw, group_elems, groups, BLOCK_M, BLOCK_K),
    )
    return mean, rsqrt, relu, bf16, le
