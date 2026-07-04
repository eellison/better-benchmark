"""cuTile port of var_mean_var_mean_e56fc242c590 (SCHEDULER_FUSION): dual
GroupNorm affine + ReLU.

Total groups = batch * num_groups; each group has GROUP_ELEMS elements.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _dual_groupnorm_affine_relu_kernel(
    x0_ptr,       # bf16 [TOTAL_GROUPS * GROUP_ELEMS]
    x1_ptr,       # bf16 [TOTAL_GROUPS * GROUP_ELEMS]
    weight0_ptr,  # bf16 [CHANNELS]
    bias0_ptr,    # bf16 [CHANNELS]
    weight1_ptr,  # bf16 [CHANNELS]
    bias1_ptr,    # bf16 [CHANNELS]
    out_ptr,      # bf16 [TOTAL_GROUPS * GROUP_ELEMS]
    TOTAL_GROUPS: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    HW_SIZE: ct.Constant[int],
    GROUP_ELEMS: ct.Constant[int],
    NUM_GROUPS: ct.Constant[int],
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
    x0_bf = ct.gather(x0_ptr, safe)
    x1_bf = ct.gather(x1_ptr, safe)
    x0 = ct.astype(x0_bf, ct.float32)
    x1 = ct.astype(x1_bf, ct.float32)

    zero_f = ct.full((BLOCK_M, BLOCK_K), 0.0, dtype=ct.float32)
    x0m = ct.where(mask, x0, zero_f)
    x1m = ct.where(mask, x1, zero_f)
    inv_count = 1.0 / float(GROUP_ELEMS)
    mean0 = ct.sum(x0m, axis=1) * inv_count
    mean1 = ct.sum(x1m, axis=1) * inv_count
    mean0_2d = ct.reshape(mean0, (BLOCK_M, 1))
    mean1_2d = ct.reshape(mean1, (BLOCK_M, 1))
    mean0_broad = ct.broadcast_to(mean0_2d, (BLOCK_M, BLOCK_K))
    mean1_broad = ct.broadcast_to(mean1_2d, (BLOCK_M, BLOCK_K))
    centered0 = x0 - mean0_broad
    centered1 = x1 - mean1_broad
    centered0_m = ct.where(mask, centered0, zero_f)
    centered1_m = ct.where(mask, centered1, zero_f)
    var0 = ct.sum(centered0_m * centered0_m, axis=1) * inv_count
    var1 = ct.sum(centered1_m * centered1_m, axis=1) * inv_count
    inv0 = ct.rsqrt(var0 + EPS)
    inv1 = ct.rsqrt(var1 + EPS)
    inv0_2d = ct.reshape(inv0, (BLOCK_M, 1))
    inv1_2d = ct.reshape(inv1, (BLOCK_M, 1))
    inv0_broad = ct.broadcast_to(inv0_2d, (BLOCK_M, BLOCK_K))
    inv1_broad = ct.broadcast_to(inv1_2d, (BLOCK_M, BLOCK_K))
    norm0 = centered0 * inv0_broad
    norm1 = centered1 * inv1_broad

    group_id = rows - (rows // NUM_GROUPS) * NUM_GROUPS
    channels_per_group = CHANNELS // NUM_GROUPS
    group_id_2d = ct.reshape(group_id, (BLOCK_M, 1))
    group_id_broad = ct.broadcast_to(group_id_2d, (BLOCK_M, BLOCK_K))
    channel = group_id_broad * channels_per_group + elems_broad // HW_SIZE
    safe_ch = ct.where(mask, channel, zero64)
    w0_bf = ct.gather(weight0_ptr, safe_ch)
    b0_bf = ct.gather(bias0_ptr, safe_ch)
    w1_bf = ct.gather(weight1_ptr, safe_ch)
    b1_bf = ct.gather(bias1_ptr, safe_ch)
    w0 = ct.astype(w0_bf, ct.float32)
    b0 = ct.astype(b0_bf, ct.float32)
    w1 = ct.astype(w1_bf, ct.float32)
    b1 = ct.astype(b1_bf, ct.float32)

    branch0 = ct.astype(norm0 * w0 + b0, ct.bfloat16)
    branch1 = ct.astype(norm1 * w1 + b1, ct.bfloat16)
    summed = ct.astype(ct.astype(branch0, ct.float32)
                        + ct.astype(branch1, ct.float32), ct.bfloat16)
    summed_f = ct.astype(summed, ct.float32)
    relu = ct.where(summed_f > 0.0, summed_f, zero_f)
    ct.scatter(out_ptr, safe, ct.astype(relu, ct.bfloat16), mask=mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="8e339e83", BLOCK_M=16, BLOCK_K=16)
@oracle_impl(hardware="B200", point="d4aaa8d6", BLOCK_M=8, BLOCK_K=32)
@oracle_impl(hardware="B200", point="a1bd90d2", BLOCK_M=4, BLOCK_K=64)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_K: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, shape0, _shape1, shape2, _shape3 = inputs
    group_shape = _as_shape(shape0)
    out_shape = _as_shape(shape2)
    batch = int(out_shape[0])
    channels = int(out_shape[1])
    height = int(out_shape[2])
    width = int(out_shape[3])
    num_groups = int(group_shape[1])
    group_elems = int(group_shape[2]) * int(group_shape[3])
    total_groups = batch * num_groups
    hw_size = height * width

    out = torch.empty(out_shape, device=arg0_1.device, dtype=torch.bfloat16)
    x0_flat = arg0_1.reshape(-1)
    x1_flat = arg1_1.reshape(-1)
    out_flat = out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((total_groups + BLOCK_M - 1) // BLOCK_M, 1, 1),
        _dual_groupnorm_affine_relu_kernel,
        (x0_flat, x1_flat, arg2_1, arg3_1, arg4_1, arg5_1, out_flat,
         total_groups, channels, hw_size, group_elems, num_groups, BLOCK_M, BLOCK_K),
    )
    return out
