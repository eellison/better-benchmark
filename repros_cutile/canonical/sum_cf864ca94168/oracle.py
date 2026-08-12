"""cuTile port of sum_cf864ca94168: max-pool backward + where + channel sum.

Uses torch scatter_add + inductor_prims low_memory_max_pool_offsets_to_indices
for the reverse-scatter (a cuTile-hostile atomic-add op). Then a cuTile
partial-reduce kernel applies the where(mask, scalar, x) AND accumulates the
per-channel sum, followed by a cuTile final-reduce kernel. This mirrors
Triton's `_pool_scatter_where_partials_kernel` + `_final_sum_kernel` split
for the where + reduction stages.
"""

import torch
import torch._inductor.inductor_prims  # noqa: F401
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _where_partial_reduce_kernel(
    mask_ptr,        # b8 flat [R*C]
    source_ptr,      # bf16 flat [R*C]
    scalar_ptr,      # bf16 [1]
    where_out_ptr,   # bf16 flat [R*C]
    partial_ptr,     # f32 [NUM_GROUPS, C] flat
    sum_ptr,         # f32 [C]
    R_: ct.Constant[int],
    C_: ct.Constant[int],
    GROUP_R: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    STORE_DIRECT_SUM: ct.Constant[bool],
):
    group = ct.bid(0)
    c_block = ct.bid(1)

    cols = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    col_valid = cols < C_
    scalar = ct.load(scalar_ptr, index=(0,), shape=(1,))

    acc_sum = ct.zeros((BLOCK_C,), dtype=ct.float32)

    # Iterate over BLOCK_R rows at a time up to GROUP_R
    num_inner = GROUP_R // BLOCK_R
    for inner in range(num_inner):
        rows = group * GROUP_R + inner * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
        row_valid = rows < R_
        rows_2d = ct.reshape(rows, (BLOCK_R, 1))
        cols_2d = ct.reshape(cols, (1, BLOCK_C))
        offsets = rows_2d * C_ + cols_2d
        row_valid_2d = ct.reshape(row_valid, (BLOCK_R, 1))
        col_valid_2d = ct.reshape(col_valid, (1, BLOCK_C))
        active = row_valid_2d & col_valid_2d

        mask_vals = ct.gather(mask_ptr, offsets, mask=active)
        source_vals = ct.gather(source_ptr, offsets, mask=active)

        # scalar broadcast to (BLOCK_R, BLOCK_C)
        scalar_bcast = ct.reshape(scalar, (1, 1)) + ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.bfloat16)
        selected = ct.where(mask_vals, scalar_bcast, source_vals)

        ct.scatter(where_out_ptr, offsets, selected, mask=active)

        selected_f = ct.astype(selected, ct.float32)
        selected_f_masked = ct.where(active, selected_f, 0.0)
        acc_sum = acc_sum + ct.sum(selected_f_masked, axis=0)

    if STORE_DIRECT_SUM:
        rounded = ct.astype(ct.astype(acc_sum, ct.bfloat16), ct.float32)
        ct.scatter(sum_ptr, cols, rounded, mask=col_valid)
    else:
        # Store partials at partial[group, cols]
        partial_offsets = group * C_ + cols
        ct.scatter(partial_ptr, partial_offsets, acc_sum, mask=col_valid)


@ct.kernel
def _final_sum_kernel(
    partial_ptr,     # f32 [NUM_GROUPS, C] flat
    sum_ptr,         # f32 [C]
    NUM_GROUPS: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_GROUPS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    groups = ct.arange(BLOCK_GROUPS, dtype=ct.int32)
    cols = c_block * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    group_valid = ct.reshape(groups < NUM_GROUPS, (BLOCK_GROUPS, 1))
    col_valid = ct.reshape(cols < C_, (1, BLOCK_C))
    mask = group_valid & col_valid
    groups_2d = ct.reshape(groups, (BLOCK_GROUPS, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_C))
    offsets = groups_2d * C_ + cols_2d

    partials = ct.gather(partial_ptr, offsets, mask=mask)
    partials_f = ct.astype(partials, ct.float32)
    partials_masked = ct.where(mask, partials_f, 0.0)
    total = ct.sum(partials_masked, axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.scatter(sum_ptr, cols, rounded, mask=cols < C_)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_pow2(x):
    return 1 << (int(x) - 1).bit_length()


@oracle_impl(hardware="B200", point="94a62ed8", KERNEL=3, GROUP_R=128, BLOCK_R=16, BLOCK_C=16, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="e579fe5c", KERNEL=3, GROUP_R=128, BLOCK_R=16, BLOCK_C=16, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="a00ef5f5", KERNEL=3, GROUP_R=128, BLOCK_R=16, BLOCK_C=16, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="b721890d", KERNEL=2, GROUP_R=1024, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=8, CONST_OFFSET_3=True)
@oracle_impl(hardware="B200", point="c0a53b72", KERNEL=2, GROUP_R=1024, BLOCK_R=64, BLOCK_C=16, FINAL_BLOCK_C=8, CONST_OFFSET_3=True)
@oracle_impl(hardware="B200", point="b3ea8ee2", KERNEL=2, GROUP_R=1024, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, CONST_OFFSET_3=True)
@oracle_impl(hardware="B200", point="bf7c8e5d", KERNEL=2, GROUP_R=512, BLOCK_R=64, BLOCK_C=32, FINAL_BLOCK_C=16, CONST_OFFSET_3=True)
def oracle_forward(inputs, *, KERNEL, GROUP_R, BLOCK_R, BLOCK_C, FINAL_BLOCK_C, CONST_OFFSET_3=False):
    grad, offsets, mask, scalar, *shape_params = inputs
    device = grad.device

    full_shape = _shape_tuple(shape_params[0])
    view_shape_grad = _shape_tuple(shape_params[1])
    kernel_size = _shape_tuple(shape_params[2])
    pool_out_hw = _shape_tuple(shape_params[3])
    pool_stride = _shape_tuple(shape_params[4])
    view_shape_indices = _shape_tuple(shape_params[5])
    out_shape = _shape_tuple(shape_params[6])

    # Reverse-scatter path (same as eager Repro).
    full = torch.zeros(full_shape, device=device, dtype=torch.float32)
    grad_c = grad.contiguous()
    grad_flat = grad_c.view(view_shape_grad)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        offsets, kernel_size, pool_out_hw, list(pool_stride), [0, 0], [1, 1],
    )
    indices_c = indices.contiguous()
    indices_flat = indices_c.view(view_shape_indices)
    grad_f32 = grad_flat.to(torch.float32)
    scatter = torch.scatter_add(full, 1, indices_flat, grad_f32)
    scatter_view = scatter.view(out_shape)
    scatter_bf = scatter_view.to(torch.bfloat16)

    n = out_shape[0]
    c = out_shape[1]
    ih = out_shape[2]
    iw = out_shape[3]
    r = n * ih * iw

    # Prepare (R, C) contiguous flat views. Since mask has channels-last stride,
    # the physical layout is (N, IH, IW, C) contiguous. We build scatter in
    # that same NHWC layout.
    scatter_hwc = scatter_bf.permute(0, 2, 3, 1).contiguous()  # (N, IH, IW, C)
    # mask channels-last: view its storage as NHWC.
    mask_nhwc = mask.as_strided((n, ih, iw, c), (c * ih * iw, iw * c, c, 1))

    scalar_flat = scalar.view(1)
    where_out_hwc = torch.empty((n, ih, iw, c), device=device, dtype=torch.bfloat16)

    num_groups = (r + GROUP_R - 1) // GROUP_R
    direct_sum = num_groups == 1
    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    if direct_sum:
        partial = sum_out
    else:
        partial = torch.empty((num_groups, c), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, ct.cdiv(c, BLOCK_C), 1),
        _where_partial_reduce_kernel,
        (mask_nhwc.contiguous().view(-1), scatter_hwc.view(-1), scalar_flat,
         where_out_hwc.view(-1), partial.view(-1), sum_out,
         r, c, GROUP_R, BLOCK_R, BLOCK_C, direct_sum),
    )

    if not direct_sum:
        block_groups = _next_pow2(num_groups)
        ct.launch(
            stream,
            (ct.cdiv(c, FINAL_BLOCK_C), 1, 1),
            _final_sum_kernel,
            (partial.view(-1), sum_out, num_groups, c, block_groups, FINAL_BLOCK_C),
        )

    # Reshape where_out_hwc as (N, C, IH, IW) with channels-last stride.
    where_out_strided = where_out_hwc.as_strided(
        (n, c, ih, iw),
        (c * ih * iw, 1, iw * c, c),
    )
    return where_out_strided, sum_out
