"""cuTile port of sum_sum_sum_f3a5ea805a22: Blenderbot-style LN-backward + dropout
with cooperative split-K column reductions. Multi-kernel plan:
  1) row-partials kernel: per-group forward compute (sources, drop, grad),
     dense f32 add + bf16 drop side stores, per-group column partials.
  2) finalize partials: sum partials over groups to produce output column sums.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROP_SCALE = 1.1111111111111112
ROW_FACTOR = 2560.0
INV_ROW_FACTOR = 1.0 / 2560.0


@ct.kernel
def _row_partials_kernel(
    x_ptr,               # bf16 [ROWS, HIDDEN]
    weight_ptr,          # f32 [HIDDEN]
    source_bf16_ptr,     # bf16 [ROWS, HIDDEN]
    keep_ptr,            # bool [ROWS, HIDDEN]
    source_residual_ptr, # f32 [ROWS, HIDDEN]
    shift_ptr,           # f32 [ROWS]
    scale_ptr,           # f32 [ROWS]
    residual_ptr,        # f32 [ROWS, HIDDEN]
    add_out_ptr,         # f32 [ROWS, HIDDEN]
    drop_out_ptr,        # bf16 [ROWS, HIDDEN]
    partials_ptr,        # f32 [NUM_GROUPS, 3, HIDDEN] flat
    ROWS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    ROWS_PER_GROUP: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    ROW_FACTOR_: ct.Constant[float],
    INV_ROW_FACTOR_: ct.Constant[float],
    DROP_SCALE_: ct.Constant[float],
):
    group = ct.bid(0)
    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_C))

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask = cols < HIDDEN
    c_mask_2d = ct.reshape(c_mask, (1, BLOCK_C))

    acc_x_rhs = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_drop = ct.zeros((BLOCK_C,), dtype=ct.float32)

    zero_2d = ct.zeros((1, BLOCK_C), dtype=ct.float32)

    for local in range(0, ROWS_PER_GROUP, BLOCK_R):
        # BLOCK_R is 1 in all shapes.
        m = group * ROWS_PER_GROUP + local
        if m < ROWS:
            x = ct.astype(ct.load(x_ptr, index=(m, 0), shape=(1, BLOCK_C)), ct.float32)
            source_bf16 = ct.astype(ct.load(source_bf16_ptr, index=(m, 0), shape=(1, BLOCK_C)), ct.float32)
            keep = ct.astype(ct.load(keep_ptr, index=(m, 0), shape=(1, BLOCK_C)), ct.float32)
            source_residual = ct.load(source_residual_ptr, index=(m, 0), shape=(1, BLOCK_C))
            shift = ct.astype(ct.load(shift_ptr, index=(m,), shape=(1,)), ct.float32)
            scale = ct.astype(ct.load(scale_ptr, index=(m,), shape=(1,)), ct.float32)
            residual = ct.load(residual_ptr, index=(m, 0), shape=(1, BLOCK_C))

            dropped_source = source_bf16 * keep * DROP_SCALE_
            # !USE_INDUCTOR_NUMERICS branch: bf16 round-trip
            dropped_source = ct.astype(ct.astype(dropped_source, ct.bfloat16), ct.float32)
            source = source_residual + dropped_source
            rhs = (source - shift) * scale

            weighted = x * weight_2d
            row_sum_tile = ct.where(c_mask_2d, weighted, zero_2d)
            row_dot_tile = ct.where(c_mask_2d, weighted * rhs, zero_2d)
            row_sum = ct.sum(row_sum_tile)
            row_dot = ct.sum(row_dot_tile)

            centered = weighted * ROW_FACTOR_ - row_sum
            centered = centered - rhs * row_dot
            grad = scale * INV_ROW_FACTOR_ * centered
            add_val = residual + grad

            ct.store(add_out_ptr, index=(m, 0), tile=add_val)

            keep_scale = ct.astype(keep * DROP_SCALE_, ct.bfloat16)
            add_bf16 = ct.astype(add_val, ct.bfloat16)
            drop_val = ct.astype(
                ct.astype(add_bf16, ct.float32) * ct.astype(keep_scale, ct.float32),
                ct.bfloat16,
            )
            ct.store(drop_out_ptr, index=(m, 0), tile=drop_val)

            acc_x_rhs_slice = ct.reshape(ct.where(c_mask_2d, x * rhs, zero_2d), (BLOCK_C,))
            acc_x_slice = ct.reshape(ct.where(c_mask_2d, x, zero_2d), (BLOCK_C,))
            acc_drop_slice = ct.reshape(
                ct.where(c_mask_2d, ct.astype(drop_val, ct.float32), zero_2d),
                (BLOCK_C,),
            )
            acc_x_rhs = acc_x_rhs + acc_x_rhs_slice
            acc_x = acc_x + acc_x_slice
            acc_drop = acc_drop + acc_drop_slice

    # partials layout: [NUM_GROUPS, 3, HIDDEN]
    # Flat offset = group * 3 * HIDDEN + <slot> * HIDDEN + col
    partial_base = group * 3 * HIDDEN + cols
    ct.scatter(partials_ptr, (partial_base,), acc_x_rhs, mask=c_mask)
    ct.scatter(partials_ptr, (partial_base + HIDDEN,), acc_x, mask=c_mask)
    ct.scatter(partials_ptr, (partial_base + 2 * HIDDEN,), acc_drop, mask=c_mask)


@ct.kernel
def _finalize_kernel(
    partials_ptr,        # f32 flat [NG * 3 * HIDDEN]
    out_x_rhs_ptr,       # f32 [HIDDEN]
    out_x_ptr,           # f32 [HIDDEN]
    out_drop_ptr,        # f32 [HIDDEN]
    NG: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    cols_1d = ct.arange(BLOCK_C, dtype=ct.int32)
    cols = c_block * BLOCK_C + cols_1d
    c_mask = cols < HIDDEN

    groups = ct.arange(GROUP_BLOCK, dtype=ct.int32)
    g_mask = groups < NG
    mask = (ct.reshape(g_mask, (GROUP_BLOCK, 1)) | ct.full((GROUP_BLOCK, BLOCK_C), False, dtype=ct.bool_)) & \
           (ct.reshape(c_mask, (1, BLOCK_C)) | ct.full((GROUP_BLOCK, BLOCK_C), False, dtype=ct.bool_))

    # Build flat offsets: group * 3 * HIDDEN + slot * HIDDEN + col
    g_2d = ct.reshape(groups, (GROUP_BLOCK, 1))
    c_2d = ct.reshape(cols, (1, BLOCK_C))
    zero_i32_2d = ct.zeros((GROUP_BLOCK, BLOCK_C), dtype=ct.int32)
    g_full = g_2d + zero_i32_2d
    c_full = c_2d + zero_i32_2d

    x_rhs_offsets = g_full * 3 * HIDDEN + c_full
    x_offsets = g_full * 3 * HIDDEN + HIDDEN + c_full
    drop_offsets = g_full * 3 * HIDDEN + 2 * HIDDEN + c_full

    x_rhs_tile = ct.gather(partials_ptr, (x_rhs_offsets,), mask=mask, padding_value=ct.float32(0.0))
    x_tile = ct.gather(partials_ptr, (x_offsets,), mask=mask, padding_value=ct.float32(0.0))
    drop_tile = ct.gather(partials_ptr, (drop_offsets,), mask=mask, padding_value=ct.float32(0.0))

    x_rhs_sum = ct.sum(x_rhs_tile, axis=0)
    x_sum = ct.sum(x_tile, axis=0)
    drop_sum = ct.sum(drop_tile, axis=0)

    drop_sum_bf16 = ct.astype(ct.astype(drop_sum, ct.bfloat16), ct.float32)

    ct.scatter(out_x_rhs_ptr, (cols,), x_rhs_sum, mask=c_mask)
    ct.scatter(out_x_ptr, (cols,), x_sum, mask=c_mask)
    ct.scatter(out_drop_ptr, (cols,), drop_sum_bf16, mask=c_mask)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="1dd4e44b", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=4096, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="5f95f719", ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_C=4096, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="b60b3745", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="17f5e1ed", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="cfea5790", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8)
@oracle_impl(hardware="B200", point="b925f5d4", ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_C=1024, FINAL_BLOCK_C=8)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
):
    (
        x_bf16,
        weight,
        source_bf16,
        keep,
        source_residual,
        shift,
        scale,
        residual,
        full_shape_param,
        _source_shape_param,
        flat_shape_param,
        sum_shape_param,
    ) = inputs
    full_shape = _shape_tuple(full_shape_param)
    flat_shape = _shape_tuple(flat_shape_param)
    sum_shape = _shape_tuple(sum_shape_param)
    rows = int(x_bf16.shape[0])
    hidden = int(x_bf16.shape[1])

    add_out = torch.empty_strided(
        full_shape,
        (full_shape[1] * full_shape[2], full_shape[2], 1),
        device=x_bf16.device,
        dtype=torch.float32,
    )
    drop_out = torch.empty_strided(
        flat_shape,
        (flat_shape[1], 1),
        device=x_bf16.device,
        dtype=torch.bfloat16,
    )
    out_x_rhs = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_x = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)
    out_drop = torch.empty_strided(sum_shape, (1,), device=x_bf16.device, dtype=torch.float32)

    num_groups = (rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
    partials = torch.empty((num_groups, 3, hidden), device=x_bf16.device, dtype=torch.float32)

    # 2D views
    source_residual_2d = source_residual.view(rows, hidden)
    keep_2d = keep.view(rows, hidden)
    residual_2d = residual.view(rows, hidden)
    shift_1d = shift.view(rows)
    scale_1d = scale.view(rows)
    add_out_2d = add_out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, 1, 1),
        _row_partials_kernel,
        (
            x_bf16, weight, source_bf16, keep_2d, source_residual_2d,
            shift_1d, scale_1d, residual_2d,
            add_out_2d, drop_out,
            partials.view(-1),
            rows, hidden, ROWS_PER_GROUP, BLOCK_R, BLOCK_C,
            ROW_FACTOR, INV_ROW_FACTOR, DROP_SCALE,
        ),
    )

    group_block = _next_pow2(num_groups)
    ct.launch(
        stream,
        ((hidden + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _finalize_kernel,
        (
            partials.view(-1),
            out_x_rhs, out_x, out_drop,
            num_groups, hidden, group_block, FINAL_BLOCK_C,
        ),
    )

    return out_x_rhs, out_x, add_out, drop_out, drop_out.permute(1, 0), out_drop
