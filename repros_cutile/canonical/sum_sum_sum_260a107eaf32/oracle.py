"""cuTile port of sum_sum_sum_260a107eaf32: BART LayerNorm-backward + dropout.

Two-kernel structure mirroring the Triton oracle:
  1. _row_partial_kernel  (grid=(num_groups,)): for each group loops over
     ROWS_PER_GROUP rows (BLOCK_R=1), performs the LN-backward pointwise
     math + per-row reductions inside the kernel, stores grad (f32) and
     returned_drop (bf16), and accumulates 3 column partials (x*rhs, x,
     exact_drop-as-f32) for that group.
  2. _finalize_partials_kernel (grid=(HIDDEN/FINAL_BLOCK_H,)): reduces
     the 3 column partial tensors across groups; the drop reduction gets
     a bf16 round-trip at the end.

Shape hashes handled: HIDDEN in {256, 768, 1024, 1536}. For non-power-of-2
HIDDEN (768, 1536), the kernel writes to a padded scratch of width BLOCK_H
and the results are narrow-copied afterwards (cuTile stores cannot mask).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROW_FACTOR = 1024.0
RETURN_DROP_SCALE = 1.1111111111111112
EXACT_DROP_SCALE = 1.109375


@ct.kernel
def _row_partial_kernel(
    x_bf16_ptr,        # bf16 [rows, HIDDEN]
    residual_ptr,      # f32  [rows, HIDDEN]
    weight_ptr,        # f32  [HIDDEN]
    rhs_ptr,           # f32  [rows, HIDDEN]
    row_scale_ptr,     # f32  [rows]
    keep_ptr,          # bool [rows, HIDDEN]
    grad_out_ptr,      # f32  [rows, BLOCK_H]           (padded scratch)
    drop_out_ptr,      # bf16 [rows, BLOCK_H]           (padded scratch)
    partials_ptr,      # f32  [num_groups, 3, BLOCK_H]  (padded scratch)
    ROWS_PER_GROUP: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    group = ct.bid(0)

    # Weight loaded once per group. Cols beyond HIDDEN pad to zero.
    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))

    acc_x_rhs = ct.zeros((BLOCK_H,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_H,), dtype=ct.float32)
    acc_drop = ct.zeros((BLOCK_H,), dtype=ct.float32)

    zero_2d = ct.zeros((1, BLOCK_H), dtype=ct.float32)

    for local in ct.static_iter(range(ROWS_PER_GROUP)):
        row = group * ROWS_PER_GROUP + local

        # All 2D loads pad OOB cols with zero (BLOCK_H >= HIDDEN).
        x_bf = ct.load(
            x_bf16_ptr, index=(row, 0), shape=(1, BLOCK_H),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_bf_f = ct.astype(x_bf, ct.float32)
        residual = ct.astype(
            ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        x = x_bf_f + residual
        rhs = ct.astype(
            ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_H),
                    padding_mode=ct.PaddingMode.ZERO),
            ct.float32,
        )
        row_scale_val = ct.astype(
            ct.load(row_scale_ptr, index=(row,), shape=(1,)),
            ct.float32,
        )
        row_scale_s = ct.reshape(row_scale_val, ())
        keep = ct.load(
            keep_ptr, index=(row, 0), shape=(1, BLOCK_H),
            padding_mode=ct.PaddingMode.ZERO,
        )

        weighted = x * weight_2d
        # Padded cols have x=weight=rhs=0, so sums are exact over HIDDEN cols.
        row_sum = ct.sum(weighted, axis=1, keepdims=True)         # (1, 1)
        row_dot = ct.sum(weighted * rhs, axis=1, keepdims=True)   # (1, 1)

        centered = weighted * ROW_FACTOR - row_sum - rhs * row_dot
        grad = row_scale_s * centered                             # (1, BLOCK_H)

        returned_drop_bf = ct.astype(
            ct.where(keep, grad * RETURN_DROP_SCALE, zero_2d), ct.bfloat16,
        )
        grad_bf_f = ct.astype(ct.astype(grad, ct.bfloat16), ct.float32)
        exact_drop_bf = ct.astype(
            ct.where(keep, grad_bf_f * EXACT_DROP_SCALE, zero_2d), ct.bfloat16,
        )
        exact_drop_f = ct.astype(exact_drop_bf, ct.float32)

        ct.store(grad_out_ptr, index=(row, 0), tile=grad)
        ct.store(drop_out_ptr, index=(row, 0), tile=returned_drop_bf)

        # Accumulate column partials (pad cols contribute 0).
        acc_x_rhs = acc_x_rhs + ct.reshape(x * rhs, (BLOCK_H,))
        acc_x = acc_x + ct.reshape(x, (BLOCK_H,))
        acc_drop = acc_drop + ct.reshape(exact_drop_f, (BLOCK_H,))

    ct.store(partials_ptr, index=(group, 0, 0),
             tile=ct.reshape(acc_x_rhs, (1, 1, BLOCK_H)))
    ct.store(partials_ptr, index=(group, 1, 0),
             tile=ct.reshape(acc_x, (1, 1, BLOCK_H)))
    ct.store(partials_ptr, index=(group, 2, 0),
             tile=ct.reshape(acc_drop, (1, 1, BLOCK_H)))


@ct.kernel
def _finalize_partials_kernel(
    partials_ptr,      # f32 [num_groups, 3, BLOCK_H]
    sum_x_rhs_ptr,     # f32 [HIDDEN]
    sum_x_ptr,         # f32 [HIDDEN]
    sum_drop_ptr,      # f32 [HIDDEN]
    GROUP_BLOCK: ct.Constant[int],
    FINAL_BLOCK_H: ct.Constant[int],
):
    col_tile = ct.bid(0)

    x_rhs_t = ct.load(
        partials_ptr, index=(0, 0, col_tile),
        shape=(GROUP_BLOCK, 1, FINAL_BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_rhs_sum = ct.sum(x_rhs_t, axis=0)  # (1, FINAL_BLOCK_H)
    ct.store(sum_x_rhs_ptr, index=(col_tile,),
             tile=ct.reshape(x_rhs_sum, (FINAL_BLOCK_H,)))

    x_t = ct.load(
        partials_ptr, index=(0, 1, col_tile),
        shape=(GROUP_BLOCK, 1, FINAL_BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_sum = ct.sum(x_t, axis=0)
    ct.store(sum_x_ptr, index=(col_tile,),
             tile=ct.reshape(x_sum, (FINAL_BLOCK_H,)))

    drop_t = ct.load(
        partials_ptr, index=(0, 2, col_tile),
        shape=(GROUP_BLOCK, 1, FINAL_BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    drop_sum = ct.sum(drop_t, axis=0)
    drop_sum_bf = ct.astype(drop_sum, ct.bfloat16)
    drop_sum_f = ct.astype(drop_sum_bf, ct.float32)
    ct.store(sum_drop_ptr, index=(col_tile,),
             tile=ct.reshape(drop_sum_f, (FINAL_BLOCK_H,)))


@oracle_impl(hardware="B200", point="540cb101",
             ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8)
@oracle_impl(hardware="B200", point="75b78b2c",
             ROWS_PER_GROUP=8, BLOCK_R=1, BLOCK_H=2048, FINAL_BLOCK_H=8)
@oracle_impl(hardware="B200", point="9846b7f2",
             ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8)
@oracle_impl(hardware="B200", point="f51dabe2",
             ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=256, FINAL_BLOCK_H=8)
@oracle_impl(hardware="B200", point="102da4dd",
             ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8)
@oracle_impl(hardware="B200", point="7f0c11fc",
             ROWS_PER_GROUP=16, BLOCK_R=1, BLOCK_H=1024, FINAL_BLOCK_H=8)
def oracle_forward(inputs, *, ROWS_PER_GROUP, BLOCK_R, BLOCK_H, FINAL_BLOCK_H):
    del BLOCK_R  # always 1 in Triton; hard-coded here via static_iter step 1
    (
        arg0_1,   # bf16 [rows, HIDDEN]
        arg1_1,   # f32  [B, S, HIDDEN]
        arg2_1,   # f32  [HIDDEN]  weight
        arg3_1,   # f32  [B, S, HIDDEN]  rhs
        arg4_1,   # f32  [B, S, 1]  scale
        arg5_1,   # b8   [B, S, HIDDEN]  keep
        shape_view_3d, shape_flat, shape_hidden,
    ) = inputs
    del shape_hidden
    device = arg0_1.device
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    view_3d = tuple(int(d) for d in shape_view_3d)
    flat_shape = tuple(int(d) for d in shape_flat)
    B, S, H = view_3d
    assert H == hidden
    assert rows == B * S

    num_groups = rows // ROWS_PER_GROUP

    # Padded scratch outputs (cuTile stores write full tile, no masking).
    grad_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    drop_padded = torch.empty((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
    partials = torch.empty(
        (num_groups, 3, BLOCK_H), device=device, dtype=torch.float32,
    )

    sum_x_rhs = torch.empty((hidden,), device=device, dtype=torch.float32)
    sum_x = torch.empty((hidden,), device=device, dtype=torch.float32)
    sum_drop = torch.empty((hidden,), device=device, dtype=torch.float32)

    x_bf16_2d = arg0_1  # already (rows, hidden), bf16
    residual_2d = arg1_1.reshape(rows, hidden)
    rhs_2d = arg3_1.reshape(rows, hidden)
    row_scale_1d = arg4_1.reshape(rows)
    keep_2d = arg5_1.reshape(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_groups, 1, 1), _row_partial_kernel,
        (x_bf16_2d, residual_2d, arg2_1, rhs_2d, row_scale_1d, keep_2d,
         grad_padded, drop_padded, partials,
         ROWS_PER_GROUP, BLOCK_H),
    )

    # num_groups is a power of 2 for every registered shape hash.
    ct.launch(
        stream, (hidden // FINAL_BLOCK_H, 1, 1), _finalize_partials_kernel,
        (partials, sum_x_rhs, sum_x, sum_drop,
         num_groups, FINAL_BLOCK_H),
    )

    grad_out = torch.empty_strided(
        view_3d, (view_3d[1] * view_3d[2], view_3d[2], 1),
        device=device, dtype=torch.float32,
    )
    grad_out.copy_(grad_padded[:, :hidden].view(view_3d))

    drop_out = torch.empty_strided(
        flat_shape, (hidden, 1), device=device, dtype=torch.bfloat16,
    )
    drop_out.copy_(drop_padded[:, :hidden].view(flat_shape))

    return grad_out, sum_x_rhs, sum_x, drop_out, drop_out.permute(1, 0), sum_drop
