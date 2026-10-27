"""cuTile port of sum_sum_sum_4cd246989343: Google FNet LN-backward with dropout.

Mirrors Triton's 2 kernels:
  1. _row_partials_kernel: row-tiled LN-bwd producer that also emits
     per-group partial column sums for the three column reductions
     (out_add_rhs, out_add, out_masked).
  2. _finalize_partials_kernel: reduces per-group column partials to (HIDDEN,).

HIDDEN=768 is not power of 2; padded to BLOCK_H=1024 with masks.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
BLOCK_H = 1024
DROP_SCALE = 1.1111111111111112
ROW_FACTOR = 768.0
ROWS_PER_GROUP = 16


def _next_p2(n):
    v = 1
    while v < int(n):
        v <<= 1
    return v


@ct.kernel
def _row_partials_kernel(
    real_pair_ptr,     # f32 (ROWS, BLOCK_H, 2) padded
    residual_ptr,      # f32 (ROWS, BLOCK_H) padded
    weight_ptr,        # f32 (BLOCK_H,) padded
    rhs_ptr,           # f32 (ROWS, BLOCK_H) padded
    row_scale_ptr,     # f32 (ROWS,)
    keep_ptr,          # b8  (ROWS, BLOCK_H) padded
    grad_out_ptr,      # f32 (ROWS, BLOCK_H) padded
    masked_out_ptr,    # f32 (ROWS, BLOCK_H) padded
    partials_ptr,      # f32 (num_groups, 3, BLOCK_H)  three column partials
    HIDDEN_C: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
    ROWS_PER_GROUP_C: ct.Constant[int],
    ROW_FACTOR_C: ct.Constant[float],
    DROP_SCALE_C: ct.Constant[float],
):
    group = ct.bid(0)
    weight_1d = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_C,))
    weight_2d = ct.reshape(weight_1d, (1, BLOCK_H_C))

    col_idx = ct.arange(BLOCK_H_C, dtype=ct.int32)
    col_valid_1d = col_idx < HIDDEN_C
    col_valid = ct.reshape(col_valid_1d, (1, BLOCK_H_C))

    zero_row = ct.zeros((1, BLOCK_H_C), dtype=ct.float32)
    acc_add_rhs = zero_row
    acc_add = zero_row
    acc_masked = zero_row

    for local in range(ROWS_PER_GROUP_C):
        row = group * ROWS_PER_GROUP_C + local

        # real_pair[row, :, 0] via 3D shape.
        selected_3d = ct.load(
            real_pair_ptr, index=(row, 0, 0), shape=(1, BLOCK_H_C, 1),
        )
        selected = ct.reshape(selected_3d, (1, BLOCK_H_C))

        residual = ct.load(residual_ptr, index=(row, 0),
                           shape=(1, BLOCK_H_C))
        rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_H_C))
        keep = ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_H_C))
        row_scale_1d = ct.load(row_scale_ptr, index=(row,), shape=(1,))
        row_scale = ct.reshape(row_scale_1d, (1, 1))

        add_value = residual + selected
        weighted = add_value * weight_2d
        weighted_masked = ct.where(col_valid, weighted, zero_row)
        row_sum = ct.sum(weighted_masked, axis=1, keepdims=True)
        weighted_rhs = weighted * rhs
        weighted_rhs_masked = ct.where(col_valid, weighted_rhs, zero_row)
        row_dot = ct.sum(weighted_rhs_masked, axis=1, keepdims=True)

        centered = weighted * ROW_FACTOR_C - row_sum - rhs * row_dot
        grad = row_scale * centered
        keep_f = ct.astype(keep, ct.float32)
        keep_scaled = keep_f * DROP_SCALE_C
        masked_grad = grad * keep_scaled

        # Zero out OOB columns in outputs (matches original semantics).
        add_masked_out = ct.where(col_valid, add_value, zero_row)
        grad_masked_out = ct.where(col_valid, grad, zero_row)
        masked_masked_out = ct.where(col_valid, masked_grad, zero_row)

        ct.store(grad_out_ptr, index=(row, 0), tile=grad_masked_out)
        ct.store(masked_out_ptr, index=(row, 0), tile=masked_masked_out)

        # Column-sum contributions (masked to valid columns).
        acc_add_rhs = acc_add_rhs + ct.where(col_valid, add_value * rhs, zero_row)
        acc_add = acc_add + add_masked_out
        acc_masked = acc_masked + masked_masked_out

    ct.store(partials_ptr, index=(group, 0, 0),
             tile=ct.reshape(acc_add_rhs, (1, 1, BLOCK_H_C)))
    ct.store(partials_ptr, index=(group, 1, 0),
             tile=ct.reshape(acc_add, (1, 1, BLOCK_H_C)))
    ct.store(partials_ptr, index=(group, 2, 0),
             tile=ct.reshape(acc_masked, (1, 1, BLOCK_H_C)))


@ct.kernel
def _finalize_partials_kernel(
    partials_ptr,          # f32 (num_groups, 3, BLOCK_H)
    out_add_rhs_ptr,       # f32 (BLOCK_H,) padded
    out_add_ptr,           # f32 (BLOCK_H,) padded
    out_masked_ptr,        # f32 (BLOCK_H,) padded
    NUM_GROUPS: ct.Constant[int],
    GROUP_BLOCK: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
):
    p_add_rhs = ct.load(
        partials_ptr, index=(0, 0, 0),
        shape=(GROUP_BLOCK, 1, BLOCK_H_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    p_add = ct.load(
        partials_ptr, index=(0, 1, 0),
        shape=(GROUP_BLOCK, 1, BLOCK_H_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    p_masked = ct.load(
        partials_ptr, index=(0, 2, 0),
        shape=(GROUP_BLOCK, 1, BLOCK_H_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    a2 = ct.reshape(p_add_rhs, (GROUP_BLOCK, BLOCK_H_C))
    b2 = ct.reshape(p_add, (GROUP_BLOCK, BLOCK_H_C))
    c2 = ct.reshape(p_masked, (GROUP_BLOCK, BLOCK_H_C))
    ct.store(out_add_rhs_ptr, index=(0,), tile=ct.sum(a2, axis=0))
    ct.store(out_add_ptr, index=(0,), tile=ct.sum(b2, axis=0))
    ct.store(out_masked_ptr, index=(0,), tile=ct.sum(c2, axis=0))


@oracle_impl(hardware="B200", point="3847e61c")
def oracle_forward(inputs):
    (
        real_pair, residual, weight, rhs, row_scale, keep,
        _flat_shape, _sum_shape,
    ) = inputs
    device = residual.device

    # Pad outputs to BLOCK_H so cuTile stores are aligned.
    add_out_padded = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    grad_out_padded = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    masked_out_padded = torch.empty((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    del add_out_padded  # not stored; we accumulate via ct.sum on-the-fly.

    # Pad inputs to BLOCK_H.
    real_pair_flat = real_pair.contiguous().view(ROWS, HIDDEN, 2)
    real_pair_padded = torch.zeros((ROWS, BLOCK_H, 2), device=device, dtype=torch.float32)
    real_pair_padded[:, :HIDDEN, :] = real_pair_flat

    residual_2d = residual.contiguous().view(ROWS, HIDDEN)
    residual_padded = torch.zeros((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    residual_padded[:, :HIDDEN] = residual_2d

    weight_padded = torch.zeros((BLOCK_H,), device=device, dtype=torch.float32)
    weight_padded[:HIDDEN] = weight

    rhs_2d = rhs.contiguous().view(ROWS, HIDDEN)
    rhs_padded = torch.zeros((ROWS, BLOCK_H), device=device, dtype=torch.float32)
    rhs_padded[:, :HIDDEN] = rhs_2d

    row_scale_1d = row_scale.contiguous().view(ROWS)

    keep_2d = keep.contiguous().view(ROWS, HIDDEN)
    keep_padded = torch.zeros((ROWS, BLOCK_H), device=device, dtype=torch.bool)
    keep_padded[:, :HIDDEN] = keep_2d

    num_groups = ROWS // ROWS_PER_GROUP  # exact: 16384 / 16 = 1024
    partials = torch.empty(
        (num_groups, 3, BLOCK_H), device=device, dtype=torch.float32,
    )
    out_add_rhs_padded = torch.empty((BLOCK_H,), device=device, dtype=torch.float32)
    out_add_padded = torch.empty((BLOCK_H,), device=device, dtype=torch.float32)
    out_masked_padded = torch.empty((BLOCK_H,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_groups, 1, 1), _row_partials_kernel,
        (real_pair_padded, residual_padded, weight_padded, rhs_padded,
         row_scale_1d, keep_padded,
         grad_out_padded, masked_out_padded, partials,
         HIDDEN, BLOCK_H, ROWS_PER_GROUP, ROW_FACTOR, DROP_SCALE),
    )
    group_block = _next_p2(num_groups)
    ct.launch(
        stream, (1, 1, 1), _finalize_partials_kernel,
        (partials, out_add_rhs_padded, out_add_padded, out_masked_padded,
         num_groups, group_block, BLOCK_H),
    )

    grad_out_2d = grad_out_padded[:, :HIDDEN]
    masked_out = masked_out_padded[:, :HIDDEN].contiguous()
    grad_out_bsh = grad_out_2d.contiguous().view(BATCH, SEQ, HIDDEN)
    out_add_rhs = out_add_rhs_padded[:HIDDEN].contiguous()
    out_add = out_add_padded[:HIDDEN].contiguous()
    out_masked = out_masked_padded[:HIDDEN].contiguous()

    return (
        grad_out_bsh, out_add_rhs, out_add, masked_out,
        masked_out.permute(1, 0), out_masked,
    )
