"""cuTile port of sum_sum_sum_d0cb44a92f6d: BERT-family LN-backward + dropout
column reductions. Two-kernel plan:
  1) row-group kernel: emits per-row grad/mask outputs and per-group column
     partials.
  2) finalize: reduce partials across groups.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROP_SCALE = 1.1111111111111112


@ct.kernel
def _row_group_kernel(
    x_ptr,           # bf16 [M, C]
    weight_ptr,      # f32 [C]
    rhs_ptr,         # f32 [M, C]
    row_scale_ptr,   # f32 [M]
    keep_ptr,        # bool [M, C]
    grad_ptr,        # f32 [M, C]
    masked_ptr,      # bf16 [M, C]
    partial_x_rhs_ptr,   # f32 [num_row_groups, C]
    partial_x_ptr,       # f32 [num_row_groups, C]
    partial_masked_ptr,  # f32 [num_row_groups, C]
    M: ct.Constant[int],
    C: ct.Constant[int],
    NORM_SIZE: ct.Constant[int],
    DROP_SCALE_: ct.Constant[float],
    ROW_GROUP: ct.Constant[int],
    XBLOCK: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_group = ct.bid(0)
    # Load weight (1D, size C, no padding needed since we mask below)
    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_C))

    c_1d = ct.arange(BLOCK_C, dtype=ct.int32)
    c_mask = c_1d < C
    c_mask_2d = ct.reshape(c_mask, (1, BLOCK_C))

    acc_x_rhs = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_masked = ct.zeros((BLOCK_C,), dtype=ct.float32)

    for local_start in range(0, ROW_GROUP, XBLOCK):
        # XBLOCK is 1 in all decorators - so this is just row-by-row
        m = row_group * ROW_GROUP + local_start
        # We use m as scalar since XBLOCK==1; simpler code.
        if m < M:
            x = ct.astype(ct.load(x_ptr, index=(m, 0), shape=(1, BLOCK_C)), ct.float32)
            rhs = ct.astype(ct.load(rhs_ptr, index=(m, 0), shape=(1, BLOCK_C)), ct.float32)
            row_scale = ct.astype(ct.load(row_scale_ptr, index=(m,), shape=(1,)), ct.float32)

            weighted = x * weight_2d
            weighted_rhs = weighted * rhs
            zero_2d = ct.zeros((1, BLOCK_C), dtype=ct.float32)
            weighted_masked = ct.where(c_mask_2d, weighted, zero_2d)
            weighted_rhs_masked = ct.where(c_mask_2d, weighted_rhs, zero_2d)
            row_sum = ct.sum(weighted_masked)
            row_dot = ct.sum(weighted_rhs_masked)
            scaled = weighted * float(NORM_SIZE)
            sub = scaled - row_sum
            rhs_row_dot = rhs * row_dot
            sub_1 = sub - rhs_row_dot
            grad = row_scale * sub_1
            ct.store(grad_ptr, index=(m, 0), tile=grad)

            keep = ct.astype(ct.load(keep_ptr, index=(m, 0), shape=(1, BLOCK_C)), ct.float32)
            keep_scale_bf16 = ct.astype(keep * DROP_SCALE_, ct.bfloat16)
            grad_bf16 = ct.astype(grad, ct.bfloat16)
            reduced_masked_bf16 = ct.astype(
                ct.astype(grad_bf16, ct.float32) * ct.astype(keep_scale_bf16, ct.float32),
                ct.bfloat16,
            )
            stored_masked_bf16 = ct.astype(
                grad * ct.astype(keep_scale_bf16, ct.float32),
                ct.bfloat16,
            )
            ct.store(masked_ptr, index=(m, 0), tile=stored_masked_bf16)

            x_rhs = x * rhs
            x_rhs_1d = ct.reshape(ct.where(c_mask_2d, x_rhs, zero_2d), (BLOCK_C,))
            x_1d = ct.reshape(ct.where(c_mask_2d, x, zero_2d), (BLOCK_C,))
            masked_1d = ct.reshape(
                ct.where(c_mask_2d, ct.astype(reduced_masked_bf16, ct.float32), zero_2d),
                (BLOCK_C,),
            )
            acc_x_rhs = acc_x_rhs + x_rhs_1d
            acc_x = acc_x + x_1d
            acc_masked = acc_masked + masked_1d

    ct.store(partial_x_rhs_ptr, index=(row_group, 0), tile=ct.reshape(acc_x_rhs, (1, BLOCK_C)))
    ct.store(partial_x_ptr, index=(row_group, 0), tile=ct.reshape(acc_x, (1, BLOCK_C)))
    ct.store(partial_masked_ptr, index=(row_group, 0), tile=ct.reshape(acc_masked, (1, BLOCK_C)))


@ct.kernel
def _finalize_kernel(
    partial_x_rhs_ptr,   # f32 [NG, C]
    partial_x_ptr,       # f32 [NG, C]
    partial_masked_ptr,  # f32 [NG, C]
    out_x_rhs_ptr,       # f32 [C]
    out_x_ptr,           # f32 [C]
    out_masked_ptr,      # f32 [C]
    NG: ct.Constant[int],
    C: ct.Constant[int],
    BLOCK_GROUPS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    c_arange = ct.arange(BLOCK_C, dtype=ct.int32)
    c_offsets = col_block * BLOCK_C + c_arange
    c_mask = c_offsets < C

    acc_x_rhs = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_x = ct.zeros((BLOCK_C,), dtype=ct.float32)
    acc_masked = ct.zeros((BLOCK_C,), dtype=ct.float32)

    zero_1d = ct.zeros((BLOCK_C,), dtype=ct.float32)

    # BLOCK_GROUPS is guaranteed >= NG for our cases.
    for g_start in range(0, NG, BLOCK_GROUPS):
        # Only if BLOCK_GROUPS > NG, else this loops once with the whole tile
        pass
    # Simpler: single sweep with BLOCK_GROUPS covering all NG rows.
    groups = ct.arange(BLOCK_GROUPS, dtype=ct.int32)
    valid_groups = groups < NG
    # Load partials as tile (BLOCK_GROUPS, BLOCK_C) and sum along groups.
    # Need to load a 2D tile from a 2D array [NG, C], starting at (0, col_block).
    # BLOCK_GROUPS may exceed NG; use padding ZERO.
    x_rhs_tile = ct.load(
        partial_x_rhs_ptr,
        index=(0, col_block),
        shape=(BLOCK_GROUPS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_tile = ct.load(
        partial_x_ptr,
        index=(0, col_block),
        shape=(BLOCK_GROUPS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    masked_tile = ct.load(
        partial_masked_ptr,
        index=(0, col_block),
        shape=(BLOCK_GROUPS, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # Also need mask on cols since BLOCK_C may exceed C - use ct.where.
    c_mask_2d = ct.reshape(c_mask, (1, BLOCK_C))
    zero_2d = ct.zeros((BLOCK_GROUPS, BLOCK_C), dtype=ct.float32)
    x_rhs_tile = ct.where(c_mask_2d, x_rhs_tile, zero_2d)
    x_tile = ct.where(c_mask_2d, x_tile, zero_2d)
    masked_tile = ct.where(c_mask_2d, masked_tile, zero_2d)

    acc_x_rhs = ct.sum(x_rhs_tile, axis=0)
    acc_x = ct.sum(x_tile, axis=0)
    acc_masked = ct.sum(masked_tile, axis=0)

    # bf16 round on masked before final store (matches Triton: to bf16 then to f32)
    acc_masked_bf16 = ct.astype(ct.astype(acc_masked, ct.bfloat16), ct.float32)

    # Store per-column (may be partial due to c_mask; store as-is since the tail
    # elements are past C — write with scatter using mask.)
    ct.scatter(out_x_rhs_ptr, (c_offsets,), acc_x_rhs, mask=c_mask)
    ct.scatter(out_x_ptr, (c_offsets,), acc_x, mask=c_mask)
    ct.scatter(out_masked_ptr, (c_offsets,), acc_masked_bf16, mask=c_mask)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="09b68f1c", ROW_GROUP=8, XBLOCK=1, BLOCK_C=2048, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="3fb352aa", ROW_GROUP=16, XBLOCK=1, BLOCK_C=1024, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="dca7eb09", ROW_GROUP=16, XBLOCK=1, BLOCK_C=1024, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="8fb12abb", ROW_GROUP=32, XBLOCK=1, BLOCK_C=256, FINAL_BLOCK_C=16)
@oracle_impl(hardware="B200", point="a1f869c6", ROW_GROUP=16, XBLOCK=1, BLOCK_C=1024, FINAL_BLOCK_C=16)
def oracle_forward(
    inputs,
    *,
    ROW_GROUP: int,
    XBLOCK: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    batch = int(arg2_1.shape[0])
    seq = int(arg2_1.shape[1])
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    num_row_groups = (rows + ROW_GROUP - 1) // ROW_GROUP

    grad = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    masked = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty(
        (3, num_row_groups, hidden),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    # arg2_1 is [batch, seq, hidden] f32; view as [rows, hidden]
    rhs_2d = arg2_1.view(rows, hidden)
    # arg3_1 is [batch, seq, 1] f32; view as [rows]
    row_scale_1d = arg3_1.view(rows)
    # arg4_1 is [batch, seq, hidden] bool; view as [rows, hidden]
    keep_2d = arg4_1.view(rows, hidden)
    # grad is [batch, seq, hidden] f32; view as [rows, hidden]
    grad_2d = grad.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_groups, 1, 1),
        _row_group_kernel,
        (
            arg0_1, arg1_1, rhs_2d, row_scale_1d, keep_2d,
            grad_2d, masked,
            partials[0], partials[1], partials[2],
            rows, hidden, 1536, DROP_SCALE,
            ROW_GROUP, XBLOCK, BLOCK_C,
        ),
    )

    reductions = torch.empty((3, hidden), device=arg0_1.device, dtype=torch.float32)
    # BLOCK_GROUPS must cover NG (num_row_groups) as pow2.
    block_groups = _next_pow2(num_row_groups)
    ct.launch(
        stream,
        ((hidden + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _finalize_kernel,
        (
            partials[0], partials[1], partials[2],
            reductions[0], reductions[1], reductions[2],
            num_row_groups, hidden, block_groups, FINAL_BLOCK_C,
        ),
    )

    return (
        grad,
        reductions[0],
        reductions[1],
        masked,
        masked.permute(1, 0),
        reductions[2],
    )
