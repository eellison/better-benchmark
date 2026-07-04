"""cuTile port of sum_sum_sum_b6027582bc37: T5/MT5 dual RMSNorm-backward
and vocabulary-gradient masked scatter-reduce.

For each of the two branches, per row:
- Load bf16 mm0/mm1/mm2 [rows, hidden] and sum in f32.
- Compute RMSNorm-backward "grad" using shared saved/rstd/prior + weight.
- Atomic-add to shared hidden-dim reduction (out0/out1).
- Atomic-add "grad" to (vocab, hidden) at row's token index (with masked
  bounds).

The vocab output is seeded from arg0_1 (bf16) cast to f32 before both
branches accumulate on top.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _init_vocab_kernel(
    base_ptr,       # bf16 (VOCAB, HIDDEN)
    out_vocab_ptr,  # f32 (VOCAB, HIDDEN)
    TOTAL: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    active = offs < TOTAL
    # Convert 1D offs to (row, col) 2D index.
    row = offs // HIDDEN
    col = offs - row * HIDDEN
    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    row_safe = ct.where(active, row, zero_i)
    col_safe = ct.where(active, col, zero_i)
    values_bf = ct.gather(base_ptr, (row_safe, col_safe), mask=active)
    values_f = ct.astype(values_bf, ct.float32)
    ct.scatter(out_vocab_ptr, (row_safe, col_safe), values_f, mask=active)


@ct.kernel
def _branch_scatter_reduce_kernel(
    mm0_ptr,        # bf16 (ROWS, HIDDEN)
    mm1_ptr,        # bf16 (ROWS, HIDDEN)
    mm2_ptr,        # bf16 (ROWS, HIDDEN)
    weight_ptr,     # f32 (HIDDEN,)
    keep_ptr,       # b8 (ROWS, HIDDEN)
    saved_ptr,      # f32 (ROWS, HIDDEN)
    rstd_ptr,       # f32 (ROWS,)
    prior_ptr,      # f32 (ROWS, HIDDEN)
    index_ptr,      # i64 (ROWS,)
    out_reduce_ptr, # f32 (HIDDEN,)
    out_vocab_ptr,  # f32 (VOCAB, HIDDEN)
    ROWS_: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    VOCAB_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    h_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_active = h_idx < HIDDEN_

    mm0_bf = ct.load(mm0_ptr, index=(row, 0), shape=(1, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)
    mm1_bf = ct.load(mm1_ptr, index=(row, 0), shape=(1, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)
    mm2_bf = ct.load(mm2_ptr, index=(row, 0), shape=(1, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    keep = ct.load(keep_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    saved = ct.load(saved_ptr, index=(row, 0), shape=(1, BLOCK_H),
                    padding_mode=ct.PaddingMode.ZERO)
    prior = ct.load(prior_ptr, index=(row, 0), shape=(1, BLOCK_H),
                    padding_mode=ct.PaddingMode.ZERO)
    rstd = ct.load(rstd_ptr, index=(row,), shape=(1,))

    added_2d = (
        ct.astype(mm0_bf, ct.float32)
        + ct.astype(mm1_bf, ct.float32)
        + ct.astype(mm2_bf, ct.float32)
    )
    added = ct.reshape(added_2d, (BLOCK_H,))
    keep_f = ct.reshape(ct.astype(keep, ct.float32), (BLOCK_H,))
    saved_1d = ct.reshape(saved, (BLOCK_H,))
    prior_1d = ct.reshape(prior, (BLOCK_H,))

    kept_saved = keep_f * saved_1d * DROPOUT_SCALE
    dropped_rstd = kept_saved * rstd

    zero_f_h = ct.full((BLOCK_H,), 0.0, dtype=ct.float32)
    # atomic_add: sum of added * dropped_rstd
    ct.atomic_add(
        out_reduce_ptr,
        (h_idx,),
        ct.where(col_active, added * dropped_rstd, zero_f_h),
    )

    weighted = added * weight
    row_dot = ct.sum(
        ct.where(col_active, weighted * kept_saved, zero_f_h)
    )
    prior_plus = prior_1d + weighted * rstd
    rstd_cubed = rstd * rstd * rstd
    correction = ((row_dot * -0.5) * rstd_cubed / HIDDEN_) * (kept_saved * 2.0)
    grad = (prior_plus + correction) * (keep_f * DROPOUT_SCALE)

    raw_index = ct.load(index_ptr, index=(row,), shape=(1,))
    zero_i64 = ct.full((1,), 0, dtype=ct.int64)
    vocab_i64 = ct.full((1,), VOCAB_, dtype=ct.int64)
    minus_one_i64 = ct.full((1,), -1, dtype=ct.int64)
    valid = (raw_index >= zero_i64) & (raw_index < vocab_i64) & (raw_index != minus_one_i64)
    safe_index = ct.where(valid, raw_index, vocab_i64)

    zero_i32 = ct.zeros((BLOCK_H,), dtype=ct.int32)
    row_bcast = ct.astype(safe_index, ct.int32) + zero_i32
    grad_masked = ct.where(col_active, grad, zero_f_h)
    ct.atomic_add(out_vocab_ptr, (row_bcast, h_idx), grad_masked)


@oracle_impl(hardware="B200", point="4e691609", INIT_BLOCK=1024, BLOCK_H=512,
             num_warps_init=4, num_warps_branch=8)
@oracle_impl(hardware="B200", point="0c10a950", INIT_BLOCK=1024, BLOCK_H=512,
             num_warps_init=4, num_warps_branch=8)
def oracle_forward(
    inputs,
    *,
    INIT_BLOCK: int,
    BLOCK_H: int,
    num_warps_init: int,
    num_warps_branch: int,
):
    del num_warps_init, num_warps_branch
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
        arg12_1,
        arg13_1,
        arg14_1,
        arg15_1,
        arg16_1,
        *_shape_params,
    ) = inputs

    vocab = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    rows = int(arg1_1.shape[0])
    total = vocab * hidden

    device = arg0_1.device
    out0 = torch.zeros((hidden,), device=device, dtype=torch.float32)
    out1 = torch.zeros((hidden,), device=device, dtype=torch.float32)
    out_vocab = torch.empty((vocab, hidden), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    init_grid = ((total + INIT_BLOCK - 1) // INIT_BLOCK, 1, 1)
    ct.launch(
        stream,
        init_grid,
        _init_vocab_kernel,
        (arg0_1, out_vocab, total, hidden, INIT_BLOCK),
    )

    # Flatten [B, T, H] to [ROWS, H] for kernels.
    keep_2d0 = arg5_1.view(rows, hidden)
    saved_2d = arg6_1.view(rows, hidden)
    rstd_1d0 = arg7_1.view(rows)
    prior_2d0 = arg8_1.view(rows, hidden)
    index_1d = arg9_1.view(rows)

    keep_2d1 = arg14_1.view(rows, hidden)
    # saved_2d is arg6_1 shared
    rstd_1d1 = arg15_1.view(rows)
    prior_2d1 = arg16_1.view(rows, hidden)

    ct.launch(
        stream,
        (rows, 1, 1),
        _branch_scatter_reduce_kernel,
        (arg1_1, arg2_1, arg3_1, arg4_1, keep_2d0, saved_2d, rstd_1d0, prior_2d0,
         index_1d, out0, out_vocab, rows, hidden, vocab, BLOCK_H),
    )
    ct.launch(
        stream,
        (rows, 1, 1),
        _branch_scatter_reduce_kernel,
        (arg10_1, arg11_1, arg12_1, arg13_1, keep_2d1, saved_2d, rstd_1d1, prior_2d1,
         index_1d, out1, out_vocab, rows, hidden, vocab, BLOCK_H),
    )
    return out0, out1, out_vocab
