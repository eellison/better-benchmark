"""cuTile port of sum_sum_sum_0b5187c8ce05: GPT-J LayerNorm-backward with
embedding-gradient scatter.

Per row (1 * 128 rows), per column-chunk of hidden=4096, compute the
LayerNorm-backward "correction" using row-level reductions, atomic-add to
two hidden-dim reduction outputs, and atomic-add to (VOCAB=50400, HIDDEN)
at the row's token index.

Two-pass structure via three cuTile launches:
1. Zero-fill scatter output.
2. Row partials of (row_sum = sum(mul), row_dot = sum(mul * centered_scaled)).
3. Scatter + column reductions using the row summaries.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 4096
VOCAB = 50400


@ct.kernel
def _zero_kernel(
    out_ptr,   # f32 (VOCAB, HIDDEN)
    TOTAL: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    active = offs < TOTAL
    row = offs // HIDDEN_C
    col = offs - row * HIDDEN_C
    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    row_safe = ct.where(active, row, zero_i)
    col_safe = ct.where(active, col, zero_i)
    zeros = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    ct.scatter(out_ptr, (row_safe, col_safe), zeros, mask=active)


@ct.kernel
def _row_reduce_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,  # bf16 (ROWS, HIDDEN)
    weight_ptr,     # f32 (HIDDEN,)
    src_ptr,        # f32 (ROWS, HIDDEN)
    mean_ptr,       # f32 (ROWS,)
    scale_ptr,      # f32 (ROWS,)
    row_sum_ptr,    # f32 (ROWS,)
    row_dot_ptr,    # f32 (ROWS,)
    HIDDEN_C: ct.Constant[int],
):
    row = ct.bid(0)
    a_bf = ct.load(a_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    b_bf = ct.load(b_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    c_bf = ct.load(c_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    d_bf = ct.load(d_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,))
    src = ct.load(src_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    mean = ct.load(mean_ptr, index=(row,), shape=(1,))
    scale = ct.load(scale_ptr, index=(row,), shape=(1,))

    summed = (
        (ct.astype(a_bf, ct.float32) + ct.astype(b_bf, ct.float32))
        + ct.astype(c_bf, ct.float32)
    ) + ct.astype(d_bf, ct.float32)
    weight_2d = ct.reshape(weight, (1, HIDDEN_C))
    weighted = summed * weight_2d
    centered_scaled = (src - mean) * scale

    row_sum = ct.sum(weighted)
    row_dot = ct.sum(weighted * centered_scaled)
    row_sum_1 = ct.full((1,), row_sum, dtype=ct.float32)
    row_dot_1 = ct.full((1,), row_dot, dtype=ct.float32)
    ct.store(row_sum_ptr, index=(row,), tile=row_sum_1)
    ct.store(row_dot_ptr, index=(row,), tile=row_dot_1)


@ct.kernel
def _scatter_and_col_reduce_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,  # bf16 (ROWS, HIDDEN)
    weight_ptr,     # f32 (HIDDEN,)
    src_ptr,        # f32 (ROWS, HIDDEN)
    mean_ptr,       # f32 (ROWS,)
    scale_ptr,      # f32 (ROWS,)
    residual_ptr,   # f32 (ROWS, HIDDEN)
    index_ptr,      # i64 (ROWS,)
    row_sum_ptr,    # f32 (ROWS,)
    row_dot_ptr,    # f32 (ROWS,)
    sum_prod_out_ptr,  # f32 (HIDDEN,)
    sum_x_out_ptr,     # f32 (HIDDEN,)
    scatter_out_ptr,   # f32 (VOCAB, HIDDEN)
    HIDDEN_C: ct.Constant[int],
    VOCAB_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    INV_HIDDEN: ct.Constant[float],
):
    row = ct.bid(0)
    chunk = ct.bid(1)
    h_idx = ct.arange(BLOCK_H, dtype=ct.int32) + chunk * BLOCK_H
    col_active = h_idx < HIDDEN_C

    # cuTile index is in TILE space: element offset = index[i] * shape[i].
    a_bf = ct.load(a_ptr, index=(row, chunk), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    b_bf = ct.load(b_ptr, index=(row, chunk), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    c_bf = ct.load(c_ptr, index=(row, chunk), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    d_bf = ct.load(d_ptr, index=(row, chunk), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(chunk,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    src = ct.load(src_ptr, index=(row, chunk), shape=(1, BLOCK_H),
                  padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, chunk), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(row,), shape=(1,))
    scale = ct.load(scale_ptr, index=(row,), shape=(1,))

    summed_2d = (
        (ct.astype(a_bf, ct.float32) + ct.astype(b_bf, ct.float32))
        + ct.astype(c_bf, ct.float32)
    ) + ct.astype(d_bf, ct.float32)
    summed = ct.reshape(summed_2d, (BLOCK_H,))

    src_1d = ct.reshape(src, (BLOCK_H,))
    residual_1d = ct.reshape(residual, (BLOCK_H,))
    centered_scaled = (src_1d - mean) * scale
    weighted = summed * weight

    row_sum = ct.load(row_sum_ptr, index=(row,), shape=(1,))
    row_dot = ct.load(row_dot_ptr, index=(row,), shape=(1,))

    zero_f_h = ct.full((BLOCK_H,), 0.0, dtype=ct.float32)
    # atomic-add to sum_prod: sum over rows of summed * centered_scaled
    ct.atomic_add(
        sum_prod_out_ptr,
        (h_idx,),
        ct.where(col_active, summed * centered_scaled, zero_f_h),
    )
    # atomic-add to sum_x: sum over rows of summed
    ct.atomic_add(
        sum_x_out_ptr,
        (h_idx,),
        ct.where(col_active, summed, zero_f_h),
    )

    correction = (weighted * HIDDEN_C - row_sum - centered_scaled * row_dot) * (
        scale * INV_HIDDEN
    )
    update = residual_1d + correction

    raw_index = ct.load(index_ptr, index=(row,), shape=(1,))
    zero_i64 = ct.full((1,), 0, dtype=ct.int64)
    vocab_i64 = ct.full((1,), VOCAB_C, dtype=ct.int64)
    minus_one_i64 = ct.full((1,), -1, dtype=ct.int64)
    valid = (raw_index >= zero_i64) & (raw_index < vocab_i64) & (raw_index != minus_one_i64)
    safe_index = ct.where(valid, raw_index, vocab_i64)

    zero_i32 = ct.zeros((BLOCK_H,), dtype=ct.int32)
    row_bcast = ct.astype(safe_index, ct.int32) + zero_i32
    update_masked = ct.where(col_active, update, zero_f_h)
    ct.atomic_add(scatter_out_ptr, (row_bcast, h_idx), update_masked)


@oracle_impl(
    hardware="B200",
    point="cce081e2",
    block_h=1024,
    block_zero=2048,
    zero_warps=8,
    scatter_warps=8,
)
def oracle_forward(inputs, *, block_h: int, block_zero: int, zero_warps: int, scatter_warps: int):
    del zero_warps, scatter_warps
    (
        arg0,
        arg1,
        arg2,
        arg3,
        weight,
        src,
        mean,
        scale,
        residual,
        index,
        *_shape_params,
    ) = inputs

    device = arg0.device
    total_scatter = VOCAB * HIDDEN

    scatter_out = torch.empty_strided(
        (VOCAB, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    sum_prod_out = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    sum_x_out = torch.zeros((HIDDEN,), device=device, dtype=torch.float32)
    row_sum_partials = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot_partials = torch.empty((ROWS,), device=device, dtype=torch.float32)

    # Flatten f32 [1, 128, 4096] to [128, 4096].
    src_2d = src.view(ROWS, HIDDEN)
    residual_2d = residual.view(ROWS, HIDDEN)
    mean_1d = mean.view(ROWS)
    scale_1d = scale.view(ROWS)
    index_1d = index.view(ROWS)

    stream = torch.cuda.current_stream()
    zero_grid = ((total_scatter + block_zero - 1) // block_zero, 1, 1)
    ct.launch(
        stream,
        zero_grid,
        _zero_kernel,
        (scatter_out, total_scatter, HIDDEN, block_zero),
    )

    # Row-level reductions: whole hidden dim per row.
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _row_reduce_kernel,
        (arg0, arg1, arg2, arg3, weight, src_2d, mean_1d, scale_1d,
         row_sum_partials, row_dot_partials, HIDDEN),
    )

    # Scatter/reduce over hidden chunks.
    h_chunks = (HIDDEN + block_h - 1) // block_h
    ct.launch(
        stream,
        (ROWS, h_chunks, 1),
        _scatter_and_col_reduce_kernel,
        (arg0, arg1, arg2, arg3, weight, src_2d, mean_1d, scale_1d,
         residual_2d, index_1d, row_sum_partials, row_dot_partials,
         sum_prod_out, sum_x_out, scatter_out,
         HIDDEN, VOCAB, block_h, 1.0 / HIDDEN),
    )
    return sum_prod_out, sum_x_out, scatter_out
