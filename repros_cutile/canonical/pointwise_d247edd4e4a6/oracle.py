"""cuTile port of pointwise_d247edd4e4a6: MobileBert embedding-gradient scatter.

Mirrors Triton's 3-kernel structure:
  1. _init_and_tail_kernel: zero the scatter destination + populate tail
     (vocab tail slice cast to f32).
  2. _scatter_kernel: build shifted-neighbor gradient rows, atomic_add into
     scatter destination at valid token positions.
  3. _add_base_kernel: add the base vocabulary table (transposed) to the
     accumulated scatter.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
TOKENS = 128
HIDDEN = 128
INPUT_HIDDEN = 384
ARG0_ROWS = 512
ARG0_COLS = 30528
VOCAB = 30522
TAIL_ROWS = 384
ROWS = BATCH * TOKENS
OUT0_NUMEL = TAIL_ROWS * VOCAB    # 11720448
OUT1_NUMEL = VOCAB * HIDDEN        # 3906816


@ct.kernel
def _init_and_tail_kernel(
    vocab_ptr,       # bf16 [ARG0_ROWS, ARG0_COLS]
    tail_out_ptr,    # f32 [OUT0_NUMEL] flat
    scatter_out_ptr, # f32 [OUT1_NUMEL] flat
    N_TAIL: ct.Constant[int],
    N_SCATTER: ct.Constant[int],
    VOCAB_COLS: ct.Constant[int],
    VOCAB_C: ct.Constant[int],
    TAIL_ROW_OFFSET: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK

    # Zero-init scatter destination (OUT1_NUMEL = 3906816).
    scatter_mask = offsets < N_SCATTER
    zeros = ct.zeros((BLOCK,), dtype=ct.float32)
    ct.scatter(scatter_out_ptr, offsets, zeros, mask=scatter_mask)

    # Tail slice: for offsets < N_TAIL, load vocab[tail_row+offset, tail_col]
    # cast to f32 and store to tail_out.
    tail_mask = offsets < N_TAIL
    tail_row = offsets // VOCAB_C  # int32 div
    tail_col = offsets - tail_row * VOCAB_C
    src_row = tail_row + TAIL_ROW_OFFSET
    # ct.gather with 2D coord
    tail_bf = ct.gather(
        vocab_ptr, (src_row, tail_col), mask=tail_mask, padding_value=0.0
    )
    tail_f = ct.astype(tail_bf, ct.float32)
    ct.scatter(tail_out_ptr, offsets, tail_f, mask=tail_mask)


@ct.kernel
def _scatter_kernel(
    hidden_ptr,      # bf16 [ROWS, INPUT_HIDDEN]
    token_id_ptr,    # i64 [ROWS]
    scatter_out_ptr, # f32 [VOCAB, HIDDEN]
    ROWS_TOTAL: ct.Constant[int],
    TOKENS_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    VOCAB_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)  # BLOCK_M=1 → one row per program

    # Load token id (scalar).
    token_id_tile = ct.load(token_id_ptr, index=(row,), shape=(1,))
    valid_token = (token_id_tile >= 0) & (token_id_tile < VOCAB_C) & (token_id_tile != 0)
    valid_scalar = ct.reshape(valid_token, ())  # scalar bool

    token_pos = row - (row // TOKENS_C) * TOKENS_C

    # Load mid row: hidden_ptr[row, hidden:2*hidden] as (BLOCK_H,).
    # In tile-space with tile (1, BLOCK_H), a row of length INPUT_HIDDEN=384
    # partitions into 3 tiles of BLOCK_H=128: index 0..2.
    mid_bf = ct.load(hidden_ptr, index=(row, 1), shape=(1, BLOCK_H))
    mid_f = ct.astype(mid_bf, ct.float32)

    # next row: hidden_ptr[row+1, 2*hidden:3*hidden]; only if token_pos<TOKENS-1
    # and row+1 < ROWS_TOTAL. Use full padding fallback via masked gather.
    cols = ct.arange(BLOCK_H, dtype=ct.int32) + 2 * BLOCK_H  # cols in [2h, 3h)
    row_next = row + 1
    has_next = (token_pos < (TOKENS_C - 1)) & (row_next < ROWS_TOTAL)
    row_next_bcast = ct.full((BLOCK_H,), row_next, dtype=ct.int32)
    next_bf = ct.gather(hidden_ptr, (row_next_bcast, cols),
                        mask=ct.full((BLOCK_H,), has_next, dtype=ct.bool_),
                        padding_value=0.0)
    next_f = ct.astype(next_bf, ct.float32)

    # prev row: hidden_ptr[row-1, 0:hidden]; only if token_pos>0 and row-1 >= 0
    cols_prev = ct.arange(BLOCK_H, dtype=ct.int32)
    row_prev = row - 1
    has_prev = (token_pos > 0) & (row_prev >= 0)
    row_prev_bcast = ct.full((BLOCK_H,), row_prev, dtype=ct.int32)
    prev_bf = ct.gather(hidden_ptr, (row_prev_bcast, cols_prev),
                        mask=ct.full((BLOCK_H,), has_prev, dtype=ct.bool_),
                        padding_value=0.0)
    prev_f = ct.astype(prev_bf, ct.float32)

    value = mid_f + next_f + prev_f

    # atomic_add into scatter_out[token_id, cols] when valid_token.
    # If invalid, redirect row to OOB (>= VOCAB) so check_bounds drops.
    invalid_row = ct.full((1,), VOCAB_C, dtype=ct.int64)
    safe_id = ct.where(valid_token, token_id_tile, invalid_row)
    dst_cols = ct.arange(BLOCK_H, dtype=ct.int32)
    safe_id_i32 = ct.astype(safe_id, ct.int32)
    token_id_bcast = ct.broadcast_to(safe_id_i32, (BLOCK_H,))
    value_1d = ct.reshape(value, (BLOCK_H,))
    ct.atomic_add(scatter_out_ptr, (token_id_bcast, dst_cols), value_1d)


@ct.kernel
def _add_base_kernel(
    vocab_ptr,        # bf16 [ARG0_ROWS, ARG0_COLS]
    scatter_out_ptr,  # f32 flat [OUT1_NUMEL]
    N_SCATTER: ct.Constant[int],
    VOCAB_COLS: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    mask = offsets < N_SCATTER
    vocab_col = offsets // HIDDEN_C
    hidden_row = offsets - vocab_col * HIDDEN_C
    # Load bf16 base value from vocab[hidden_row, vocab_col]. cast to f32.
    base_bf = ct.gather(vocab_ptr, (hidden_row, vocab_col),
                        mask=mask, padding_value=0.0)
    base_f = ct.astype(base_bf, ct.float32)
    accum = ct.gather(scatter_out_ptr, offsets,
                      mask=mask, padding_value=0.0)
    result = base_f + accum
    ct.scatter(scatter_out_ptr, offsets, result, mask=mask)


@oracle_impl(
    hardware="B200",
    point="11f71568",
    INIT_BLOCK=1024,
    BLOCK_M=1,
    BLOCK_H=128,
)
def oracle_forward(inputs, *, INIT_BLOCK: int, BLOCK_M: int, BLOCK_H: int):
    del BLOCK_M  # cuTile version launches 1 row per pid
    arg0_1, arg1_1, arg2_1, _sh0, _sh1, _sh2 = inputs
    device = arg0_1.device

    # arg0: bf16 [512, 30528] vocabulary (with 6-col tail padding beyond 30522).
    # arg1: bf16 flat / [256, 128, 384] hidden values.
    # arg2: i64 [256, 128] token ids.
    # Output 0: f32 [384, 30522] tail slice of arg0 (rows [128..512]).
    # Output 1: f32 [30522, 128] scatter accumulator + vocab-slice base.

    tail = torch.empty_strided(
        (TAIL_ROWS, VOCAB),
        (VOCAB, 1),
        device=device,
        dtype=torch.float32,
    )
    scatter = torch.empty_strided(
        (VOCAB, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()

    # Kernel 1: init scatter + tail slice
    init_grid = (
        (max(OUT0_NUMEL, OUT1_NUMEL) + INIT_BLOCK - 1) // INIT_BLOCK, 1, 1
    )
    tail_flat = tail.view(-1)
    scatter_flat = scatter.view(-1)
    ct.launch(
        stream,
        init_grid,
        _init_and_tail_kernel,
        (arg0_1, tail_flat, scatter_flat, OUT0_NUMEL, OUT1_NUMEL,
         ARG0_COLS, VOCAB, ARG0_ROWS - TAIL_ROWS, INIT_BLOCK),
    )

    # Kernel 2: scatter (BLOCK_M=1 -> one row per program)
    # arg1_1 as [ROWS, INPUT_HIDDEN] view. arg1_1 has 3 dims.
    hidden_2d = arg1_1.view(ROWS, INPUT_HIDDEN)
    token_id_flat = arg2_1.view(ROWS)
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _scatter_kernel,
        (hidden_2d, token_id_flat, scatter, ROWS, TOKENS, HIDDEN, VOCAB, BLOCK_H),
    )

    # Kernel 3: add base vocab slice
    add_grid = ((OUT1_NUMEL + INIT_BLOCK - 1) // INIT_BLOCK, 1, 1)
    ct.launch(
        stream,
        add_grid,
        _add_base_kernel,
        (arg0_1, scatter_flat, OUT1_NUMEL, ARG0_COLS, HIDDEN, INIT_BLOCK),
    )
    return tail, scatter
