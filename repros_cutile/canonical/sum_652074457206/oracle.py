"""cuTile port of sum_652074457206 (SCHEDULER_FUSION): OPT bf16 masked
materialization plus dim-0 column sum, plus returned permute alias.

Per column tile, we compute:
  where = (arg0 <= 0) ? 0 : arg1     ← bf16 store
  partial = sum(where.to(f32), axis=0)   ← per-tile partial

Then a small kernel reduces partials to the final [3072] bf16-rounded sum.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
COLS = 3072
ROW_BLOCK = 128
BLOCK_N = 256
FINAL_BLOCK_N = 32


@ct.kernel
def _masked_materialize_partial_sum_kernel(
    pred_ptr,    # bf16 [ROWS, COLS]
    source_ptr,  # bf16 [ROWS, COLS]
    where_ptr,   # bf16 [ROWS, COLS]
    partial_ptr, # f32 [NUM_ROW_TILES, COLS]
    ROW_BLOCK_: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)
    pred = ct.load(pred_ptr, index=(row_tile, col_tile), shape=(ROW_BLOCK_, BLOCK_N_))
    source = ct.load(source_ptr, index=(row_tile, col_tile), shape=(ROW_BLOCK_, BLOCK_N_))
    pred_f = ct.astype(pred, ct.float32)
    zero_f = ct.full((ROW_BLOCK_, BLOCK_N_), 0.0, dtype=ct.float32)
    source_f = ct.astype(source, ct.float32)
    selected_f = ct.where(pred_f <= zero_f, zero_f, source_f)
    selected = ct.astype(selected_f, ct.bfloat16)
    ct.store(where_ptr, index=(row_tile, col_tile), tile=selected)
    partial = ct.sum(ct.astype(selected, ct.float32), axis=0)
    ct.store(
        partial_ptr,
        index=(row_tile, col_tile),
        tile=ct.reshape(partial, (1, BLOCK_N_)),
    )


@ct.kernel
def _final_sum_kernel(
    partial_ptr, # f32 [NUM_ROW_TILES, COLS]
    sum_ptr,     # f32 [COLS]
    NUM_ROW_TILES_: ct.Constant[int],
    BLOCK_N_: ct.Constant[int],
):
    col_tile = ct.bid(0)
    partial = ct.load(
        partial_ptr, index=(0, col_tile), shape=(NUM_ROW_TILES_, BLOCK_N_)
    )
    total = ct.sum(ct.astype(partial, ct.float32), axis=0)
    rounded = ct.astype(ct.astype(total, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col_tile,), tile=rounded)


@oracle_impl(
    hardware="B200",
    point="58ff2bc5",
    ROW_BLOCK=ROW_BLOCK,
    BLOCK_N=BLOCK_N,
    FINAL_BLOCK_N=FINAL_BLOCK_N,
)
def oracle_forward(
    inputs, *, ROW_BLOCK, BLOCK_N, FINAL_BLOCK_N,
):
    arg0_1, arg1_1, _shape_param_0 = inputs
    m = int(arg1_1.shape[0])
    n = int(arg1_1.shape[1])
    num_row_tiles = (m + ROW_BLOCK - 1) // ROW_BLOCK

    where = torch.empty_strided((m, n), (n, 1), device=arg1_1.device, dtype=torch.bfloat16)
    partial = torch.empty_strided(
        (num_row_tiles, n),
        (n, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    sum_out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_0),
        (1,),
        device=arg1_1.device,
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_tiles, (n + BLOCK_N - 1) // BLOCK_N, 1),
        _masked_materialize_partial_sum_kernel,
        (arg0_1, arg1_1, where, partial, ROW_BLOCK, BLOCK_N),
    )
    ct.launch(
        stream,
        ((n + FINAL_BLOCK_N - 1) // FINAL_BLOCK_N, 1, 1),
        _final_sum_kernel,
        (partial, sum_out, num_row_tiles, FINAL_BLOCK_N),
    )
    return where, where.permute(1, 0), sum_out
