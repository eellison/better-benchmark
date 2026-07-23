"""cuTile port of sum_59fc6f384249: beit QKV cat + column sum.

Triton does 3 kernels: (materialize q/v + partial sum), (materialize k), and
(final sum with bf16 rounding). cuTile matches by using host-side torch.cat
for the QKV materialization (metadata-level shuffle, decreasing fusion is
allowed) plus 2 in-kernel reductions for the column sum: a partial-sum
kernel and a finalize kernel with in-kernel bf16 rounding. NO torch.sum in
oracle_forward.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 25216
COLS = 2304
HEAD_COLS = 768
Q_OR_V_COLS = 1536
# Triton uses 59 partial chunks of 428 rows; cuTile requires power-of-2 tile
# dims so we use 128 chunks of 256 rows (128 * 256 = 32768 >= 25216).
PARTIAL_CHUNKS = 128
PARTIAL_ROWS = 256
BLOCK_COL = 64
FINAL_BLOCK = 128  # >= PARTIAL_CHUNKS, power of 2


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@ct.kernel
def _partial_col_sum_kernel(
    qkv_ptr,         # bf16 [ROWS, COLS]
    partial_ptr,     # f32  [PARTIAL_CHUNKS, COLS]
    ROWS_C: ct.Constant[int],
    PARTIAL_ROWS_C: ct.Constant[int],
    BLOCK_COL_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    col_block = ct.bid(1)
    # Load a (PARTIAL_ROWS, BLOCK_COL) tile from qkv.
    tile = ct.load(
        qkv_ptr, index=(chunk, col_block),
        shape=(PARTIAL_ROWS_C, BLOCK_COL_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_f = ct.astype(tile, ct.float32)
    # Mask out rows past ROWS_C (padding is zero which is correct — the sum
    # is unaffected).
    row_idx = ct.arange(PARTIAL_ROWS_C, dtype=ct.int32) + chunk * PARTIAL_ROWS_C
    row_valid = ct.reshape(row_idx < ROWS_C, (PARTIAL_ROWS_C, 1))
    zero = ct.zeros((PARTIAL_ROWS_C, BLOCK_COL_C), dtype=ct.float32)
    masked = ct.where(row_valid, tile_f, zero)
    partial = ct.sum(masked, axis=0)  # (BLOCK_COL,)
    partial_2d = ct.reshape(partial, (1, BLOCK_COL_C))
    ct.store(partial_ptr, index=(chunk, col_block), tile=partial_2d)


@ct.kernel
def _final_col_sum_kernel(
    partial_ptr,     # f32  [PARTIAL_CHUNKS, COLS]
    sum_ptr,         # f32  [COLS]
    PARTIAL_CHUNKS_C: ct.Constant[int],
    FINAL_BLOCK_C: ct.Constant[int],
    BLOCK_COL_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    tile = ct.load(
        partial_ptr, index=(0, col_block),
        shape=(FINAL_BLOCK_C, BLOCK_COL_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    chunk_idx = ct.arange(FINAL_BLOCK_C, dtype=ct.int32)
    chunk_valid = ct.reshape(chunk_idx < PARTIAL_CHUNKS_C, (FINAL_BLOCK_C, 1))
    zero = ct.zeros((FINAL_BLOCK_C, BLOCK_COL_C), dtype=ct.float32)
    masked = ct.where(chunk_valid, tile, zero)
    total = ct.sum(masked, axis=0)  # (BLOCK_COL,)
    # Match Triton: bf16 round then back to f32 before storing.
    total_bf = ct.astype(total, ct.bfloat16)
    total_f = ct.astype(total_bf, ct.float32)
    ct.store(sum_ptr, index=(col_block,), tile=total_f)


@oracle_impl(hardware="B200", point="96e55468")
def oracle_forward(inputs):
    arg0, arg1, arg2, sp0, sp1, sp2, sp3 = inputs

    # QKV materialization via metadata reshape (cat + view + permute).
    # This mirrors what Triton stores into qkv from q/v/k reads, done in one
    # torch call rather than 3 cuTile kernels — decreasing fusion is allowed.
    cat = torch.cat([arg0, arg1, arg2])
    view = cat.view(*_shape_tuple(sp0))
    permute = view.permute(1, 3, 0, 2, 4).contiguous()
    view_1 = permute.view(*_shape_tuple(sp1))
    view_2 = view_1.view(*_shape_tuple(sp2))  # [ROWS, COLS] bf16
    # view_2 is contiguous [ROWS, COLS]

    device = view_2.device
    partial = torch.empty(
        (PARTIAL_CHUNKS, COLS), device=device, dtype=torch.float32,
    )
    sum_base = torch.empty_strided(
        (COLS,), (1,), device=device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (PARTIAL_CHUNKS, COLS // BLOCK_COL, 1),
        _partial_col_sum_kernel,
        (view_2, partial, ROWS, PARTIAL_ROWS, BLOCK_COL),
    )
    ct.launch(
        stream,
        (COLS // BLOCK_COL, 1, 1),
        _final_col_sum_kernel,
        (partial, sum_base, PARTIAL_CHUNKS, FINAL_BLOCK, BLOCK_COL),
    )

    view_2_t = torch.as_strided(view_2, (COLS, ROWS), (1, COLS), 0)
    slice_q = torch.as_strided(sum_base, (HEAD_COLS,), (1,), 0)
    slice_v = torch.as_strided(sum_base, (HEAD_COLS,), (1,), 2 * HEAD_COLS)
    return view_2, view_2_t, slice_q, slice_v
