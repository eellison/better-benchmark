"""cuTile port of sum_sum_sum_c304a1364550: ALBERT 12x column-sum with clone.

Computes column-sum (sum over dim=0) for 11 [4096,4096] bf16 matrices plus a
12th matrix (a permute-clone of a [512,512,64] input viewed as [4096,4096]).
Each per-column sum is bf16-rounded then f32-added into a running total in a
sequential chain.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4096
C = 4096


@ct.kernel
def _column_sum_partial_kernel(
    matrix_ptr,      # bf16 [N, C]
    partial_ptr,     # f32 [num_row_blocks, C]
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    row_block = ct.bid(1)
    tile = ct.load(matrix_ptr, index=(row_block, col_block), shape=(BLOCK_ROWS, BLOCK_COLS))
    tile_f = ct.astype(tile, ct.float32)
    partial = ct.sum(tile_f, axis=0, keepdims=True)  # (1, BLOCK_COLS)
    ct.store(partial_ptr, index=(row_block, col_block), tile=partial)


@ct.kernel
def _finalize_column_sum_kernel(
    partial_ptr,
    out_ptr,
    BLOCK_P: ct.Constant[int],
    BLOCK_COLS: ct.Constant[int],
):
    col_block = ct.bid(0)
    values = ct.load(partial_ptr, index=(0, col_block), shape=(BLOCK_P, BLOCK_COLS))
    total = ct.sum(values, axis=0)  # (BLOCK_COLS,)
    total_bf = ct.astype(total, ct.bfloat16)
    total_bf_f = ct.astype(total_bf, ct.float32)
    ct.store(out_ptr, index=(col_block,), tile=total_bf_f)


def _column_sum_bf16_rounded(matrix, partial_buf, per_sum_out):
    stream = torch.cuda.current_stream()
    BLOCK_ROWS = 128
    BLOCK_COLS = 64
    num_row_blocks = N // BLOCK_ROWS
    ct.launch(
        stream,
        (C // BLOCK_COLS, num_row_blocks, 1),
        _column_sum_partial_kernel,
        (matrix, partial_buf, BLOCK_ROWS, BLOCK_COLS),
    )
    FINAL_BLOCK_COLS = 16
    ct.launch(
        stream,
        (C // FINAL_BLOCK_COLS, 1, 1),
        _finalize_column_sum_kernel,
        (partial_buf, per_sum_out, num_row_blocks, FINAL_BLOCK_COLS),
    )


@oracle_impl(hardware="B200", point="4a97732f")
def oracle_forward(inputs):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11,
        *_shape_args,
    ) = inputs
    device = arg0.device
    BLOCK_ROWS = 128
    num_row_blocks = N // BLOCK_ROWS

    clone = arg11.view(8, 64, 512, 64).permute(0, 2, 1, 3).contiguous().view(N, C)

    partial_buf = torch.empty((num_row_blocks, C), device=device, dtype=torch.float32)
    per_sum = torch.empty((C,), device=device, dtype=torch.float32)
    running = torch.zeros((C,), device=device, dtype=torch.float32)

    for mat in [arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, clone]:
        _column_sum_bf16_rounded(mat, partial_buf, per_sum)
        running = running + per_sum

    return clone, clone.permute(1, 0), running
