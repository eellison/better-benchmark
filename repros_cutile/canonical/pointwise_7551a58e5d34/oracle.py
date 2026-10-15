"""cuTile port of pointwise_7551a58e5d34 (SCHEDULER_FUSION): DistillGPT2
dual pad layout — bf16 [ROWS, COLS] input turned into a padded transpose
[PAD_COLS, ROWS] and a padded row-major [ROWS, PAD_COLS].

Strategy: zero-fill both outputs with torch.zeros, then run a cuTile kernel
over the valid [ROWS, COLS] input region and use ct.scatter (masked write)
to place elements into their padded destinations.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
COLS = 50257
PAD = 7
PAD_COLS = COLS + PAD


@ct.kernel
def _dual_pad_kernel(
    input_ptr,             # bf16 [ROWS, COLS]
    out_transposed_ptr,    # bf16 [PAD_COLS, ROWS]
    out_rowmajor_ptr,      # bf16 [ROWS, PAD_COLS]
    COLS_CONST: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    rows = ct.arange(BLOCK_M, dtype=ct.int64) + row_block * BLOCK_M
    cols = ct.arange(BLOCK_N, dtype=ct.int64) + col_block * BLOCK_N
    rows_2d = ct.reshape(rows, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    rows_broad = ct.broadcast_to(rows_2d, (BLOCK_M, BLOCK_N))
    cols_broad = ct.broadcast_to(cols_2d, (BLOCK_M, BLOCK_N))
    in_bounds = cols_broad < COLS_CONST

    zero64_2d = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.int64)
    safe_cols = ct.where(in_bounds, cols_broad, zero64_2d)
    values = ct.gather(input_ptr, (rows_broad, safe_cols))

    ct.scatter(out_transposed_ptr, (cols_broad, rows_broad), values, mask=in_bounds)
    ct.scatter(out_rowmajor_ptr, (rows_broad, cols_broad), values, mask=in_bounds)


@oracle_impl(hardware="B200", point="846b3407", BLOCK_M=16, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    x, _s0, _s1, _s2 = inputs
    device = x.device

    out_transposed = torch.zeros(
        (PAD_COLS, ROWS), device=device, dtype=torch.bfloat16
    )
    out_rowmajor = torch.zeros(
        (ROWS, PAD_COLS), device=device, dtype=torch.bfloat16
    )

    x_2d = x.view(ROWS, COLS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((ROWS + BLOCK_M - 1) // BLOCK_M, (COLS + BLOCK_N - 1) // BLOCK_N, 1),
        _dual_pad_kernel,
        (x_2d, out_transposed, out_rowmajor, COLS, BLOCK_M, BLOCK_N),
    )
    return out_transposed, out_rowmajor
