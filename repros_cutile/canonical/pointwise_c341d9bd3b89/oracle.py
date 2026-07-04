"""cuTile port of pointwise_c341d9bd3b89: Blenderbot embedding gather.

Per row: gather from f32 weight table using i64 token indices. HIDDEN=2560.
2560 = 5 * 512, non-power-of-2. Use gather over columns.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 2560
SEQ = 128


@ct.kernel
def _embedding_gather_kernel(
    weight_ptr,  # f32 [VOCAB, HIDDEN]
    index_ptr,   # i64 [N_ROWS]
    out_ptr,     # f32 [N_ROWS, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
):
    row = ct.bid(0)
    d_tile = ct.bid(1)

    token_id = ct.load(index_ptr, index=(row,), shape=(1,))
    tid_2d = ct.reshape(token_id, (1, 1))
    cols = ct.arange(BLOCK_D, dtype=ct.int64) + d_tile * BLOCK_D
    col_mask = cols < HIDDEN_C
    cols_2d = ct.reshape(cols, (1, BLOCK_D))
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_D))

    values = ct.gather(weight_ptr, (tid_2d, cols_2d), mask=col_mask_2d, padding_value=ct.float32(0.0))

    # scatter to out
    col_idx = cols
    row_idx = ct.full(shape=(BLOCK_D,), fill_value=row, dtype=ct.int64)
    ct.scatter(
        out_ptr, (row_idx, col_idx),
        ct.reshape(values, (BLOCK_D,)),
        mask=col_mask,
    )


@oracle_impl(hardware="B200", point="c4650a85")
@oracle_impl(hardware="B200", point="0dc67f27")
def oracle_forward(inputs):
    weight, indices = inputs
    batch = int(indices.shape[0])
    n_rows = int(indices.numel())
    output = torch.empty_strided(
        (batch, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=weight.device,
        dtype=torch.float32,
    )
    output_2d = output.view(n_rows, HIDDEN)
    indices_flat = indices.view(-1)

    BLOCK_D = 512
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, (HIDDEN + BLOCK_D - 1) // BLOCK_D, 1),
        _embedding_gather_kernel,
        (weight, indices_flat, output_2d, HIDDEN, BLOCK_D),
    )
    return output
