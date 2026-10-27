"""cuTile port of pointwise_a4aa707cd2f4: embedding gather (with padding_idx=1, but mul_by_1).

Uses ct.gather over the flattened weight table indexed by token_ids * hidden + col.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK_M = 4
BLOCK_D = 256


@ct.kernel
def _embedding_gather_kernel(
    weight_ptr,   # f32 (V*HIDDEN,) flattened
    index_ptr,    # i64 (N_ROWS,)
    out_ptr,      # f32 (N_ROWS, HIDDEN)
    N_ROWS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_M_C: ct.Constant[int],
    BLOCK_D_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    rows = row_block * BLOCK_M_C + ct.arange(BLOCK_M_C, dtype=ct.int32)
    cols = col_block * BLOCK_D_C + ct.arange(BLOCK_D_C, dtype=ct.int32)
    row_mask = rows < N_ROWS
    col_mask = cols < HIDDEN
    mask_2d = ct.reshape(row_mask, (BLOCK_M_C, 1)) & ct.reshape(col_mask, (1, BLOCK_D_C))

    # Load token ids
    token_ids = ct.gather(index_ptr, rows, mask=row_mask, padding_value=0)
    token_ids_i32 = ct.astype(token_ids, ct.int32)
    # Compute gather indices: token_ids[i] * HIDDEN + cols[j]
    tok_2d = ct.reshape(token_ids_i32, (BLOCK_M_C, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_D_C))
    idx_2d = tok_2d * HIDDEN + cols_2d

    values = ct.gather(weight_ptr, idx_2d, mask=mask_2d, padding_value=0.0)

    # Scatter into output.
    row_2d = ct.reshape(rows, (BLOCK_M_C, 1))
    out_idx_2d = row_2d * HIDDEN + cols_2d
    ct.scatter(out_ptr, out_idx_2d, values, mask=mask_2d)


@oracle_impl(hardware="B200", point="5be451a9")
@oracle_impl(hardware="B200", point="a3ebf15b")
@oracle_impl(hardware="B200", point="7f3a04f0")
@oracle_impl(hardware="B200", point="835de63c")
@oracle_impl(hardware="B200", point="e849d31e")
def oracle_forward(inputs):
    weight, indices = inputs
    batch = int(indices.shape[0])
    seq = int(indices.shape[1])
    hidden = int(weight.shape[1])
    n_rows = int(indices.numel())
    out = torch.empty_strided(
        (batch, seq, hidden), (seq * hidden, hidden, 1),
        device=weight.device, dtype=torch.float32,
    )

    weight_flat = weight.reshape(-1)
    indices_flat = indices.reshape(-1)
    out_flat = out.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), ct.cdiv(hidden, BLOCK_D), 1),
        _embedding_gather_kernel,
        (weight_flat, indices_flat, out_flat, n_rows, hidden, BLOCK_M, BLOCK_D),
    )
    return out
