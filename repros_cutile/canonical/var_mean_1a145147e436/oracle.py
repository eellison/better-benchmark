"""cuTile port of var_mean_1a145147e436 (NEW_PATTERN): MegatronBERT embed+LN.

Per row: token gather + token_type + position gather; two bf16 rounding
boundaries in the add chain; HIDDEN=1024 mean/var, rsqrt+eps=1e-12, affine.
Returns summed embedding (bf16) plus three aliasing views of the normalized
output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _embedding_layernorm_kernel(
    token_table_ptr,      # bf16 [V, HIDDEN]
    token_ids_ptr,        # i64 [ROWS]
    token_type_table_ptr, # bf16 [2, HIDDEN]  (we use row 0)
    position_table_ptr,   # bf16 [512, HIDDEN]
    position_ids_ptr,     # i64 [SEQ_LEN]
    weight_ptr,           # bf16 [HIDDEN]
    bias_ptr,             # bf16 [HIDDEN]
    add_out_ptr,          # bf16 [ROWS, HIDDEN]
    norm_out_ptr,         # bf16 [ROWS, HIDDEN]
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    row_tile = ct.full(shape=(1,), fill_value=row, dtype=ct.int64)
    seq = row_tile - (row_tile // SEQ_LEN) * SEQ_LEN

    token_id = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    position_id = ct.gather(position_ids_ptr, (seq,))

    cols = ct.arange(BLOCK_H, dtype=ct.int64)
    cols_2d = ct.reshape(cols, (1, BLOCK_H))
    token_id_2d = ct.reshape(token_id, (1, 1))
    position_id_2d = ct.reshape(position_id, (1, 1))

    token = ct.astype(
        ct.gather(token_table_ptr, (token_id_2d, cols_2d)),
        ct.float32,
    )
    token_type = ct.astype(
        ct.load(token_type_table_ptr, index=(0, 0), shape=(1, BLOCK_H)),
        ct.float32,
    )
    position = ct.astype(
        ct.gather(position_table_ptr, (position_id_2d, cols_2d)),
        ct.float32,
    )

    add0_bf16 = ct.astype(token + token_type, ct.bfloat16)
    add0_f = ct.astype(add0_bf16, ct.float32)
    x_bf16 = ct.astype(add0_f + position, ct.bfloat16)
    x = ct.astype(x_bf16, ct.float32)
    ct.store(add_out_ptr, index=(row, 0), tile=x_bf16)

    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine_bf16 = ct.astype(normalized * weight_2d + bias_2d, ct.bfloat16)
    ct.store(norm_out_ptr, index=(row, 0), tile=affine_bf16)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="dc92a431", BLOCK_M=2, BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    (
        token_table,
        token_ids,
        token_type_table,
        position_table,
        position_ids,
        weight,
        bias,
        _token_type_shape,
        view_shape0,
        view_shape1,
        view_shape2,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    device = token_table.device
    base_shape = (batch, seq_len, hidden)
    base_stride = (seq_len * hidden, hidden, 1)
    add_out = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bfloat16)
    norm_out = torch.empty_strided(base_shape, base_stride, device=device, dtype=torch.bfloat16)
    add_out_2d = add_out.view(rows, hidden)
    norm_out_2d = norm_out.view(rows, hidden)
    token_ids_flat = token_ids.view(-1)
    position_ids_flat = position_ids.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _embedding_layernorm_kernel,
        (token_table, token_ids_flat, token_type_table, position_table,
         position_ids_flat, weight, bias, add_out_2d, norm_out_2d,
         seq_len, hidden, BLOCK_H),
    )

    return (
        add_out,
        norm_out.view(_as_shape(view_shape0)),
        norm_out.view(_as_shape(view_shape1)),
        norm_out.view(_as_shape(view_shape2)),
    )
