"""cuTile port of var_mean_31bb65844b3f (NEW_PATTERN): ALBERT/Electra embedding
LayerNorm. Since cuTile lacks a general kernel-side embedding gather primitive,
we materialize the embedding sum via torch and then apply cuTile LayerNorm.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 128
EPS = 1.0e-12


@ct.kernel
def _layernorm_row_kernel(
    x_ptr,           # bf16 [ROWS, HIDDEN]
    weight_ptr,      # bf16 [HIDDEN]
    bias_ptr,        # bf16 [HIDDEN]
    out_ptr,         # bf16 [ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = ct.astype(x_bf, ct.float32)
    inv_h = 1.0 / HIDDEN_
    x_sum = ct.sum(x, axis=1, keepdims=True)
    x_sq_sum = ct.sum(x * x, axis=1, keepdims=True)
    mean = x_sum * inv_h
    variance = x_sq_sum * inv_h - mean * mean
    invstd = ct.rsqrt(variance + 1.0e-12)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    centered = x - mean
    affine = centered * invstd * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="87078625", BLOCK_M=4, BLOCK_H=128)
@oracle_impl(hardware="B200", point="67f7e2f4", BLOCK_M=8, BLOCK_H=128)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_H):
    (
        word_table,
        token_ids,
        token_type_source,
        position_ids,
        token_type_table,
        position_table,
        weight,
        bias,
        _expand_shape,
        flat_shape,
    ) = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(word_table.shape[1])
    rows = batch * seq_len
    flat_shape = tuple(int(dim) for dim in flat_shape)

    # Materialize embedding sum via torch.
    embedding = torch.ops.aten.embedding.default(word_table, token_ids, 0)
    expand = token_type_source.expand([1, -1])
    gather = torch.ops.aten.gather.default(expand, 1, position_ids)
    expand_1 = gather.expand([batch, seq_len])
    embedding_1 = torch.ops.aten.embedding.default(token_type_table, expand_1)
    add = embedding + embedding_1
    embedding_2 = torch.ops.aten.embedding.default(position_table, position_ids)
    add_1 = add + embedding_2
    # add_1 is bf16 [batch, seq_len, hidden]; reshape to [rows, hidden].
    x_2d = add_1.view(rows, hidden)

    out = torch.empty_strided(
        flat_shape,
        (hidden, 1),
        device=word_table.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_row_kernel,
        (x_2d, weight, bias, out, hidden, BLOCK_H),
    )
    return out
