"""cuTile port of var_mean_var_mean_0dd15b92dc70: MBART dual LayerNorm with embedding fusion.

For each row (batch*seq): gather token embedding + position embedding, add,
apply two chained LayerNorms with bf16 affine boundaries. Emit first LN output
(bf16) and final LN output (bf16, three aliased views).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ_LEN = 1024
HIDDEN = 1024
POSITION_OFFSET = 2
EPS = 1.0e-5


@ct.kernel
def _mbart_dual_layernorm_kernel(
    token_table_ptr,      # bf16 (VOCAB, HIDDEN)
    token_ids_ptr,        # i64 (rows,)
    position_table_ptr,   # bf16 (POS_MAX, HIDDEN)
    weight0_ptr,          # bf16 (HIDDEN,)
    bias0_ptr,            # bf16 (HIDDEN,)
    weight1_ptr,          # bf16 (HIDDEN,)
    bias1_ptr,            # bf16 (HIDDEN,)
    first_out_ptr,        # bf16 (rows, HIDDEN)
    final_out_ptr,        # bf16 (rows, HIDDEN)
    BLOCK_N: ct.Constant[int],
    SEQ_LEN_: ct.Constant[int],
):
    row = ct.bid(0)

    token_id_tile = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    # Convert to scalar-broadcast — use as a 1-elem index
    token_id_scalar = ct.reshape(token_id_tile, ())
    position_id_scalar = (row % SEQ_LEN_) + POSITION_OFFSET

    token = ct.load(token_table_ptr, index=(token_id_scalar, 0), shape=(1, BLOCK_N))
    position = ct.load(position_table_ptr, index=(position_id_scalar, 0), shape=(1, BLOCK_N))

    token_f = ct.astype(token, ct.float32)
    position_f = ct.astype(position, ct.float32)
    x0_bf = ct.astype(token_f + position_f, ct.bfloat16)
    x0 = ct.astype(x0_bf, ct.float32)

    mean0 = ct.sum(x0, axis=1, keepdims=True) * (1.0 / BLOCK_N)
    centered0 = x0 - mean0
    var0 = ct.sum(centered0 * centered0, axis=1, keepdims=True) * (1.0 / BLOCK_N)
    invstd0 = ct.rsqrt(var0 + EPS)

    weight0 = ct.load(weight0_ptr, index=(0,), shape=(BLOCK_N,))
    bias0 = ct.load(bias0_ptr, index=(0,), shape=(BLOCK_N,))
    weight0_f = ct.reshape(ct.astype(weight0, ct.float32), (1, BLOCK_N))
    bias0_f = ct.reshape(ct.astype(bias0, ct.float32), (1, BLOCK_N))
    norm0 = centered0 * invstd0
    affine0 = norm0 * weight0_f + bias0_f
    first_bf16 = ct.astype(affine0, ct.bfloat16)
    ct.store(first_out_ptr, index=(row, 0), tile=first_bf16)

    x1 = ct.astype(first_bf16, ct.float32)
    mean1 = ct.sum(x1, axis=1, keepdims=True) * (1.0 / BLOCK_N)
    centered1 = x1 - mean1
    var1 = ct.sum(centered1 * centered1, axis=1, keepdims=True) * (1.0 / BLOCK_N)
    invstd1 = ct.rsqrt(var1 + EPS)

    weight1 = ct.load(weight1_ptr, index=(0,), shape=(BLOCK_N,))
    bias1 = ct.load(bias1_ptr, index=(0,), shape=(BLOCK_N,))
    weight1_f = ct.reshape(ct.astype(weight1, ct.float32), (1, BLOCK_N))
    bias1_f = ct.reshape(ct.astype(bias1, ct.float32), (1, BLOCK_N))
    norm1 = centered1 * invstd1
    affine1 = norm1 * weight1_f + bias1_f
    ct.store(final_out_ptr, index=(row, 0), tile=ct.astype(affine1, ct.bfloat16))


@oracle_impl(hardware="B200", point="7a83b18a")
def oracle_forward(inputs):
    (
        token_table,
        token_ids,
        position_table,
        weight0,
        bias0,
        weight1,
        bias1,
        shape0,
        shape1,
        shape2,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len

    first = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    final = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    first_2d = first.view(rows, hidden)
    final_2d = final.view(rows, hidden)
    token_ids_1d = token_ids.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _mbart_dual_layernorm_kernel,
        (
            token_table,
            token_ids_1d,
            position_table,
            weight0,
            bias0,
            weight1,
            bias1,
            first_2d,
            final_2d,
            hidden,
            seq_len,
        ),
    )
    return (
        first,
        final.view(tuple(int(dim) for dim in shape0)),
        final.view(tuple(int(dim) for dim in shape1)),
        final.view(tuple(int(dim) for dim in shape2)),
    )
