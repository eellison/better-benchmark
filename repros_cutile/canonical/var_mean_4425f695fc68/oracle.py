"""cuTile port of var_mean_4425f695fc68: TrOCR embedding + LayerNorm.

Uses torch for the embedding gathers, cuTile for the LN row kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
SEQ_LEN = 256
HIDDEN = 1024
ROWS = BATCH * SEQ_LEN
POSITION_OFFSET = 2
EPS = 1.0e-5


@ct.kernel
def _layernorm_kernel(
    input_bf16_ptr,   # (ROWS, HIDDEN) bf16
    weight_ptr,       # (HIDDEN,) bf16
    bias_ptr,         # (HIDDEN,) bf16
    out_ptr,          # (ROWS, HIDDEN) bf16
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.astype(ct.load(input_bf16_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    mean = ct.sum(x) * (1.0 / BLOCK_H)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / BLOCK_H)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.reshape(
        ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32),
        (1, BLOCK_H),
    )
    bias = ct.reshape(
        ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32),
        (1, BLOCK_H),
    )
    affine = normalized * weight + bias
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="c9f1009b", BLOCK_M=1, BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    token_table, token_ids, position_table, weight, bias, _s0, shape1, shape2, shape3 = inputs

    # Do the embedding gathers with torch.
    embedding = torch.ops.aten.embedding.default(token_table, token_ids, 1)
    mul = embedding * 1.0
    iota = torch.arange(SEQ_LEN, device=token_ids.device, dtype=torch.int64)
    expand = iota.expand(BATCH, SEQ_LEN)
    add = expand + POSITION_OFFSET
    embedding_1 = torch.ops.aten.embedding.default(position_table, add)
    add_1 = mul + embedding_1  # bf16
    # Now LN this in cuTile.
    input_flat = add_1.view(ROWS, HIDDEN)
    out = torch.empty_strided(
        (BATCH, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _layernorm_kernel,
        (input_flat, weight, bias, out.view(ROWS, HIDDEN), BLOCK_H),
    )

    def _as_shape(s):
        return tuple(int(d) for d in s)
    return (
        out,
        out.view(_as_shape(shape1)),
        out.view(_as_shape(shape2)),
        out.view(_as_shape(shape3)),
    )
