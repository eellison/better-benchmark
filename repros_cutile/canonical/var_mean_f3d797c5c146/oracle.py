"""cuTile port of var_mean_f3d797c5c146: GPT-J embedding + LayerNorm.

Uses torch for the embedding gather (indirect index) and cuTile for the LN.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 4096
EPS = 1.0e-5


@ct.kernel
def _layernorm_kernel(
    input_bf16_ptr,   # (ROWS, HIDDEN) bf16
    weight_ptr,       # (HIDDEN,) bf16
    bias_ptr,         # (HIDDEN,) bf16
    norm_out_ptr,     # (ROWS, HIDDEN) bf16
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
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="4553b7e1", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    table, token_ids, weight, bias, shape0, shape1, shape2, shape3 = inputs
    device = table.device

    # Embedding gather in torch (indirect index).
    ids = token_ids.view(ROWS)
    embedding = table[ids].contiguous()  # (ROWS, HIDDEN) bf16
    embedding = embedding.view(1, ROWS, HIDDEN)
    normalized_flat = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _layernorm_kernel,
        (embedding.view(ROWS, HIDDEN), weight, bias, normalized_flat, BLOCK_H),
    )
    normalized = normalized_flat.view(1, ROWS, HIDDEN)

    def _as_shape(s):
        return tuple(int(d) for d in s)

    return (
        embedding,
        normalized.view(_as_shape(shape0)),
        normalized.view(_as_shape(shape1)),
        normalized.view(_as_shape(shape2)),
        normalized.view(_as_shape(shape3)),
    )
