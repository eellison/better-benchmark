"""cuTile port of var_mean_59d866109a4c: GPT-J embedding + LayerNorm.

Embedding lookup is pre-computed with torch.embedding (indirect indexing not
required inside the kernel), then a row LayerNorm computes mean/rsqrt and
affine+bf16 outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@ct.kernel
def _layernorm_kernel(
    embedding_ptr,      # f32 [rows, HIDDEN]
    weight_ptr,         # f32 [HIDDEN]
    bias_ptr,           # f32 [HIDDEN]
    mean_out_ptr,       # f32 [rows]
    invstd_out_ptr,     # f32 [rows]
    bf16_out_ptr,       # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)

    x_f32 = ct.load(embedding_ptr, index=(row, 0), shape=(1, HIDDEN))
    x = ct.astype(x_f32, ct.float64)  # match Triton's f64 reduction

    mean64 = ct.sum(x) * (1.0 / HIDDEN_F)
    centered64 = x - mean64
    variance64 = ct.sum(centered64 * centered64) * (1.0 / HIDDEN_F)
    invstd64 = ct.rsqrt(variance64 + EPS)
    mean = ct.astype(mean64, ct.float32)
    invstd = ct.astype(invstd64, ct.float32)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN,))
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN,))
    weight_64 = ct.astype(weight, ct.float64)
    bias_64 = ct.astype(bias, ct.float64)
    weight_2d = ct.reshape(weight_64, (1, HIDDEN))
    bias_2d = ct.reshape(bias_64, (1, HIDDEN))
    affine64 = centered64 * invstd64 * weight_2d + bias_2d
    affine32 = ct.astype(affine64, ct.float32)
    affine_bf16 = ct.astype(affine32, ct.bfloat16)

    ct.store(mean_out_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(invstd_out_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.store(bf16_out_ptr, index=(row, 0), tile=affine_bf16)


@oracle_impl(hardware="B200", point="c18de35b")
def oracle_forward(inputs):
    embedding_table, token_ids, weight, bias, view_shape = inputs
    batch = int(token_ids.shape[0])
    seq = int(token_ids.shape[1])
    hidden = int(embedding_table.shape[1])
    rows = batch * seq
    device = embedding_table.device

    # Pre-compute embedding on host (torch)
    embedding = torch.ops.aten.embedding.default(embedding_table, token_ids)
    # embedding has shape (1, 128, 4096), stride (128*4096, 4096, 1)

    mean = torch.empty_strided(
        (batch, seq, 1), (seq, 1, 1),
        device=device, dtype=torch.float32,
    )
    invstd = torch.empty_strided(
        (batch, seq, 1), (seq, 1, 1),
        device=device, dtype=torch.float32,
    )
    bf16_base = torch.empty_strided(
        (batch, seq, hidden), (seq * hidden, hidden, 1),
        device=device, dtype=torch.bfloat16,
    )

    embedding_2d = embedding.view(rows, hidden)
    mean_1d = mean.view(rows)
    invstd_1d = invstd.view(rows)
    bf16_2d = bf16_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _layernorm_kernel,
        (embedding_2d, weight, bias, mean_1d, invstd_1d, bf16_2d, hidden, float(hidden)),
    )

    return embedding, mean, invstd, bf16_base.view(_shape_tuple(view_shape))
