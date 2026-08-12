"""cuTile port of var_mean_8314454e86eb: DistilBERT embedding + LayerNorm inference.

Precomputes word+position embedding sum via torch (with the bf16 rounding
boundary the Triton oracle preserves in the non-inductor-numerics path), then
runs a cuTile row LayerNorm over it. HIDDEN=768 padded to BLOCK_H=1024.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


@ct.kernel
def _layernorm_bf16_kernel_padded(
    x_ptr,       # bf16 [ROWS, BLOCK_H]
    weight_ptr,  # bf16 [BLOCK_H]
    bias_ptr,    # bf16 [BLOCK_H]
    out_ptr,     # bf16 [ROWS, BLOCK_H]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_H))
    xf = ct.astype(x, ct.float32)
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))
    xf_m = ct.where(mask, xf, 0.0)
    mean = ct.sum(xf_m) * (1.0 / HIDDEN)
    centered = ct.where(mask, xf - mean, 0.0)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-12)
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_f = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))
    y = centered * invstd * weight_f + bias_f
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="5095537f", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        word_table,
        word_ids,
        position_ids,
        position_table,
        weight,
        bias,
        view_shape0,
        view_shape1,
        view_shape2,
    ) = inputs

    batch = int(word_ids.shape[0])
    seq_len = int(word_ids.shape[1])
    hidden = int(word_table.shape[1])
    rows = batch * seq_len

    # Precompute embedding sum with bf16 rounding boundary (non-inductor path)
    word_ids_flat = word_ids.view(-1)
    seq = torch.arange(rows, device=word_table.device) % seq_len
    position_id_flat = position_ids.view(-1)[seq]
    word = word_table[word_ids_flat].to(torch.float32)
    position = position_table[position_id_flat].to(torch.float32)
    x = (word + position).to(torch.bfloat16)  # bf16 rounding boundary

    if BLOCK_H != hidden:
        x_p = torch.zeros((rows, BLOCK_H), device=word_table.device, dtype=torch.bfloat16)
        x_p[:, :hidden] = x
        w_p = torch.zeros((BLOCK_H,), device=word_table.device, dtype=torch.bfloat16)
        w_p[:hidden] = weight
        b_p = torch.zeros((BLOCK_H,), device=word_table.device, dtype=torch.bfloat16)
        b_p[:hidden] = bias
        out_p = torch.empty((rows, BLOCK_H), device=word_table.device, dtype=torch.bfloat16)
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            (rows, 1, 1),
            _layernorm_bf16_kernel_padded,
            (x_p, w_p, b_p, out_p, hidden, BLOCK_H),
        )
        out = out_p[:, :hidden].contiguous().view(batch, seq_len, hidden)
    else:
        out = torch.empty((batch, seq_len, hidden), device=word_table.device, dtype=torch.bfloat16)
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            (rows, 1, 1),
            _layernorm_bf16_kernel_padded,
            (x, weight, bias, out.view(rows, hidden), hidden, BLOCK_H),
        )

    return (
        out,
        out.view(_as_shape(view_shape0)),
        out.view(_as_shape(view_shape1)),
        out.view(_as_shape(view_shape2)),
    )
