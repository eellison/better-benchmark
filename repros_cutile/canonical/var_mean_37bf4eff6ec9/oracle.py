"""cuTile port of var_mean_37bf4eff6ec9: GPT-Neo token+position embedding + LayerNorm.

Uses PyTorch's embedding + add to build the bf16 sum, then a cuTile row LayerNorm
kernel to compute the affine-normalized output. A constant False mask is
emitted via torch.zeros.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _layernorm_kernel(
    add_ptr,        # bf16 [rows, hidden]
    weight_ptr,     # bf16 [hidden]
    bias_ptr,       # bf16 [hidden]
    out_ptr,        # bf16 [rows, hidden]
    hidden: ct.Constant[int],
    eps: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x_bf16 = ct.load(add_ptr, index=(row, 0), shape=(1, BLOCK_H))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))

    x = ct.astype(x_bf16, ct.float32)
    # mean = sum(x) / hidden
    mean = ct.sum(x) * (1.0 / hidden)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / hidden)
    invstd = ct.rsqrt(variance + eps)

    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    # Broadcast weight/bias from (BLOCK_H,) to (1, BLOCK_H)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    affine = centered * invstd * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="5247d883", BLOCK_H=2048)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        view_shape0,
        view_shape1,
        _expand_shape,
        view_shape2,
    ) = inputs

    batch = int(token_ids.shape[0])
    seq = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq

    # Token embedding: [batch, seq, hidden] bf16
    token_embed = torch.nn.functional.embedding(token_ids, token_table)
    # Position embedding for positions [0..seq-1]: [seq, hidden] bf16
    position_ids = torch.arange(seq, device=token_ids.device, dtype=torch.int64)
    position_embed = torch.nn.functional.embedding(
        position_ids.unsqueeze(0), position_table
    )
    add_out = token_embed + position_embed  # bf16 add (matches "rtne" default)

    add_out = add_out.contiguous().view(rows, hidden)

    norm_base = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    norm_2d = norm_base.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_kernel,
        (add_out, weight, bias, norm_2d, hidden, EPS, BLOCK_H),
    )

    # Mask is all-False for adjacent-position diffs (since positions are 0,1,2,...)
    mask = torch.zeros((batch, seq), device=token_ids.device, dtype=torch.bool)

    add_out_view = add_out.view(batch, seq, hidden)
    return (
        add_out_view,
        norm_base.view(_shape_tuple(view_shape0)),
        norm_base.view(_shape_tuple(view_shape1)),
        mask,
        norm_base.view(_shape_tuple(view_shape2)),
    )
