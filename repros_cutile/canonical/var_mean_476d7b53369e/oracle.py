"""cuTile port of var_mean_476d7b53369e: XGLM token+position embedding LayerNorm.

Uses torch.embedding + torch.index for the two lookups (cheaper than
in-kernel gather over 256k-row tables), then runs the residual add + bf16
round + LayerNorm + affine in a single cuTile row kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
EMBED_SCALE = 32.0
POSITION_OFFSET = 2


@ct.kernel
def _residual_ln_kernel(
    scaled_ptr,      # bf16 [rows, HIDDEN] - already scaled-and-rounded token embedding
    position_ptr,    # bf16 [rows, HIDDEN] - position embedding
    weight_ptr,      # bf16 [HIDDEN]
    bias_ptr,        # bf16 [HIDDEN]
    add_out_ptr,     # bf16 [rows, HIDDEN]
    final_out_ptr,   # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    INV_HIDDEN: ct.Constant[float],
):
    row = ct.bid(0)

    scaled = ct.load(scaled_ptr, index=(row, 0), shape=(1, HIDDEN))
    position = ct.load(position_ptr, index=(row, 0), shape=(1, HIDDEN))
    scaled_f = ct.astype(scaled, ct.float32)
    position_f = ct.astype(position, ct.float32)
    add_f = scaled_f + position_f
    add_bf = ct.astype(add_f, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=add_bf)

    x = ct.astype(add_bf, ct.float32)
    mean = ct.sum(x) * INV_HIDDEN
    centered = x - mean
    variance = ct.sum(centered * centered) * INV_HIDDEN
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN,))
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN,))
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, HIDDEN))
    bias_2d = ct.reshape(bias_f, (1, HIDDEN))
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    affine_bf = ct.astype(affine, ct.bfloat16)
    ct.store(final_out_ptr, index=(row, 0), tile=affine_bf)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="a587b5e7")
def oracle_forward(inputs):
    (
        token_table,
        token_ids,
        position_table,
        weight,
        bias,
        _position_view_shape,
        view_shape0,
        view_shape1,
        view_shape2,
    ) = inputs
    batch = int(token_ids.shape[0])
    seq_len = int(token_ids.shape[1])
    hidden = int(token_table.shape[1])
    rows = batch * seq_len
    base_shape = (batch, seq_len, hidden)
    base_stride = (seq_len * hidden, hidden, 1)

    # Precompute the scaled token embedding and position embedding via torch.
    # Triton kernel's forward path also does an in-kernel gather; we do it
    # outside since embeddings tables are the primary source of latency here.
    token_emb = torch.ops.aten.embedding.default(token_table, token_ids, 1)
    scaled = torch.ops.aten.mul.Tensor(token_emb, EMBED_SCALE)  # bf16 mul rounds

    iota = torch.arange(seq_len, dtype=torch.int64, device=token_ids.device) + POSITION_OFFSET
    position = position_table[iota]  # [seq_len, hidden]
    # Broadcast across batch. The Triton kernel computes for each row using
    # (row // SEQ_LEN, row % SEQ_LEN) so position is (batch, seq_len, hidden).
    position_full = position.unsqueeze(0).expand(batch, seq_len, hidden).contiguous()

    add_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        base_shape,
        base_stride,
        device=token_table.device,
        dtype=torch.bfloat16,
    )

    scaled_2d = scaled.view(rows, hidden)
    position_2d = position_full.view(rows, hidden)
    add_out_2d = add_out.view(rows, hidden)
    final_out_2d = final_out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_ln_kernel,
        (
            scaled_2d,
            position_2d,
            weight,
            bias,
            add_out_2d,
            final_out_2d,
            hidden,
            1.0 / hidden,
        ),
    )
    return (
        add_out,
        final_out.view(_as_shape(view_shape0)),
        final_out.view(_as_shape(view_shape1)),
        final_out.view(_as_shape(view_shape2)),
    )
