"""cuTile port of var_mean_63cad2383953: Whisper-tiny decoder token +
position embedding LayerNorm.

Steps:
1. Look up token vector using PyTorch (dynamic index lookup outside kernel).
2. Copy position row 0 outside kernel.
3. cuTile kernel: add embeddings, compute var/mean, normalize, affine.
4. Return the pre-norm add plus three views of the normalized output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _embed_layernorm_kernel(
    token_row_ptr,     # bf16 (1, HIDDEN) — pre-gathered token embed row
    position_row_ptr,  # bf16 (1, HIDDEN) — position embed row 0
    weight_ptr,        # bf16 (HIDDEN,)
    bias_ptr,          # bf16 (HIDDEN,)
    add_out_ptr,       # bf16 (1, HIDDEN)
    norm_out_ptr,      # bf16 (1, HIDDEN)
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    tok = ct.load(
        token_row_ptr, index=(0, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pos = ct.load(
        position_row_ptr, index=(0, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tok_f = ct.astype(tok, ct.float32)
    pos_f = ct.astype(pos, ct.float32)
    added_f = tok_f + pos_f
    added_bf16 = ct.astype(added_f, ct.bfloat16)
    # Store to add_out — OOB elements past HIDDEN are ignored by cuTile.
    ct.store(add_out_ptr, index=(0, 0), tile=added_bf16)

    # Mask out OOB columns before reduction so they don't affect sums.
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))
    x = ct.where(col_mask_2d, added_f, 0.0)

    total = ct.sum(x)
    mean = total * (1.0 / HIDDEN)
    centered = ct.where(col_mask_2d, x - mean, 0.0)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(0, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="a5e05174")
def oracle_forward(inputs):
    token_table, token_ids, position_table, weight, bias, shape0, shape1, shape2 = inputs
    hidden = int(token_table.shape[1])
    device = token_table.device

    # Gather the token embedding via PyTorch (matches aten.embedding).
    token_idx = int(token_ids.view(-1)[0].item())
    token_row = token_table[token_idx:token_idx + 1].contiguous()  # (1, HIDDEN)
    position_row = position_table[0:1].contiguous()  # (1, HIDDEN)

    add_out = torch.empty_strided(
        (1, 1, hidden),
        (hidden, hidden, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        (1, 1, hidden),
        (hidden, hidden, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    # View as (1, HIDDEN) for the kernel (contiguous).
    add_out_2d = add_out.view(1, hidden)
    norm_base_2d = norm_base.view(1, hidden)

    block_h = 1 << (hidden - 1).bit_length()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _embed_layernorm_kernel,
        (token_row, position_row, weight, bias, add_out_2d, norm_base_2d, hidden, block_h),
    )
    return (
        add_out,
        norm_base.view(_as_shape(shape0)),
        norm_base.view(_as_shape(shape1)),
        norm_base.view(_as_shape(shape2)),
    )
