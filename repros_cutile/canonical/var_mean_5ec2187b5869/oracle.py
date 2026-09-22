"""cuTile port of var_mean_5ec2187b5869: GPT-2 token+position embedding + LayerNorm.

Per row: gather token+position embeddings, bf16 add, fp32 var+mean over HIDDEN,
rsqrt+eps, affine bf16 output, and a constant-False side output. HIDDEN=768 padded to 1024.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _gpt2_embedding_ln_kernel(
    token_table_ptr,  # bf16 [50257, HIDDEN]
    token_ids_ptr,    # i64 [ROWS]
    position_table_ptr,  # bf16 [1024, HIDDEN]
    weight_ptr,       # bf16 [HIDDEN]
    bias_ptr,         # bf16 [HIDDEN]
    add_out_ptr,      # bf16 [ROWS, HIDDEN]
    norm_out_ptr,     # bf16 [ROWS, HIDDEN]
    SEQ_LEN: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    token_id_tile = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    # position_id = row % SEQ_LEN
    row_tile = ct.full(shape=(1,), fill_value=row, dtype=ct.int64)
    position_id = row_tile - (row_tile // SEQ_LEN) * SEQ_LEN

    cols = ct.arange(BLOCK_H, dtype=ct.int64)
    col_mask = cols < HIDDEN

    tid_2d = ct.reshape(token_id_tile, (1, 1))
    pid_2d = ct.reshape(position_id, (1, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_H))
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H))

    token = ct.astype(
        ct.gather(token_table_ptr, (tid_2d, cols_2d), mask=col_mask_2d, padding_value=ct.bfloat16(0)),
        ct.float32,
    )
    position = ct.astype(
        ct.gather(position_table_ptr, (pid_2d, cols_2d), mask=col_mask_2d, padding_value=ct.bfloat16(0)),
        ct.float32,
    )

    add_f = token + position
    add_bf16 = ct.astype(add_f, ct.bfloat16)
    # Scatter add_out: HIDDEN=768 but stride matches HIDDEN so [row*HIDDEN + col]
    col_idx = ct.arange(BLOCK_H, dtype=ct.int64)
    row_idx = ct.full(shape=(BLOCK_H,), fill_value=row, dtype=ct.int64)
    ct.scatter(
        add_out_ptr, (row_idx, col_idx),
        ct.reshape(add_bf16, (BLOCK_H,)),
        mask=col_mask,
    )

    x = ct.astype(add_bf16, ct.float32)
    x_masked = ct.where(col_mask_2d, x, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = centered * invstd * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    ct.scatter(
        norm_out_ptr, (row_idx, col_idx),
        ct.reshape(affine_bf16, (BLOCK_H,)),
        mask=col_mask,
    )


def _as_shape(shape):
    out = []
    known = 1
    unknown = -1
    for index, dim in enumerate(shape):
        value = int(dim)
        out.append(value)
        if value == -1:
            unknown = index
        else:
            known *= value
    if unknown >= 0:
        out[unknown] = 8192 * 768 // known
    return tuple(out)


@oracle_impl(hardware="B200", point="3b387564", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, shape0, shape1, _shape2 = inputs
    batch = int(shape0[0])
    seq_len = int(arg1_1.shape[1])
    hidden = int(arg3_1.shape[0])
    rows = int(arg1_1.numel())
    norm_shape = _as_shape(shape1)

    add_out = torch.empty_strided(
        (batch, seq_len, hidden),
        (seq_len * hidden, hidden, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape, (hidden, 1), device=arg0_1.device, dtype=torch.bfloat16
    )

    add_out_2d = add_out.view(rows, hidden)
    token_ids_flat = arg1_1.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _gpt2_embedding_ln_kernel,
        (arg0_1, token_ids_flat, arg2_1, arg3_1, arg4_1, add_out_2d, norm_out,
         seq_len, hidden, BLOCK_H),
    )
    # mask output is constant False
    mask_out = torch.zeros((batch, seq_len), device=arg0_1.device, dtype=torch.bool)
    return add_out, norm_out, mask_out
