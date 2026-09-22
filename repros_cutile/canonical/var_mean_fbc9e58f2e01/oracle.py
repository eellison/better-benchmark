"""cuTile port of var_mean_fbc9e58f2e01: OPT token+position embedding + LayerNorm.

For each row: embedding gather (token + position with (pos-1).to(int64)+2),
bf16 add, fp32 var+mean, rsqrt+eps, affine bf16 store. HIDDEN=768 padded to 1024.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8192
HIDDEN = 768
SEQ_LEN = 2048


@ct.kernel
def _opt_embedding_ln_kernel(
    token_table_ptr,  # bf16 [50272, HIDDEN]
    token_ids_ptr,    # i64 [ROWS]
    position_ids_ptr, # f32 [ROWS]
    position_table_ptr,  # bf16 [2050, HIDDEN]
    weight_ptr,       # bf16 [HIDDEN]
    bias_ptr,         # bf16 [HIDDEN]
    add_out_ptr,      # bf16 [ROWS, HIDDEN]
    norm_out_ptr,     # bf16 [ROWS, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    token_id_tile = ct.load(token_ids_ptr, index=(row,), shape=(1,))
    position_f32 = ct.load(position_ids_ptr, index=(row,), shape=(1,))
    position_id = ct.astype(position_f32 - 1.0, ct.int64) + ct.full(shape=(1,), fill_value=2, dtype=ct.int64)

    # Prepare column indices [0..BLOCK_H)
    cols = ct.arange(BLOCK_H, dtype=ct.int64)
    col_mask = cols < HIDDEN_C

    # Broadcast token_id and cols to (1, BLOCK_H) for gather
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

    x_fp32 = token + position
    x_bf16 = ct.astype(x_fp32, ct.bfloat16)
    ct.store(add_out_ptr, index=(row, 0), tile=x_bf16)
    x = ct.astype(x_bf16, ct.float32)

    x_masked = ct.where(col_mask_2d, x, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN_C)
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_C)
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
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="801db743", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, *_shape_params = inputs
    add_out = torch.empty_strided(
        (4, SEQ_LEN, HIDDEN),
        (SEQ_LEN * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    add_out_2d = add_out.view(ROWS, HIDDEN)
    token_ids_flat = arg1_1.view(-1)
    position_ids_flat = arg2_1.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _opt_embedding_ln_kernel,
        (arg0_1, token_ids_flat, position_ids_flat, arg3_1, arg4_1, arg5_1,
         add_out_2d, norm_out, HIDDEN, BLOCK_H),
    )
    return add_out, norm_out, norm_out, norm_out
