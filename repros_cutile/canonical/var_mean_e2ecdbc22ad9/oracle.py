"""cuTile port of var_mean_e2ecdbc22ad9: bf16 flat + f32 residual LayerNorm training.

Per row: x = residual + flat (fp32), stored to add_out; LayerNorm, stored to
norm_out and final_out (bf16), plus invstd/HIDDEN scalar per row.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_training_kernel(
    flat_bf16_ptr,   # bf16 [rows, HIDDEN]
    residual_ptr,    # f32  [rows, HIDDEN]
    weight_ptr,      # f32  [HIDDEN]
    bias_ptr,        # f32  [HIDDEN]
    add_out_ptr,     # f32  [rows, HIDDEN]
    norm_out_ptr,    # f32  [rows, HIDDEN]
    final_out_ptr,   # bf16 [rows, HIDDEN]
    invstd_div_ptr,  # f32  [rows]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    INV_HIDDEN: ct.Constant[float],
):
    row = ct.bid(0)

    flat = ct.load(flat_bf16_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(residual, ct.float32) + ct.astype(flat, ct.float32)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    valid_1d = cols < HIDDEN
    valid = ct.reshape(valid_1d, (1, BLOCK_H))
    zero_f = ct.zeros((1, BLOCK_H), dtype=ct.float32)
    zero_bf = ct.zeros((1, BLOCK_H), dtype=ct.bfloat16)

    x_masked = ct.where(valid, x, zero_f)
    mean = ct.sum(x_masked) * INV_HIDDEN
    centered = x - mean
    centered_masked = ct.where(valid, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) * INV_HIDDEN
    invstd = ct.rsqrt(variance + 1.0e-6)
    norm = centered * invstd

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    final = norm * weight_2d + bias_2d
    final_bf = ct.astype(final, ct.bfloat16)

    if BLOCK_H == HIDDEN:
        ct.store(add_out_ptr, index=(row, 0), tile=x)
        ct.store(norm_out_ptr, index=(row, 0), tile=norm)
        ct.store(final_out_ptr, index=(row, 0), tile=final_bf)
    else:
        # Use masked scatter for OOB
        rows_2d = ct.full((1, BLOCK_H), row, dtype=ct.int32)
        cols_2d = ct.reshape(cols, (1, BLOCK_H))
        ct.scatter(add_out_ptr, (rows_2d, cols_2d), x, mask=valid)
        ct.scatter(norm_out_ptr, (rows_2d, cols_2d), norm, mask=valid)
        ct.scatter(final_out_ptr, (rows_2d, cols_2d), final_bf, mask=valid)

    # Store invstd / HIDDEN (Triton hardcodes 192, but that's shape-specific;
    # use general HIDDEN divisor which is what the eager reference computes).
    ct.store(invstd_div_ptr, index=(row,),
             tile=ct.reshape(invstd / HIDDEN, (1,)))


@oracle_impl(hardware="B200", point="0ff22f63", BLOCK_H=256)
@oracle_impl(hardware="B200", point="f13eb73e", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="7b097b88", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="5d43e450", BLOCK_H=2048)
def oracle_forward(inputs, *, BLOCK_H):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    bsz = int(arg1_1.shape[0])
    tokens = int(arg1_1.shape[1])

    add_out = torch.empty_like(arg1_1)
    norm_out = torch.empty_like(arg1_1)
    final_out = torch.empty_like(arg0_1)
    invstd_div = torch.empty(
        (bsz, tokens, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    add_out_2d = add_out.view(rows, hidden)
    norm_out_2d = norm_out.view(rows, hidden)
    residual_2d = arg1_1.view(rows, hidden)
    invstd_1d = invstd_div.view(rows)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_training_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, add_out_2d, norm_out_2d,
         final_out, invstd_1d, hidden, BLOCK_H, 1.0 / hidden),
    )
    return add_out, norm_out, final_out.view(tuple(_shape_param_1)), invstd_div
