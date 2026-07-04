"""cuTile port of var_mean_2bd313f3c17f: Blenderbot residual-add LayerNorm (H=2560).

For each row: bf16 add flat + residual (bf16 rounded), fp32 var_mean(dim=2, corr=0),
rsqrt with 1e-5, bf16 affine output. HIDDEN=2560 needs BLOCK_H=4096 with zero
padding, and a mask to zero out cols >= HIDDEN in the reductions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 2560
EPS = 1.0e-5
BLOCK_H = 4096


@ct.kernel
def _residual_layernorm_h2560_kernel(
    flat_ptr,      # bf16 [ROWS, HIDDEN]
    residual_ptr,  # bf16 [ROWS, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    out_ptr,       # bf16 [ROWS, BLOCK_H] (padded)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H_C),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H_C),
                       padding_mode=ct.PaddingMode.ZERO)
    added = ct.astype(ct.astype(flat, ct.float32) + ct.astype(residual, ct.float32), ct.bfloat16)
    x = ct.astype(added, ct.float32)

    # Mask out cols >= HIDDEN
    cols = ct.arange(BLOCK_H_C, dtype=ct.int32)
    col_mask = cols < HIDDEN_C
    col_mask_2d = ct.reshape(col_mask, (1, BLOCK_H_C))
    zero = ct.full((1, BLOCK_H_C), 0.0, dtype=ct.float32)

    inv_h = 1.0 / HIDDEN_C
    x_masked = ct.where(col_mask_2d, x, zero)
    mean = ct.sum(x_masked) * inv_h
    centered = x - mean
    centered_masked = ct.where(col_mask_2d, centered, zero)
    variance = ct.sum(centered_masked * centered_masked) * inv_h
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H_C))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H_C))
    y = centered * invstd * weight_2d + bias_2d
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="7f824027", ROW_BLOCK=2)
def oracle_forward(inputs, *, ROW_BLOCK):
    flat, residual, weight, bias, _input_shape, *output_shapes = inputs

    # Padded output buffer with BLOCK_H columns for full-tile stores; slice back.
    padded = torch.empty(
        (ROWS, BLOCK_H),
        device=flat.device,
        dtype=torch.bfloat16,
    )
    residual_2d = residual.view(ROWS, HIDDEN)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _residual_layernorm_h2560_kernel,
        (flat, residual_2d, weight, bias, padded, HIDDEN, BLOCK_H),
    )
    out_flat = padded[:, :HIDDEN].contiguous()
    out = out_flat.view(BATCH, SEQ_LEN, HIDDEN)
    return (out, *(out.view(_shape_tuple(shape)) for shape in output_shapes))
