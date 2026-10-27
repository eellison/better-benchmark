"""cuTile port of var_mean_bb07b717c0ae: Longformer bf16 bias+residual LayerNorm.

Per row: (bf16(flat+bias)+bf16(residual)) -> bf16 -> f32 x; pop var_mean;
rsqrt(var+1e-5); affine; bf16 output; return (out, flattened view).
HIDDEN=768 is not pow2 — use BLOCK_H=1024 with padding_mode=ZERO on loads
and scatter with mask on the store to avoid OOB writes.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
HIDDEN = 768


@ct.kernel
def _bf16_residual_layernorm_kernel(
    flat_ptr,          # bf16 (rows, HIDDEN)
    bias_ptr,          # bf16 (HIDDEN,)
    residual_ptr,      # bf16 (rows, HIDDEN)
    weight_ptr,        # bf16 (HIDDEN,)
    affine_bias_ptr,   # bf16 (HIDDEN,)
    out_ptr,           # bf16 (rows, HIDDEN)
    HIDDEN_C: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    flat_f = ct.astype(flat, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    add0 = ct.astype(flat_f + bias_2d, ct.bfloat16)
    x = ct.astype(ct.astype(ct.astype(add0, ct.float32) + resid_f, ct.bfloat16), ct.float32)

    # Mask padding for the reductions.
    idx = ct.arange(BLOCK_H, dtype=ct.int32)
    idx_2d = ct.reshape(idx, (1, BLOCK_H))
    mask = idx_2d < HIDDEN_C
    zero_tile = ct.zeros(shape=(1, BLOCK_H), dtype=ct.float32)
    x_for_reduce = ct.where(mask, x, zero_tile)

    mean = ct.sum(x_for_reduce) * (1.0 / HIDDEN_C)
    centered_all = x - mean
    centered = ct.where(mask, centered_all, zero_tile)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_C)
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    affine_bias = ct.load(affine_bias_ptr, index=(0,), shape=(BLOCK_H,),
                          padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    affine_bias_f = ct.astype(affine_bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    affine_bias_2d = ct.reshape(affine_bias_f, (1, BLOCK_H))
    out = centered_all * invstd * weight_2d + affine_bias_2d
    out_bf16 = ct.astype(out, ct.bfloat16)
    # Scatter with mask to only write in-bounds columns (BLOCK_H=1024 > HIDDEN=768).
    r_idx = ct.full(shape=(1, BLOCK_H), fill_value=row, dtype=ct.int32)
    c_idx = ct.reshape(idx, (1, BLOCK_H))
    ct.scatter(out_ptr, (r_idx, c_idx), out_bf16, mask=mask)


@oracle_impl(hardware="B200", point="432bb161", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _s0, _s1 = inputs

    residual_2d = arg2_1.view(ROWS, HIDDEN)
    out = torch.empty_like(arg2_1)
    out_2d = out.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _bf16_residual_layernorm_kernel,
        (arg0_1, arg1_1, residual_2d, arg3_1, arg4_1, out_2d, HIDDEN, BLOCK_H),
    )
    return out, out.view((ROWS, HIDDEN))
