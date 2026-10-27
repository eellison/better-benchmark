"""cuTile port of var_mean_813b85fdc01b: DeiT class-token LayerNorm.

For each batch, compute LayerNorm over token 0's 192 hidden dims and store
result. HIDDEN=192 is not power of 2, so we load a BLOCK_H=256 tile with
zero-padding and mask sums with a range mask.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 197
HIDDEN = 192
EPS = 1.0e-6


@ct.kernel
def _class_token_layernorm_kernel(
    flat_ptr,       # bf16 [BATCH, TOKENS, HIDDEN]
    residual_ptr,   # bf16 [BATCH, TOKENS, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    out_ptr,        # bf16 [BATCH, HIDDEN]
    BLOCK_H: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
):
    batch = ct.bid(0)

    # Load token 0 (index=(batch, 0, 0)), tile shape (1, 1, BLOCK_H)
    flat = ct.load(flat_ptr, index=(batch, 0, 0), shape=(1, 1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(batch, 0, 0), shape=(1, 1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(flat, ct.float32) + ct.astype(residual, ct.float32)
    inv_h = 1.0 / HIDDEN_C

    # Mask out cols >= HIDDEN using column index arange.
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = cols < HIDDEN_C
    col_mask_3d = ct.reshape(col_mask, (1, 1, BLOCK_H))
    zero = ct.full((1, 1, BLOCK_H), 0.0, dtype=ct.float32)
    x_for_reduce = ct.where(col_mask_3d, x_f, zero)
    mean = ct.sum(x_for_reduce) * inv_h
    centered = x_f - mean
    centered_masked = ct.where(col_mask_3d, centered, zero)
    variance = ct.sum(centered_masked * centered_masked) * inv_h
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_3d = ct.reshape(ct.astype(weight, ct.float32), (1, 1, BLOCK_H))
    bias_3d = ct.reshape(ct.astype(bias, ct.float32), (1, 1, BLOCK_H))

    y = centered * invstd * weight_3d + bias_3d
    y_bf = ct.astype(y, ct.bfloat16)
    y_2d = ct.reshape(y_bf, (1, BLOCK_H))
    ct.store(out_ptr, index=(batch, 0), tile=y_2d)


@oracle_impl(hardware="B200", point="ad6d6241", BLOCK_B=4, BLOCK_H=256)
def oracle_forward(inputs, *, BLOCK_B, BLOCK_H):
    flat, residual, weight, bias, _shape0 = inputs
    # Allocate a padded output of shape (BATCH, BLOCK_H) then slice back to
    # (BATCH, HIDDEN) so the kernel's full-tile store is safe.
    padded = torch.empty(
        (BATCH, BLOCK_H),
        device=flat.device,
        dtype=torch.bfloat16,
    )
    flat3 = flat.view(BATCH, TOKENS, HIDDEN)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, 1, 1),
        _class_token_layernorm_kernel,
        (flat3, residual, weight, bias, padded, BLOCK_H, HIDDEN),
    )
    # Extract logical [BATCH, HIDDEN].
    out = padded[:, :HIDDEN].contiguous()
    return out
