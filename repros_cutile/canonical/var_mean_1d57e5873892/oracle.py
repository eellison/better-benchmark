"""cuTile port of var_mean_1d57e5873892: BEiT class-token + patch-token cat + LayerNorm.

Mirrors the single-kernel Triton structure: one @ct.kernel that (per row)
gathers either the class token or a patch value, writes the pre-LN cat,
computes fp32 var/mean, applies the LN affine, and writes the bf16 output.
No torch.cat, no separate pad copies for inputs/weights.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


TOKENS = 197
HIDDEN = 768
BATCH = 128
PATCHES = 196
BLOCK_H = 1024  # power-of-2 padding
EPS = 1.0e-6


@ct.kernel
def _cat_ln_kernel(
    class_token_ptr,   # bf16 [HIDDEN]
    patches_ptr,       # bf16 [BATCH, PATCHES, HIDDEN] (metadata-permuted)
    weight_ptr,        # bf16 [HIDDEN]
    bias_ptr,          # bf16 [HIDDEN]
    cat_ptr,           # bf16 [BATCH, TOKENS, BLOCK_H] (padded)
    out_ptr,           # bf16 [BATCH, TOKENS, BLOCK_H] (padded)
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
    EPS_: ct.Constant[float],
):
    batch = ct.bid(0)
    token = ct.bid(1)
    is_class = token == 0
    patch_idx = ct.maximum(token - 1, 0)

    class_val = ct.load(
        class_token_ptr,
        index=(0,),
        shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    patch_val_3d = ct.load(
        patches_ptr,
        index=(batch, patch_idx, 0),
        shape=(1, 1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    patch_val = ct.reshape(patch_val_3d, (BLOCK_H_,))
    x = ct.where(is_class, class_val, patch_val)

    # Write cat (pre-LN)
    ct.store(cat_ptr, index=(batch, token, 0),
             tile=ct.reshape(x, (1, 1, BLOCK_H_)))

    # LN in fp32
    x_f = ct.astype(x, ct.float32)
    cols = ct.arange(BLOCK_H_, dtype=ct.int32)
    valid = cols < HIDDEN_
    zero_f = ct.zeros((BLOCK_H_,), dtype=ct.float32)
    x_masked = ct.where(valid, x_f, zero_f)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN_)
    centered = ct.where(valid, x_f - mean, zero_f)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS_)
    normalized = centered * invstd

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    affine = normalized * weight + bias

    ct.store(
        out_ptr,
        index=(batch, token, 0),
        tile=ct.reshape(ct.astype(affine, ct.bfloat16), (1, 1, BLOCK_H_)),
    )


def _as_shape(s):
    return tuple(int(d) for d in s)


@oracle_impl(hardware="B200", point="8b5672b4", BLOCK_H=BLOCK_H)
def oracle_forward(inputs, *, BLOCK_H: int):
    class_token, patches, weight, bias, _expand_shape, _patch_shape, view_shape = inputs
    device = patches.device

    # Metadata-only reshapes: patches is channels-last, so permute+view are free.
    patches_reshaped = patches.permute(0, 2, 3, 1).reshape(BATCH, PATCHES, HIDDEN)
    class_token_flat = class_token.view(HIDDEN)

    # Padded output buffers; kernel writes (1, 1, BLOCK_H) per (batch, token).
    cat_padded = torch.empty(
        (BATCH, TOKENS, BLOCK_H), device=device, dtype=torch.bfloat16
    )
    out_padded = torch.empty(
        (BATCH, TOKENS, BLOCK_H), device=device, dtype=torch.bfloat16
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, TOKENS, 1),
        _cat_ln_kernel,
        (class_token_flat, patches_reshaped, weight, bias,
         cat_padded, out_padded, HIDDEN, BLOCK_H, EPS),
    )

    # Strided views expose the (unpadded) HIDDEN-wide slice without a copy.
    cat = cat_padded.as_strided(
        (BATCH, TOKENS, HIDDEN),
        (TOKENS * BLOCK_H, BLOCK_H, 1),
    )
    view_1 = out_padded.as_strided(
        _as_shape(view_shape),  # (25216, 768)
        (BLOCK_H, 1),
    )
    return cat, view_1
