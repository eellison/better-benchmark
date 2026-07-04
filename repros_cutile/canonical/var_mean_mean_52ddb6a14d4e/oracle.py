"""cuTile port of var_mean_mean_52ddb6a14d4e: Swin residual-add + LayerNorm +
spatial mean [B, HW, C] -> [B, C].

Two cuTile kernels: (1) residual bf16 add + LayerNorm affine emitting bf16
per-row output; (2) spatial mean over 49 tokens per row producing bf16 [B, C]
output. Hidden is 1024, tokens per batch is 49.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
HEIGHT = 7
WIDTH = 7
TOKENS = HEIGHT * WIDTH  # 49
ROWS = BATCH * TOKENS    # 6272
HIDDEN = 1024
EPS = 1.0e-5


@ct.kernel
def _residual_layernorm_kernel(
    flat_ptr,       # bf16 [ROWS, HIDDEN]
    residual_ptr,   # bf16 [ROWS, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    norm_ptr,       # bf16 [ROWS, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    # bf16 boundary matches the Triton oracle: cast the residual-add sum back
    # to bf16 before consuming.
    added_bf = ct.astype(resid_f + flat_f, ct.bfloat16)
    x = ct.astype(added_bf, ct.float32)

    inv_h = 1.0 / HIDDEN_
    mean = ct.sum(x) * inv_h
    centered = x - mean
    variance = ct.sum(centered * centered) * inv_h
    invstd = ct.rsqrt(variance + 1.0e-5)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    w_f = ct.astype(weight, ct.float32)
    b_f = ct.astype(bias, ct.float32)
    w_2d = ct.reshape(w_f, (1, BLOCK_H))
    b_2d = ct.reshape(b_f, (1, BLOCK_H))
    y = centered * invstd * w_2d + b_2d
    ct.store(norm_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


@ct.kernel
def _spatial_mean_kernel(
    norm_ptr,       # bf16 [BATCH, TOKENS, HIDDEN]
    out_ptr,        # bf16 [BATCH, HIDDEN]
    TOKENS_: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_T: ct.Constant[int],
):
    batch = ct.bid(0)
    channel_block = ct.bid(1)

    # Load a (1, BLOCK_T, BLOCK_C) tile. Index is in tile units, so the
    # element origin is (batch*1, 0*BLOCK_T, channel_block*BLOCK_C).
    vals = ct.load(
        norm_ptr,
        index=(batch, 0, channel_block),
        shape=(1, BLOCK_T, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    vals_f = ct.astype(vals, ct.float32)
    # Sum over the tokens dimension (axis=1).
    total = ct.sum(vals_f, axis=1, keepdims=False)  # -> (1, BLOCK_C)
    mean = total * (1.0 / TOKENS_)
    mean_bf = ct.astype(mean, ct.bfloat16)
    ct.store(out_ptr, index=(batch, channel_block), tile=mean_bf)


@oracle_impl(
    hardware="B200",
    point="1a2bb10a",
    BLOCK_H=1024,
    BLOCK_C=32,
    BLOCK_T=64,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_H: int,
    BLOCK_C: int,
    BLOCK_T: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1 = inputs
    device = arg0_1.device

    norm = torch.empty_strided(
        (BATCH, TOKENS, HIDDEN),
        (TOKENS * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out = torch.empty(
        (BATCH, HIDDEN),
        device=device,
        dtype=torch.bfloat16,
    )

    flat_2d = arg0_1.contiguous().view(ROWS, HIDDEN)
    residual_2d = arg1_1.contiguous().view(ROWS, HIDDEN)
    norm_2d = norm.view(ROWS, HIDDEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _residual_layernorm_kernel,
        (flat_2d, residual_2d, arg2_1, arg3_1, norm_2d,
         HIDDEN, BLOCK_H),
    )
    ct.launch(
        stream,
        (BATCH, HIDDEN // BLOCK_C, 1),
        _spatial_mean_kernel,
        (norm, out, TOKENS, BLOCK_C, BLOCK_T),
    )
    return out
