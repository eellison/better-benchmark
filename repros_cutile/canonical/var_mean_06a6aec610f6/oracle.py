"""cuTile port of var_mean_06a6aec610f6: DINOv2 class-token LayerNorm.

Only the token-0 row of each batch is materialized as the output. For each
batch, one program: load source row (bf16), scale + bf16 round, add residual +
bf16 round, LayerNorm (fp32 var_mean, eps=1e-6), affine (weight, bias), bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
TOKENS = 1370
HIDDEN = 768
EPS = 1.0e-6


@ct.kernel
def _class_token_layernorm_kernel(
    source_ptr,    # bf16 [BATCH*TOKENS, HIDDEN]
    scale_ptr,     # bf16 [HIDDEN]
    residual_ptr,  # bf16 [BATCH*TOKENS, HIDDEN]
    weight_ptr,    # bf16 [HIDDEN]
    bias_ptr,      # bf16 [HIDDEN]
    out_ptr,       # bf16 [BATCH, HIDDEN]
    BLOCK_H: ct.Constant[int],
):
    batch = ct.bid(0)
    # Source and residual are indexed at row = batch * TOKENS.
    src_row = batch * TOKENS
    src = ct.load(source_ptr, index=(src_row, 0), shape=(1, BLOCK_H))
    resid = ct.load(residual_ptr, index=(src_row, 0), shape=(1, BLOCK_H))
    scale = ct.load(scale_ptr, index=(0,), shape=(BLOCK_H,))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))

    src_f = ct.astype(src, ct.float32)
    resid_f = ct.astype(resid, ct.float32)
    scale_f = ct.reshape(ct.astype(scale, ct.float32), (1, BLOCK_H))
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_H))
    bias_f = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_H))

    scaled_bf = ct.astype(src_f * scale_f, ct.bfloat16)
    scaled_f = ct.astype(scaled_bf, ct.float32)
    x_bf = ct.astype(resid_f + scaled_f, ct.bfloat16)
    x = ct.astype(x_bf, ct.float32)

    mean = ct.sum(x, axis=1, keepdims=True) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    norm = centered * invstd
    affine = norm * weight_f + bias_f
    ct.store(out_ptr, index=(batch, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="01a8f9c7", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    source, scale, residual, weight, bias, _shape0 = inputs
    # source: bf16[175360, 768] (viewed as [BATCH*TOKENS, HIDDEN])
    # residual: bf16[BATCH, TOKENS, HIDDEN]
    out = torch.empty_strided(
        (BATCH, HIDDEN),
        (HIDDEN, 1),
        device=source.device,
        dtype=torch.bfloat16,
    )
    residual_2d = residual.view(BATCH * TOKENS, HIDDEN)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, 1, 1),
        _class_token_layernorm_kernel,
        (source, scale, residual_2d, weight, bias, out, BLOCK_H),
    )
    return out
