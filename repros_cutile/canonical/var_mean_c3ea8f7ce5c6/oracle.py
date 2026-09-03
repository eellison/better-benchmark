"""cuTile port of var_mean_c3ea8f7ce5c6: residual-add LayerNorm with alias views."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _residual_layernorm_kernel(
    flat_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)

    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))

    flat_f = ct.astype(flat, ct.float32)
    resid_f = ct.astype(residual, ct.float32)
    add_bf16 = ct.astype(flat_f + resid_f, ct.bfloat16)
    x = ct.astype(add_bf16, ct.float32)

    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    var = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(var + EPS)

    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    y = ct.astype(centered * invstd * weight_2d + bias_2d, ct.bfloat16)

    ct.store(out_ptr, index=(row, 0), tile=y)


@oracle_impl(hardware="B200", point="0b3dc49f", ROWS=8192, BATCH=8, SEQ=1024, HIDDEN=1024, BLOCK_H=1024)
@oracle_impl(hardware="B200", point="d1f40ce2", ROWS=16384, BATCH=16, SEQ=1024, HIDDEN=768, BLOCK_H=1024)
@oracle_impl(hardware="B200", point="d4cc3e3e", ROWS=16384, BATCH=64, SEQ=256, HIDDEN=1024, BLOCK_H=1024)
def oracle_forward(inputs, *, ROWS, BATCH, SEQ, HIDDEN, BLOCK_H):
    flat, residual, weight, bias = inputs[:4]
    base = torch.empty_strided(
        (BATCH, SEQ, HIDDEN),
        (SEQ * HIDDEN, HIDDEN, 1),
        device=flat.device,
        dtype=torch.bfloat16,
    )
    # residual is [B, S, H] contiguous; view flat as [rows, H]
    residual_2d = residual.view(ROWS, HIDDEN)
    base_2d = base.view(ROWS, HIDDEN)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _residual_layernorm_kernel,
        (flat, residual_2d, weight, bias, base_2d, HIDDEN, BLOCK_H),
    )
    view = base.view(ROWS, HIDDEN)
    return base, view, view, view
