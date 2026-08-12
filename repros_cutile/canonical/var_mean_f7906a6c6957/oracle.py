"""cuTile port of var_mean_f7906a6c6957: static LayerNorm forward (hidden=512, bf16 out).

Multi-row: BLOCK_M rows per block to match Triton.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _layernorm_forward_kernel(
    x_ptr,           # bf16 [rows, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    out_ptr,         # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    block = ct.bid(0)
    x = ct.load(x_ptr, index=(block, 0), shape=(BLOCK_M, BLOCK_H))
    x_f = ct.astype(x, ct.float32)
    mean = ct.sum(x_f, axis=1, keepdims=True) * (1.0 / HIDDEN_F)
    centered = x_f - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * (1.0 / HIDDEN_F)
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    out = centered * invstd * weight_2d
    ct.store(out_ptr, index=(block, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="e5ae55b5", BLOCK_M=8, BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    x, weight = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    out = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _layernorm_forward_kernel,
        (x, weight, out, hidden, BLOCK_M, BLOCK_H, float(hidden)),
    )
    return out
