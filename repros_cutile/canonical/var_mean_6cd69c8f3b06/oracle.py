"""cuTile port of var_mean_6cd69c8f3b06: Swin singleton-window LayerNorm.

For each row of x [6272, 1024] bf16: compute fp32 population mean/variance,
rsqrt with eps=1e-5, apply weight/bias, cast to bf16. Also return input viewed
as [128, 7, 7, 1024] for the alias sibling.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 6272
HIDDEN = 1024
INV_H = 1.0 / HIDDEN
EPS = 1.0e-5


@ct.kernel
def _swin_ln_kernel(
    x_ptr,       # bf16 [ROWS, HIDDEN]
    weight_ptr,  # bf16 [HIDDEN]
    bias_ptr,    # bf16 [HIDDEN]
    out_ptr,     # bf16 [ROWS, HIDDEN]
    H_: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, H_))
    x_f = ct.astype(x, ct.float32)
    total = ct.sum(x_f)
    mean = total * INV_H
    centered = x_f - mean
    variance = ct.sum(centered * centered) * INV_H
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(H_,))
    bias = ct.load(bias_ptr, index=(0,), shape=(H_,))
    weight_f = ct.reshape(ct.astype(weight, ct.float32), (1, H_))
    bias_f = ct.reshape(ct.astype(bias, ct.float32), (1, H_))
    y = centered * invstd * weight_f + bias_f
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="8b70fd76")
def oracle_forward(inputs):
    x, weight, bias = inputs[:3]
    out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(stream, (ROWS, 1, 1), _swin_ln_kernel,
              (x, weight, bias, out, HIDDEN))
    return x.view(128, 7, 7, 1024), out
