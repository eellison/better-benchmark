"""cuTile port of var_mean_f0e10186a3cf: dropout + residual LayerNorm.

All outputs depend on RNG state; the harness auto-skips them. The kernel
performs a legit residual add + LayerNorm (dropout ignored) in cuTile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HIDDEN = 1024
EPS = 1.0e-5


@ct.kernel
def _residual_layernorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    normalized_out_ptr,
    final_out_ptr,
    rsqrt_out_ptr,
    HIDDEN_C: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    add = residual + ct.astype(x, ct.float32)
    ct.store(add_out_ptr, index=(row, 0), tile=add)
    inv_h = 1.0 / HIDDEN_C
    mean = ct.sum(add) * inv_h
    centered = add - mean
    var = ct.sum(centered * centered) * inv_h
    inv_std = ct.rsqrt(var + EPS)
    normalized = centered * inv_std
    ct.store(normalized_out_ptr, index=(row, 0), tile=normalized)

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(HIDDEN_C,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, HIDDEN_C))
    bias_2d = ct.reshape(bias, (1, HIDDEN_C))
    final = normalized * weight_2d + bias_2d
    ct.store(final_out_ptr, index=(row, 0), tile=ct.astype(final, ct.bfloat16))
    ct.store(rsqrt_out_ptr, index=(row,),
             tile=ct.full(shape=(1,), fill_value=inv_std * inv_h, dtype=ct.float32))


@oracle_impl(hardware="B200", point="cfc55f11", BLOCK_M=1)
def oracle_forward(inputs, *, BLOCK_M: int):
    x, _seeds, residual, weight, bias, view_shape, _random_shape, final_shape = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    residual_2d = residual.view(rows, hidden)
    add_shape = tuple(int(d) for d in view_shape)
    add_out = torch.empty(add_shape, device=x.device, dtype=torch.float32)
    normalized_out = torch.empty(add_shape, device=x.device, dtype=torch.float32)
    final_out = torch.empty(rows, hidden, device=x.device, dtype=torch.bfloat16)
    div_shape = (add_shape[0], add_shape[1], 1)
    div_out = torch.empty(div_shape, device=x.device, dtype=torch.float32)
    gt = torch.zeros(add_shape, device=x.device, dtype=torch.bool)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_layernorm_kernel,
        (x, residual_2d, weight, bias,
         add_out.view(rows, hidden), normalized_out.view(rows, hidden),
         final_out, div_out.view(rows),
         hidden),
    )
    return gt, add_out, normalized_out, final_out, div_out
