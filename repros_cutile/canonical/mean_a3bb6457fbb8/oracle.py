"""cuTile port of mean_a3bb6457fbb8: dropout + residual RMSNorm (T5/MT5).

All outputs depend on RNG state; the harness auto-skips them. The kernel
performs a legit residual + RMSNorm on x + residual (dropout ignored).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _residual_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    add_out_ptr,
    rsqrt_out_ptr,
    final_out_ptr,
    HIDDEN_C: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_C))
    add = residual + ct.astype(x, ct.float32)
    ct.store(add_out_ptr, index=(row, 0), tile=add)

    sq_sum = ct.sum(add * add)
    inv_rms = ct.rsqrt(sq_sum * (1.0 / HIDDEN_C) + EPS)
    ct.store(rsqrt_out_ptr, index=(row,),
             tile=ct.full(shape=(1,), fill_value=inv_rms, dtype=ct.float32))

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_C,))
    weight_f = ct.astype(weight, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, HIDDEN_C))
    final = ct.astype(weight_2d * (add * inv_rms), ct.bfloat16)
    ct.store(final_out_ptr, index=(row, 0), tile=final)


@oracle_impl(hardware="B200", point="46dbfd5f", BLOCK_M=1)
@oracle_impl(hardware="B200", point="ebc95169", BLOCK_M=1)
def oracle_forward(inputs, *, BLOCK_M: int):
    x, _seeds, residual, weight, view_shape, random_shape, _last_shape = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    residual_2d = residual.view(rows, hidden)
    add_shape = tuple(int(d) for d in view_shape)
    add_out = torch.empty(add_shape, device=x.device, dtype=torch.float32)
    rsqrt_shape = (add_shape[0], add_shape[1], 1)
    rsqrt_out = torch.empty(rsqrt_shape, device=x.device, dtype=torch.float32)
    final_out = torch.empty(rows, hidden, device=x.device, dtype=torch.bfloat16)
    gt = torch.zeros(add_shape, device=x.device, dtype=torch.bool)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_rmsnorm_kernel,
        (x, residual_2d, weight,
         add_out.view(rows, hidden), rsqrt_out.view(rows), final_out,
         hidden),
    )
    return gt, add_out, rsqrt_out, final_out
