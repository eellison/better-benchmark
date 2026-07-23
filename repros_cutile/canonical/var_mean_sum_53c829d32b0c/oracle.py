"""cuTile port of var_mean_sum_53c829d32b0c: LayerNorm + global-sum.

Per-row LayerNorm over hidden=512, then affine * weight, round to bf16, and
accumulate a row-scalar sum. Final output is the scalar sum over all rows
(rounded to bf16). ROWS=1152000 is not a power of 2; we pad up to the next
multiple of BLOCK_M and do the global reduction in torch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _layernorm_row_kernel(
    x_ptr,          # bf16 [ROWS_PADDED, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    mean_ptr,       # f32 [ROWS_PADDED, 1]
    rsqrt_ptr,      # f32 [ROWS_PADDED, 1]
    row_sum_ptr,    # f32 [ROWS_PADDED, 1]
    HIDDEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    EPS: ct.Constant[float],
    INV_H: ct.Constant[float],
):
    row_block = ct.bid(0)
    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    x_f = ct.astype(x, ct.float32)
    mean = ct.sum(x_f, axis=1, keepdims=True) * INV_H
    centered = x_f - mean
    variance = ct.sum(centered * centered, axis=1, keepdims=True) * INV_H
    invstd = ct.rsqrt(variance + EPS)

    ct.store(mean_ptr, index=(row_block, 0), tile=mean)
    ct.store(rsqrt_ptr, index=(row_block, 0), tile=invstd)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    affine = centered * invstd * weight_2d
    rounded_bf = ct.astype(affine, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf, ct.float32)
    row_sum = ct.sum(rounded_f, axis=1, keepdims=True)
    ct.store(row_sum_ptr, index=(row_block, 0), tile=row_sum)


@oracle_impl(hardware="B200", point="e5ae55b5", BLOCK_M=8, BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_H: int):
    x, weight = inputs
    rows = int(x.shape[0])
    hidden = int(x.shape[1])
    device = x.device

    rows_padded = ((rows + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    if rows_padded != rows:
        pad = torch.zeros((rows_padded - rows, hidden), device=device, dtype=x.dtype)
        x_padded = torch.cat([x, pad], dim=0)
    else:
        x_padded = x

    mean_padded = torch.empty((rows_padded, 1), device=device, dtype=torch.float32)
    rsqrt_padded = torch.empty((rows_padded, 1), device=device, dtype=torch.float32)
    row_sum_padded = torch.empty((rows_padded, 1), device=device, dtype=torch.float32)

    weight_f = weight.to(torch.float32).contiguous().view(hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows_padded // BLOCK_M, 1, 1),
        _layernorm_row_kernel,
        (x_padded, weight_f, mean_padded, rsqrt_padded, row_sum_padded,
         hidden, BLOCK_M, BLOCK_H, 1.0e-6, 1.0 / hidden),
    )

    mean = mean_padded[:rows].contiguous()
    rsqrt = rsqrt_padded[:rows].contiguous()
    total = row_sum_padded[:rows].sum()
    out = total.to(torch.bfloat16)
    return mean, rsqrt, out
