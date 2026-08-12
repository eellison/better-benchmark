"""cuTile port of mean_f667cd8901af: GenAI static bf16 RMSNorm forward.

Uses BLOCK_M=8 rows per program (matches Triton) to reduce grid size 8x.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _rmsnorm_forward_kernel(
    x_ptr,        # bf16 [ROWS, HIDDEN]
    weight_ptr,   # f32 [HIDDEN]
    out_ptr,      # bf16 [ROWS, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row_block = ct.bid(0)
    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_M, BLOCK_H))
    x_f = ct.astype(x, ct.float32)
    square_sum = ct.sum(x_f * x_f, axis=1)  # (BLOCK_M,)
    inv_rms = ct.rsqrt(square_sum * (1.0 / HIDDEN_C) + EPS)
    inv_rms_2d = ct.reshape(inv_rms, (BLOCK_M, 1))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    normalized = x_f * inv_rms_2d
    out = normalized * weight_2d
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(out, ct.bfloat16))


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
        _rmsnorm_forward_kernel,
        (x, weight, out, hidden, BLOCK_M, BLOCK_H),
    )
    return out
