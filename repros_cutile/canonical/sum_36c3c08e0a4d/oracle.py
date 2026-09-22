"""cuTile port of sum_36c3c08e0a4d (SCHEDULER_FUSION): per-column materialize
`a * b * (1 - b)` in fp32, round to bf16, sum-column in fp32, bf16-round the sum.

The 96-row axis is loaded as a padded (BLOCK_M=128, 1) tile with ZERO padding;
OOB rows contribute 0 to the sum. Output is stored to a padded buffer and
sliced with `narrow` afterwards.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 96
COLS = 65


@ct.kernel
def _materialize_and_sum_kernel(
    a_ptr,     # bf16 (BLOCK_M, 65) padded
    b_ptr,     # bf16 (BLOCK_M, 65) padded
    out_ptr,   # bf16 (BLOCK_M, 65) padded
    sum_ptr,   # f32  (65,)
    BLOCK_M: ct.Constant[int],
):
    col = ct.bid(0)
    a = ct.load(a_ptr, index=(0, col), shape=(BLOCK_M, 1))
    b = ct.load(b_ptr, index=(0, col), shape=(BLOCK_M, 1))
    a_f = ct.astype(a, ct.float32)
    b_f = ct.astype(b, ct.float32)
    value_f = a_f * (b_f * (1.0 - b_f))
    value_bf16 = ct.astype(value_f, ct.bfloat16)
    ct.store(out_ptr, index=(0, col), tile=value_bf16)

    value_f_back = ct.astype(value_bf16, ct.float32)
    col_sum = ct.sum(value_f_back, axis=0)  # shape (1,)
    rounded = ct.astype(ct.astype(col_sum, ct.bfloat16), ct.float32)
    ct.store(sum_ptr, index=(col,), tile=rounded)


@oracle_impl(hardware="B200", point="286ee5cf", BLOCK_M=128)
def oracle_forward(inputs, *, BLOCK_M: int):
    arg0, arg1, _shape0 = inputs
    device = arg0.device
    # Pad row axis to BLOCK_M for cuTile's power-of-2 tile constraint.
    a_padded = torch.zeros((BLOCK_M, COLS), device=device, dtype=torch.bfloat16)
    a_padded[:ROWS] = arg0
    b_padded = torch.zeros((BLOCK_M, COLS), device=device, dtype=torch.bfloat16)
    b_padded[:ROWS] = arg1
    out_padded = torch.empty((BLOCK_M, COLS), device=device, dtype=torch.bfloat16)
    sums = torch.empty_strided((COLS,), (1,), device=device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (COLS, 1, 1),
        _materialize_and_sum_kernel,
        (a_padded, b_padded, out_padded, sums, BLOCK_M),
    )
    out = out_padded[:ROWS].contiguous()
    return out, out.t(), sums
