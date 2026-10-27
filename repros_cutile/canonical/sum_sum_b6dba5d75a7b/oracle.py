"""cuTile port of sum_sum_b6dba5d75a7b: tiny BERT softmax-backward tail.

Rows=16, Cols=2. One kernel does row-sum, exp*sum, subtract, cast to bf16,
then column-sum of the bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16
COLS = 2
OUT_SHAPE = (ROWS, COLS)
OUT_STRIDE = (COLS, 1)
PERMUTE_SHAPE = (COLS, ROWS)
PERMUTE_STRIDE = (1, COLS)


@ct.kernel
def _softmax_backward_tiny_kernel(
    logits_ptr,   # (32,) f32
    grad_ptr,     # (32,) f32
    dense_ptr,    # (32,) bf16
    sum_ptr,      # (2,) f32
    BLOCK_M: ct.Constant[int],
):
    rows = ct.arange(BLOCK_M, dtype=ct.int32)
    off0 = rows * 2
    off1 = off0 + 1
    active = rows < 16

    logit0 = ct.gather(logits_ptr, off0, mask=active, padding_value=0.0)
    logit1 = ct.gather(logits_ptr, off1, mask=active, padding_value=0.0)
    grad0 = ct.gather(grad_ptr, off0, mask=active, padding_value=0.0)
    grad1 = ct.gather(grad_ptr, off1, mask=active, padding_value=0.0)

    row_sum = grad0 + grad1
    v0 = grad0 - ct.exp(logit0) * row_sum
    v1 = grad1 - ct.exp(logit1) * row_sum
    dense0 = ct.astype(v0, ct.bfloat16)
    dense1 = ct.astype(v1, ct.bfloat16)

    ct.scatter(dense_ptr, off0, dense0, mask=active)
    ct.scatter(dense_ptr, off1, dense1, mask=active)

    zero = ct.full((BLOCK_M,), 0.0, dtype=ct.float32)
    s0 = ct.sum(ct.where(active, ct.astype(dense0, ct.float32), zero))
    s1 = ct.sum(ct.where(active, ct.astype(dense1, ct.float32), zero))
    # Round-trip bf16 then back to f32
    s0_rt = ct.astype(ct.astype(s0, ct.bfloat16), ct.float32)
    s1_rt = ct.astype(ct.astype(s1, ct.bfloat16), ct.float32)
    # Store scalars — use 1-sized tile scatter
    ct.scatter(sum_ptr, ct.full((1,), 0, dtype=ct.int32), ct.reshape(s0_rt, (1,)))
    ct.scatter(sum_ptr, ct.full((1,), 1, dtype=ct.int32), ct.reshape(s1_rt, (1,)))


@oracle_impl(hardware="B200", point="840e34c4", BLOCK_M=16)
def oracle_forward(inputs, *, BLOCK_M: int):
    logits, grad, _shape = inputs
    dense_out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=grad.device, dtype=torch.bfloat16)
    sum_out = torch.empty_strided((COLS,), (1,), device=grad.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _softmax_backward_tiny_kernel,
        (logits.reshape(-1), grad.reshape(-1), dense_out.reshape(-1), sum_out, BLOCK_M),
    )
    return dense_out, dense_out.as_strided(PERMUTE_SHAPE, PERMUTE_STRIDE), sum_out
