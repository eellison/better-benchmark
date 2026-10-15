"""cuTile port of sum_8db0f751a574: DistilBERT attention-probability backward row.

Row-reduction: dropout-scaled grad * prob, sum over last dim, then FMA:
out[i,j] = product[i,j] - prob[i,j] * row_sum[i].

Uses BLOCK_M rows per program (matching Triton's BLOCK_M=8 for K=128 and
BLOCK_M=2 for K=512) to reduce grid size.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112


@ct.kernel
def _row_sum_fma_bf16_kernel(
    grad_ptr,
    keep_ptr,
    prob_ptr,
    out_ptr,
    K: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
):
    row_block = ct.bid(0)

    grad = ct.load(grad_ptr, index=(row_block, 0), shape=(BLOCK_M, K))
    keep = ct.load(keep_ptr, index=(row_block, 0), shape=(BLOCK_M, K))
    prob = ct.load(prob_ptr, index=(row_block, 0), shape=(BLOCK_M, K))

    grad_f = ct.astype(grad, ct.float32)
    keep_f = ct.astype(keep, ct.float32)
    prob_f = ct.astype(prob, ct.float32)

    keep_scale = ct.astype(ct.astype(keep_f * DROPOUT_SCALE, ct.bfloat16), ct.float32)
    scaled_grad = ct.astype(ct.astype(grad_f * keep_scale, ct.bfloat16), ct.float32)
    product = scaled_grad * prob_f
    row_sum = ct.sum(product, axis=1)  # (BLOCK_M,)
    row_sum_2d = ct.reshape(row_sum, (BLOCK_M, 1))
    out = product - prob_f * row_sum_2d
    ct.store(out_ptr, index=(row_block, 0), tile=ct.astype(out, ct.bfloat16))


def _launch(inputs, *, K: int, BLOCK_M: int):
    grad, keep, prob, _shape_param_0, _shape_param_1 = inputs
    out = torch.empty_strided(
        tuple(grad.shape),
        tuple(grad.stride()),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    rows = prob.numel() // K
    grad_2d = grad.reshape(rows, K)
    keep_2d = keep.reshape(rows, K)
    prob_2d = prob.reshape(rows, K)
    out_2d = out.reshape(rows, K)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _row_sum_fma_bf16_kernel,
        (grad_2d, keep_2d, prob_2d, out_2d, K, BLOCK_M),
    )
    return out


@oracle_impl(hardware="B200", point="2e48b06a", K=128, BLOCK_M=8)
@oracle_impl(hardware="B200", point="87ed7153", K=128, BLOCK_M=8)
@oracle_impl(hardware="B200", point="6ffb5b71", K=128, BLOCK_M=8)
@oracle_impl(hardware="B200", point="c99a78fd", K=512, BLOCK_M=2)
@oracle_impl(hardware="B200", point="d02283d5", K=512, BLOCK_M=2)
def oracle_forward(inputs, *, K: int, BLOCK_M: int):
    return _launch(inputs, K=K, BLOCK_M=BLOCK_M)
