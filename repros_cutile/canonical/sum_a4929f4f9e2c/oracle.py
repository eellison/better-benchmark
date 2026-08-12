"""cuTile port of sum_a4929f4f9e2c: Demucs GLU-derivative cat + channel sum.

For each channel c in C_HALF: compute first = sigmoid(gate)*grad and
second = (1-sig)*sig*first_half*grad; store both halves of [BATCH, C_TOTAL, TIME]
via a padded scatter; reduce over (batch, time) to yield f32 sum vector [C_TOTAL].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 4
C_HALF = 2048
C_TOTAL = 4096
BLOCK_K = 512  # power-of-2 covering BATCH*TIME (max 4*92=368)


@ct.kernel
def _cat_sum_kernel(
    grad_ptr,        # bf16 [BATCH, C_HALF, TIME]
    gate_pair_ptr,   # bf16 [BATCH, C_TOTAL, TIME]
    cat_out_ptr,     # bf16 [BATCH, C_TOTAL, TIME]
    sum_out_ptr,     # f32 [C_TOTAL]
    TIME: ct.Constant[int],
    BATCH_: ct.Constant[int],
    C_HALF_: ct.Constant[int],
    C_TOTAL_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
):
    c = ct.bid(0)
    k = ct.arange(BLOCK_K_, dtype=ct.int32)
    k_total = BATCH_ * TIME
    valid = k < k_total
    b = k // TIME
    t = k - b * TIME

    grad_offsets = b * (C_HALF_ * TIME) + c * TIME + t
    pair_base = b * (C_TOTAL_ * TIME) + c * TIME + t
    cat_base = b * (C_TOTAL_ * TIME) + c * TIME + t
    second_offset = C_HALF_ * TIME

    # Use gather to handle irregular offsets
    grad = ct.astype(ct.gather(grad_ptr, grad_offsets), ct.float32)
    first_half = ct.astype(ct.gather(gate_pair_ptr, pair_base), ct.float32)
    gate = ct.astype(
        ct.gather(gate_pair_ptr, pair_base + second_offset),
        ct.float32,
    )

    sigmoid = 1.0 / (ct.exp(0.0 - gate) + 1.0)
    first = sigmoid * grad
    second = (1.0 - sigmoid) * sigmoid * first_half * grad
    first_bf16 = ct.astype(first, ct.bfloat16)
    second_bf16 = ct.astype(second, ct.bfloat16)

    ct.scatter(cat_out_ptr, cat_base, first_bf16, mask=valid)
    ct.scatter(cat_out_ptr, cat_base + second_offset, second_bf16, mask=valid)

    first_masked = ct.where(valid, ct.astype(first_bf16, ct.float32), 0.0)
    second_masked = ct.where(valid, ct.astype(second_bf16, ct.float32), 0.0)
    first_sum = ct.sum(first_masked)
    second_sum = ct.sum(second_masked)
    first_sum_rt = ct.astype(ct.astype(first_sum, ct.bfloat16), ct.float32)
    second_sum_rt = ct.astype(ct.astype(second_sum, ct.bfloat16), ct.float32)
    ct.store(sum_out_ptr, index=(c,), tile=ct.reshape(first_sum_rt, (1,)))
    ct.store(sum_out_ptr, index=(c + C_HALF_,), tile=ct.reshape(second_sum_rt, (1,)))


@oracle_impl(hardware="B200", point="61711b7c", TIME=90)
@oracle_impl(hardware="B200", point="e6148dfd", TIME=92)
def oracle_forward(inputs, *, TIME: int):
    grad, gate_pair = inputs
    device = grad.device

    cat_out = torch.empty_strided(
        (BATCH, C_TOTAL, TIME),
        (C_TOTAL * TIME, TIME, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided(
        (C_TOTAL,),
        (1,),
        device=device,
        dtype=torch.float32,
    )

    # cuTile gather/scatter need a 1D flat view for offset indexing
    grad_flat = grad.contiguous().view(-1)
    gate_pair_flat = gate_pair.contiguous().view(-1)
    cat_out_flat = cat_out.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (C_HALF, 1, 1), _cat_sum_kernel,
        (grad_flat, gate_pair_flat, cat_out_flat, sum_out,
         TIME, BATCH, C_HALF, C_TOTAL, BLOCK_K),
    )
    return cat_out, sum_out
