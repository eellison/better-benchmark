"""cuTile port of sum_06d4cf240d3c: DeBERTa attention softmax-backward tail.

Row kernel: reads grad, keep-mask, score, row_shift, row_denom for each row,
reconstructs probs, computes row-product sum, and emits FMA(-probs, sum,
scaled_grad*probs) as bf16. Emits the scalar-zero and false-mask side outputs
via torch outside the kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 8
HEADS = 24
Q_LEN = 512
K_LEN = 512
N_ROWS = BATCH * HEADS * Q_LEN
SCALE = 1.1111111640930176


@ct.kernel
def _softmax_backward_kernel(
    grad_ptr,        # bf16 (N_ROWS, K_LEN)
    keep_ptr,        # bool (N_ROWS, K_LEN)
    score_ptr,       # bf16 (N_ROWS, K_LEN)
    row_shift_ptr,   # f32 (N_ROWS,)
    row_denom_ptr,   # f32 (N_ROWS,)
    out_ptr,         # bf16 (N_ROWS, K_LEN)
    K_LEN_: ct.Constant[int],
):
    row = ct.bid(0)
    grad = ct.astype(ct.load(grad_ptr, index=(row, 0), shape=(1, K_LEN_)), ct.float32)
    keep = ct.astype(ct.load(keep_ptr, index=(row, 0), shape=(1, K_LEN_)), ct.float32)
    score = ct.astype(ct.load(score_ptr, index=(row, 0), shape=(1, K_LEN_)), ct.float32)
    row_shift = ct.load(row_shift_ptr, index=(row,), shape=(1,))
    row_denom = ct.load(row_denom_ptr, index=(row,), shape=(1,))

    row_shift_2d = ct.reshape(row_shift, (1, 1))
    row_denom_2d = ct.reshape(row_denom, (1, 1))
    probs = ct.exp(score - row_shift_2d) / row_denom_2d
    scaled_grad = grad * (keep * SCALE)
    product = scaled_grad * probs
    row_sum = ct.sum(product, axis=1, keepdims=True)
    out = product + (-probs) * row_sum
    out_bf = ct.astype(out, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out_bf)


@oracle_impl(hardware="B200", point="c2297120")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _s0, _s1, _s2 = inputs
    device = arg0_1.device

    scalar_zero = torch.zeros((), device=device, dtype=torch.bfloat16)
    false_mask = torch.zeros(
        (BATCH, 1, Q_LEN, K_LEN),
        device=device, dtype=torch.bool,
    )

    # Reshape inputs to (N_ROWS, K_LEN).
    grad_2d = arg0_1.view(N_ROWS, K_LEN)
    keep_2d = arg1_1.view(N_ROWS, K_LEN)
    score_2d = arg2_1.view(N_ROWS, K_LEN)
    shift_1d = arg3_1.view(N_ROWS)
    denom_1d = arg4_1.view(N_ROWS)

    # Output shape matches _shape_param_2 = [192, 512, 512] with contiguous stride
    out = torch.empty_strided(
        (BATCH * HEADS, Q_LEN, K_LEN),
        (Q_LEN * K_LEN, K_LEN, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_2d = out.view(N_ROWS, K_LEN)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N_ROWS, 1, 1),
        _softmax_backward_kernel,
        (grad_2d, keep_2d, score_2d, shift_1d, denom_1d, out_2d, K_LEN),
    )

    # Post-op: apply the false-mask (all False -> keep out unchanged) and view to final shape
    view_final = out.view(192, 512, 512)
    return scalar_zero, false_mask, view_final
