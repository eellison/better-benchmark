"""cuTile port of amax_sum_sum_1bdd2568d0b8: GPT-2 SeqCls scatter tail.

Softmax-backward for 8 rows x 2 cols, then index_put into an [8, 1024, 2]
zeros buffer via torch. Uses one cuTile kernel to compute per-row grad;
the index_put is done via torch outside the kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 8
COLS = 2
ROW_BLOCK = 8


@ct.kernel
def _softmax_backward_kernel(
    arg0_ptr,        # f32 scalar
    arg1_ptr,        # f32 scalar
    mask_ptr,        # bool (ROWS, 2)
    row_mask_ptr,    # bool (ROWS, 1)
    logits_ptr,      # bf16 (ROWS, 2)
    residual_ptr,    # bf16 (ROWS, 2)
    out_ptr,         # bf16 (ROWS, 2)
    ROW_BLOCK_: ct.Constant[int],
):
    a0 = ct.load(arg0_ptr, index=(0,), shape=(1,))
    a1 = ct.load(arg1_ptr, index=(0,), shape=(1,))
    div_scalar = a0 / a1
    # Broadcast to (ROW_BLOCK, 2)
    div_2d = ct.reshape(div_scalar, (1, 1))

    mask_bf = ct.load(mask_ptr, index=(0, 0), shape=(ROW_BLOCK_, 2))
    mask_f = ct.astype(mask_bf, ct.float32)
    row_enabled = ct.load(row_mask_ptr, index=(0, 0), shape=(ROW_BLOCK_, 1))
    row_enabled_f = ct.astype(row_enabled, ct.float32)
    scale = row_enabled_f * div_2d

    upstream_f = mask_f * (-1.0) * scale
    upstream_bf = ct.astype(upstream_f, ct.bfloat16)
    upstream_rt = ct.astype(upstream_bf, ct.float32)
    upstream_sum = ct.sum(upstream_rt, axis=1, keepdims=True)

    logits_bf = ct.load(logits_ptr, index=(0, 0), shape=(ROW_BLOCK_, 2))
    logits_f = ct.astype(logits_bf, ct.float32)
    row_max = ct.max(logits_f, axis=1, keepdims=True)
    shifted = logits_f - row_max
    exp_val = ct.exp(shifted)
    denom = ct.sum(exp_val, axis=1, keepdims=True)
    log_denom = ct.log(denom)
    log_prob = shifted - log_denom
    log_prob_bf = ct.astype(log_prob, ct.bfloat16)
    log_prob_rt = ct.astype(log_prob_bf, ct.float32)
    prob = ct.exp(log_prob_rt)

    grad = upstream_rt - prob * upstream_sum
    grad_bf = ct.astype(grad, ct.bfloat16)
    grad_rt = ct.astype(grad_bf, ct.float32)

    residual_bf = ct.load(residual_ptr, index=(0, 0), shape=(ROW_BLOCK_, 2))
    residual_f = ct.astype(residual_bf, ct.float32)
    out_f = residual_f + grad_rt
    out_bf = ct.astype(out_f, ct.bfloat16)
    ct.store(out_ptr, index=(0, 0), tile=out_bf)


@oracle_impl(hardware="B200", point="0fd7b2d6")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        shape_full, shape_view,
    ) = inputs

    full_shape = tuple(int(dim) for dim in shape_full)
    view_shape = tuple(int(dim) for dim in shape_view)
    device = arg4_1.device

    # Intermediate result buffer for per-row softmax-backward+residual
    grad_add = torch.empty((ROWS, COLS), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    a0_1d = arg0_1.view(1)
    a1_1d = arg1_1.view(1)
    ct.launch(
        stream,
        (1, 1, 1),
        _softmax_backward_kernel,
        (a0_1d, a1_1d, arg2_1, arg3_1, arg4_1, arg5_1, grad_add, ROW_BLOCK),
    )

    # Now scatter into [8, 1024, 2] buffer via index_put with accumulate.
    buf = torch.zeros(full_shape, device=device, dtype=torch.bfloat16)
    buf.index_put_((arg6_1, arg7_1), grad_add, accumulate=True)
    view = buf.view(view_shape)
    permute = view.permute(1, 0)
    return view, permute
