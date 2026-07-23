"""cuTile port of sum_33b36e228e74: T5/MT5 attention softmax-backward row."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@ct.kernel
def _softmax_backward_kernel(
    grad_ptr, keep_ptr, logits_ptr, row_max_ptr, row_sum_ptr, out_ptr,
    N_COLS: ct.Constant[int],
):
    row = ct.bid(0)
    grad = ct.load(grad_ptr, index=(row, 0), shape=(1, N_COLS))
    keep = ct.load(keep_ptr, index=(row, 0), shape=(1, N_COLS))
    logits = ct.load(logits_ptr, index=(row, 0), shape=(1, N_COLS))
    row_max = ct.load(row_max_ptr, index=(row,), shape=(1,))
    denom = ct.load(row_sum_ptr, index=(row,), shape=(1,))

    grad_f = ct.astype(grad, ct.float32)
    logits_f = ct.astype(logits, ct.float32)
    keep_f = ct.astype(keep, ct.float32)

    scaled_keep = ct.astype(ct.astype(keep_f * DROPOUT_SCALE, ct.bfloat16), ct.float32)
    dropped_grad = ct.astype(ct.astype(grad_f * scaled_keep, ct.bfloat16), ct.float32)

    row_max_2d = ct.reshape(row_max, (1, 1))
    denom_2d = ct.reshape(denom, (1, 1))
    probs = ct.exp(logits_f - row_max_2d) / denom_2d
    product = dropped_grad * probs
    row_dot = ct.sum(product, axis=1, keepdims=True)
    # Match prims.fma.default(neg(probs), row_dot, product) = product - probs*row_dot
    # cuTile compiler is expected to fuse this into an FMA.
    out = product - probs * row_dot
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="e0876406")
def _oracle_forward_t5(inputs, **_kwargs):
    # T5 1024x1024 case: bf16 mul+add rounding differs from prims.fma over
    # 1024-element rows with adversarial synthetic row_sum values.
    raise NotImplementedError(
        "cuTile port unsupported for e0876406: bf16 softmax-backward drift "
        "vs prims.fma over 1024-element rows"
    )


@oracle_impl(hardware="B200", point="59cff286")
def oracle_forward(inputs):
    (grad, keep, logits, row_max, row_sum,
     _shape0, _shape1, out_shape_arg, flat_shape_arg) = inputs
    out_shape = tuple(int(dim) for dim in out_shape_arg)
    flat_shape = tuple(int(dim) for dim in flat_shape_arg)
    n_cols = int(out_shape[-1])
    n_rows = logits.numel() // n_cols
    device = grad.device

    out = torch.empty_strided(out_shape, _contiguous_stride(out_shape),
                              device=device, dtype=torch.bfloat16)

    grad_2d = grad.view(n_rows, n_cols)
    keep_2d = keep.view(n_rows, n_cols)
    logits_2d = logits.view(n_rows, n_cols)
    row_max_1d = row_max.view(n_rows)
    row_sum_1d = row_sum.view(n_rows)
    out_2d = out.view(n_rows, n_cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n_rows, 1, 1), _softmax_backward_kernel,
        (grad_2d, keep_2d, logits_2d, row_max_1d, row_sum_1d, out_2d, n_cols),
    )
    return out, out.view(flat_shape)
