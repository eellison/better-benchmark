"""cuTile port of sum_b2873b4c9052: CrossEntropy backward with vocab=262144.

For each row: subtract shift0, shift1; round to bf16 then back to f32; exp;
multiply by -grad (if label != -100) then subtract sparse -grad at label.
262144 is a power of 2, so all tile shapes align.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N_COLS = 262144


@ct.kernel
def _xent_backward_kernel(
    logits_ptr,        # bf16 (rows, N_COLS)
    shift0_ptr,        # f32 (rows, 1) -> viewed as (rows,)
    shift1_ptr,        # f32 (rows, 1) -> viewed as (rows,)
    labels_ptr,        # i64 (rows,)
    grad_ptr,          # bf16 () -> (1,)
    out_ptr,           # bf16 (rows, N_COLS)
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)

    # Load label + grad
    label_tile = ct.load(labels_ptr, index=(row,), shape=(1,))
    label_scalar = ct.reshape(label_tile, ())
    grad_tile = ct.load(grad_ptr, index=(0,), shape=(1,))
    grad_scalar = ct.reshape(ct.astype(grad_tile, ct.float32), ())

    ne_ignore = label_scalar != -100
    # For sparse: at column == label, value = -grad if ne_ignore else 0
    # For dense: exp(shifted) * row_sum
    # row_sum = -grad if ne_ignore else 0
    zero_scalar = ct.astype(ct.zeros((1,), dtype=ct.float32), ct.float32)
    zero_scalar = ct.reshape(zero_scalar, ())
    neg_grad = -grad_scalar
    row_sum = ct.where(ne_ignore, neg_grad, zero_scalar)

    shift0_tile = ct.load(shift0_ptr, index=(row,), shape=(1,))
    shift0 = ct.astype(shift0_tile, ct.float32)
    shift0_scalar = ct.reshape(shift0, ())
    shift1_tile = ct.load(shift1_ptr, index=(row,), shape=(1,))
    shift1 = ct.astype(shift1_tile, ct.float32)
    shift1_scalar = ct.reshape(shift1, ())

    logits = ct.load(logits_ptr, index=(row, col_block), shape=(1, BLOCK_N))
    logits_f = ct.astype(logits, ct.float32)
    shifted = logits_f - shift0_scalar - shift1_scalar
    rounded = ct.astype(ct.astype(shifted, ct.bfloat16), ct.float32)
    exp_val = ct.exp(rounded)

    cols = ct.arange(BLOCK_N, dtype=ct.int32) + col_block * BLOCK_N
    label_i32 = ct.astype(label_scalar, ct.int32)
    is_label = cols == label_i32
    is_label_2d = ct.reshape(is_label, (1, BLOCK_N))
    active_mask = ne_ignore
    sparse = ct.where(is_label_2d & active_mask,
                      ct.full((1, BLOCK_N), 0.0, dtype=ct.float32) + neg_grad,
                      ct.zeros((1, BLOCK_N), dtype=ct.float32))
    dense = exp_val * row_sum
    out = sparse - dense
    ct.store(out_ptr, index=(row, col_block), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="ed4a60c9", BLOCK_N=2048)
def oracle_forward(inputs, *, BLOCK_N: int):
    logits, shift0, shift1, labels, grad_scalar, *_shapes = inputs
    rows = int(logits.shape[0])
    n_cols = int(logits.shape[1])
    out = torch.empty_strided((rows, n_cols), (n_cols, 1),
                              device=logits.device, dtype=torch.bfloat16)
    shift0_1d = shift0.view(rows)
    shift1_1d = shift1.view(rows)
    grad_view = grad_scalar.view(1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, n_cols // BLOCK_N, 1),
        _xent_backward_kernel,
        (logits, shift0_1d, shift1_1d, labels, grad_view, out, BLOCK_N),
    )
    return out
