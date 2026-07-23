"""cuTile port of sum_983bc7365627: LayoutLM attention softmax-backward
row epilogue with dropout gate, branch selection, and post-scale.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE = 1.1111111111111112
POST_SCALE = 0.125


@ct.kernel
def _softmax_backward_row_kernel(
    grad_ptr,             # bf16 (rows, N_COLS)
    keep_ptr,             # b8 (rows, N_COLS)
    logits_ptr,           # bf16 (rows, N_COLS)
    row_shift_true_ptr,   # f32 (rows,)
    row_shift_false_ptr,  # f32 (rows,)
    branch_ptr,           # b8 (rows,)
    row_denom_ptr,        # f32 (rows,)
    out_ptr,              # bf16 (rows, N_COLS)
    N_COLS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    grad = ct.load(grad_ptr, index=(row_tile, 0), shape=(BLOCK_M, BLOCK_N))
    keep = ct.load(keep_ptr, index=(row_tile, 0), shape=(BLOCK_M, BLOCK_N))
    logits = ct.load(logits_ptr, index=(row_tile, 0), shape=(BLOCK_M, BLOCK_N))
    row_shift_true = ct.load(row_shift_true_ptr, index=(row_tile,), shape=(BLOCK_M,))
    row_shift_false = ct.load(row_shift_false_ptr, index=(row_tile,), shape=(BLOCK_M,))
    branch = ct.load(branch_ptr, index=(row_tile,), shape=(BLOCK_M,))
    row_denom = ct.load(row_denom_ptr, index=(row_tile,), shape=(BLOCK_M,))

    grad_f = ct.astype(grad, ct.float32)
    keep_f = ct.astype(keep, ct.float32)
    logits_f = ct.astype(logits, ct.float32)
    row_shift_true_f = ct.astype(row_shift_true, ct.float32)
    row_shift_false_f = ct.astype(row_shift_false, ct.float32)
    row_denom_f = ct.astype(row_denom, ct.float32)

    keep_scale = ct.astype(ct.astype(keep_f * DROPOUT_SCALE, ct.bfloat16), ct.float32)
    dropped_grad = ct.astype(ct.astype(grad_f * keep_scale, ct.bfloat16), ct.float32)

    logits_scaled_bf16 = ct.astype(
        ct.astype(logits_f * POST_SCALE, ct.bfloat16), ct.float32
    )
    rst_2d = ct.reshape(row_shift_true_f, (BLOCK_M, 1))
    rsf_2d = ct.reshape(row_shift_false_f, (BLOCK_M, 1))
    true_scores = (logits_f - rst_2d) * POST_SCALE
    false_scores = logits_scaled_bf16 - rsf_2d
    branch_2d = ct.reshape(branch, (BLOCK_M, 1))
    scores = ct.where(branch_2d, true_scores, false_scores)

    probs = ct.exp(scores) / ct.reshape(row_denom_f, (BLOCK_M, 1))
    product = dropped_grad * probs
    row_sum = ct.sum(product, axis=1)
    row_sum_2d = ct.reshape(row_sum, (BLOCK_M, 1))
    fma = -probs * row_sum_2d + product
    rounded = ct.astype(ct.astype(fma, ct.bfloat16), ct.float32)
    out = ct.astype(rounded * POST_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(row_tile, 0), tile=out)


@oracle_impl(hardware="B200", point="03fa1c14", BLOCK_M=4, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    n_cols = int(arg2_1.shape[-1])
    n_rows = arg2_1.numel() // n_cols

    # Reshape inputs to 2D contiguous views for the kernel.
    grad_2d = arg0_1.view(n_rows, n_cols)
    keep_2d = arg1_1.view(n_rows, n_cols)
    logits_2d = arg2_1.view(n_rows, n_cols)
    rst_1d = arg3_1.reshape(-1)
    rsf_1d = arg4_1.reshape(-1)
    branch_1d = arg5_1.reshape(-1)
    denom_1d = arg6_1.reshape(-1)
    out_2d = out.view(n_rows, n_cols)

    stream = torch.cuda.current_stream()
    grid_m = (n_rows + BLOCK_M - 1) // BLOCK_M
    ct.launch(
        stream,
        (grid_m, 1, 1),
        _softmax_backward_row_kernel,
        (grad_2d, keep_2d, logits_2d, rst_1d, rsf_1d, branch_1d, denom_1d,
         out_2d, n_cols, BLOCK_M, BLOCK_N),
    )
    return out
