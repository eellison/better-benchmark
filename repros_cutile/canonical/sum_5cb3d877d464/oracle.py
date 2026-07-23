"""cuTile port of sum_5cb3d877d464: Visformer attention softmax backward.

Per-row (128*6*196 rows of length 196): recompute softmax probs, compute
grad*probs product with row sum, apply fma epilogue, bf16 rounding + 0.125 scale.
N_COLS=196 (non-pow2) so uses BLOCK_N=256 with column masking.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
HEADS = 6
N_COLS = 196
BLOCK_N = 256  # next pow2 >= 196


@ct.kernel
def _visformer_recompute_softmax_bwd_kernel(
    grad_ptr,        # bf16 flat (n_rows * N_COLS,)
    logits_ptr,      # bf16 flat — strided view; index computed via strides
    arg2_ptr,        # f32 flat (BATCH * HEADS * 196,) via given strides
    arg3_ptr,        # f32 same
    mask_bool_ptr,   # bool flat
    denom_ptr,       # f32 flat
    out_ptr,         # bf16 flat (n_rows * N_COLS,)
    LOGITS_STRIDE_BH: ct.Constant[int],
    LOGITS_STRIDE_Q: ct.Constant[int],
    ROW_ARG2_STRIDE_B: ct.Constant[int],
    ROW_ARG2_STRIDE_H: ct.Constant[int],
    ROW_ARG2_STRIDE_Q: ct.Constant[int],
    ROW_ARG3_STRIDE_B: ct.Constant[int],
    ROW_ARG3_STRIDE_H: ct.Constant[int],
    ROW_ARG3_STRIDE_Q: ct.Constant[int],
    ROW_MASK_STRIDE_B: ct.Constant[int],
    ROW_MASK_STRIDE_H: ct.Constant[int],
    ROW_MASK_STRIDE_Q: ct.Constant[int],
    ROW_DENOM_STRIDE_B: ct.Constant[int],
    ROW_DENOM_STRIDE_H: ct.Constant[int],
    ROW_DENOM_STRIDE_Q: ct.Constant[int],
    HEADS_C: ct.Constant[int],
    N_COLS_C: ct.Constant[int],
    BLOCK_N_C: ct.Constant[int],
):
    row = ct.bid(0)  # row in [0, n_rows)
    cols = ct.arange(BLOCK_N_C, dtype=ct.int32)
    col_mask = cols < N_COLS_C

    flat_bh = row // N_COLS_C
    q_idx = row - flat_bh * N_COLS_C
    batch = flat_bh // HEADS_C
    head = flat_bh - batch * HEADS_C

    compact_offs = row * N_COLS_C + cols
    logits_offs = flat_bh * LOGITS_STRIDE_BH + q_idx * LOGITS_STRIDE_Q + cols
    arg2_off = batch * ROW_ARG2_STRIDE_B + head * ROW_ARG2_STRIDE_H + q_idx * ROW_ARG2_STRIDE_Q
    arg3_off = batch * ROW_ARG3_STRIDE_B + head * ROW_ARG3_STRIDE_H + q_idx * ROW_ARG3_STRIDE_Q
    mask_off = batch * ROW_MASK_STRIDE_B + head * ROW_MASK_STRIDE_H + q_idx * ROW_MASK_STRIDE_Q
    denom_off = batch * ROW_DENOM_STRIDE_B + head * ROW_DENOM_STRIDE_H + q_idx * ROW_DENOM_STRIDE_Q

    grad = ct.astype(
        ct.gather(grad_ptr, compact_offs, mask=col_mask, padding_value=0), ct.float32)
    logits = ct.astype(
        ct.gather(logits_ptr, logits_offs, mask=col_mask, padding_value=0), ct.float32)
    arg2_val = ct.astype(ct.gather(arg2_ptr, arg2_off, padding_value=0.0), ct.float32)
    arg3_val = ct.astype(ct.gather(arg3_ptr, arg3_off, padding_value=0.0), ct.float32)
    choose = ct.gather(mask_bool_ptr, mask_off, padding_value=False)
    denom = ct.astype(ct.gather(denom_ptr, denom_off, padding_value=1.0), ct.float32)

    logits_scaled_bf = ct.astype(ct.astype(logits * 0.125, ct.bfloat16), ct.float32)
    branch_scaled_sub = (logits - arg2_val) * 0.125
    branch_sub_scaled = logits_scaled_bf - arg3_val
    exp_arg = ct.where(choose, branch_scaled_sub, branch_sub_scaled)
    probs = ct.exp(exp_arg) / denom

    product = grad * probs
    zero_f = ct.zeros((BLOCK_N_C,), dtype=ct.float32)
    product_masked = ct.where(col_mask, product, zero_f)
    row_sum = ct.sum(product_masked)
    # fma = -probs * row_sum + product
    fma = product - probs * row_sum
    rounded = ct.astype(ct.astype(fma, ct.bfloat16), ct.float32)
    out = ct.astype(rounded * 0.125, ct.bfloat16)

    ct.scatter(out_ptr, compact_offs, out, mask=col_mask)


@oracle_impl(hardware="B200", point="66d9f76b")
def oracle_forward(inputs):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    del _shape_param_0, _shape_param_1

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    n_cols = out_shape[2]
    n_rows = out_shape[0] * out_shape[1]
    out = torch.empty_strided(
        out_shape, (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )

    # Force contiguous layouts to make stride math straightforward.
    arg2_c = arg2_1.contiguous()
    arg3_c = arg3_1.contiguous()
    arg4_c = arg4_1.contiguous()
    arg5_c = arg5_1.contiguous()
    arg1_c = arg1_1.contiguous()

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _visformer_recompute_softmax_bwd_kernel,
        (
            arg0_1.reshape(-1),
            arg1_c.reshape(-1),
            arg2_c.reshape(-1),
            arg3_c.reshape(-1),
            arg4_c.reshape(-1),
            arg5_c.reshape(-1),
            out.reshape(-1),
            int(arg1_c.stride(0)), int(arg1_c.stride(1)),
            int(arg2_c.stride(0)), int(arg2_c.stride(1)), int(arg2_c.stride(2)),
            int(arg3_c.stride(0)), int(arg3_c.stride(1)), int(arg3_c.stride(2)),
            int(arg4_c.stride(0)), int(arg4_c.stride(1)), int(arg4_c.stride(2)),
            int(arg5_c.stride(0)), int(arg5_c.stride(1)), int(arg5_c.stride(2)),
            HEADS, n_cols, BLOCK_N,
        ),
    )
    return out
