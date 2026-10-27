"""cuTile port of amax_sum_sum_bf54d8c8c94e: GPT-Neo seq-classifier grad scatter.

For each row: compute a log-softmax backward, add residual, then
index_put(accumulate=True) into a dense [32, 128, 2] buffer via
`ct.atomic_add`. Uses the `sum_2` intermediate (row-sum of upstream) computed
inline. All arithmetic runs at fp32 with bf16 rounding boundaries.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _grad_scatter_kernel(
    div_scalar,     # f32 (1,)
    mask0_ptr,      # b8 (rows,)
    mask1_ptr,      # b8 (rows,)
    row_mask_ptr,   # b8 (rows,)
    logit0_ptr,     # bf16 (rows,)
    logit1_ptr,     # bf16 (rows,)
    resid0_ptr,     # bf16 (rows,)
    resid1_ptr,     # bf16 (rows,)
    dst_row_ptr,    # i64 (rows,)
    dst_col_ptr,    # i64 (rows,)
    view_out_ptr,   # bf16 (dst_dim0, dst_dim1, 2)
    ROWS: ct.Constant[int],
    DST_DIM1: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
):
    pid = ct.bid(0)
    # single grid block covers all rows since ROWS <= BLOCK_ROWS
    div_tile = ct.load(div_scalar, index=(0,), shape=(1,))
    div_v = ct.reshape(div_tile, ())  # 0-d scalar; but cuTile has no 0d; keep as (1,)
    # Instead just broadcast (1,)
    row_enabled = ct.load(row_mask_ptr, index=(0,), shape=(BLOCK_ROWS,),
                          padding_mode=ct.PaddingMode.ZERO)
    scale_f = ct.astype(row_enabled, ct.float32) * ct.reshape(div_tile, (1,))
    # scale shape (BLOCK_ROWS,)

    mask0 = ct.load(mask0_ptr, index=(0,), shape=(BLOCK_ROWS,),
                    padding_mode=ct.PaddingMode.ZERO)
    mask1 = ct.load(mask1_ptr, index=(0,), shape=(BLOCK_ROWS,),
                    padding_mode=ct.PaddingMode.ZERO)
    upstream0_f = ct.where(mask0, -1.0, 0.0) * scale_f
    upstream0 = ct.astype(ct.astype(upstream0_f, ct.bfloat16), ct.float32)
    upstream1_f = ct.where(mask1, -1.0, 0.0) * scale_f
    upstream1 = ct.astype(ct.astype(upstream1_f, ct.bfloat16), ct.float32)
    upstream_sum = upstream0 + upstream1

    logit0 = ct.astype(ct.load(logit0_ptr, index=(0,), shape=(BLOCK_ROWS,),
                               padding_mode=ct.PaddingMode.ZERO), ct.float32)
    logit1 = ct.astype(ct.load(logit1_ptr, index=(0,), shape=(BLOCK_ROWS,),
                               padding_mode=ct.PaddingMode.ZERO), ct.float32)
    row_max = ct.where(logit0 > logit1, logit0, logit1)
    shifted0 = logit0 - row_max
    shifted1 = logit1 - row_max
    denom = ct.exp(shifted0) + ct.exp(shifted1)
    log_denom = ct.log(denom)
    log_prob0 = ct.astype(ct.astype(shifted0 - log_denom, ct.bfloat16), ct.float32)
    log_prob1 = ct.astype(ct.astype(shifted1 - log_denom, ct.bfloat16), ct.float32)
    prob0 = ct.exp(log_prob0)
    prob1 = ct.exp(log_prob1)

    grad0 = ct.astype(ct.astype(upstream0 - prob0 * upstream_sum, ct.bfloat16), ct.float32)
    grad1 = ct.astype(ct.astype(upstream1 - prob1 * upstream_sum, ct.bfloat16), ct.float32)

    resid0 = ct.astype(ct.load(resid0_ptr, index=(0,), shape=(BLOCK_ROWS,),
                               padding_mode=ct.PaddingMode.ZERO), ct.float32)
    resid1 = ct.astype(ct.load(resid1_ptr, index=(0,), shape=(BLOCK_ROWS,),
                               padding_mode=ct.PaddingMode.ZERO), ct.float32)
    out0 = ct.astype(resid0 + grad0, ct.bfloat16)
    out1 = ct.astype(resid1 + grad1, ct.bfloat16)

    dst_row = ct.load(dst_row_ptr, index=(0,), shape=(BLOCK_ROWS,),
                      padding_mode=ct.PaddingMode.ZERO)
    dst_col = ct.load(dst_col_ptr, index=(0,), shape=(BLOCK_ROWS,),
                      padding_mode=ct.PaddingMode.ZERO)
    # Mask out padded lanes (rows >= ROWS)
    lane = ct.arange(BLOCK_ROWS, dtype=ct.int32)
    active = lane < ROWS
    # Redirect invalid lanes to out-of-bounds so atomic_add drops.
    invalid_row = ct.full((BLOCK_ROWS,), 1 << 60, dtype=ct.int64)
    dst_row_safe = ct.where(active, dst_row, invalid_row)
    zero_i64 = ct.full((BLOCK_ROWS,), 0, dtype=ct.int64)
    dst_col_safe = ct.where(active, dst_col, zero_i64)
    col0 = ct.full((BLOCK_ROWS,), 0, dtype=ct.int64)
    col1 = ct.full((BLOCK_ROWS,), 1, dtype=ct.int64)
    ct.atomic_add(view_out_ptr, (dst_row_safe, dst_col_safe, col0), out0)
    ct.atomic_add(view_out_ptr, (dst_row_safe, dst_col_safe, col1), out1)


@oracle_impl(hardware="B200", point="535c8dce")
def oracle_forward(inputs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1,
        shape_3d, shape_2d,
    ) = inputs

    view_shape = tuple(int(dim) for dim in shape_3d)   # (32, 128, 2)
    flat_shape = tuple(int(dim) for dim in shape_2d)   # (4096, 2)
    device = arg4_1.device
    rows = int(arg2_1.shape[0])

    full = torch.zeros((), device=device, dtype=torch.float32)
    view3 = torch.zeros(view_shape, device=device, dtype=torch.bfloat16)

    # Split logits into two 1D tensors (bf16 [rows, 2] -> two [rows] each).
    logit0_view = arg4_1[:, 0].contiguous()
    logit1_view = arg4_1[:, 1].contiguous()
    resid0_view = arg5_1[:, 0].contiguous()
    resid1_view = arg5_1[:, 1].contiguous()
    mask0_view = arg2_1[:, 0].contiguous()
    mask1_view = arg2_1[:, 1].contiguous()
    row_mask = arg3_1.view(rows).contiguous()

    # Compute div scalar (arg0_1 / arg1_1) as 0-d f32 -> view (1,)
    div_val = (arg0_1 / arg1_1).view(1)

    BLOCK_ROWS = 32  # rows == 32 so no OOB
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (1, 1, 1), _grad_scatter_kernel,
        (div_val, mask0_view, mask1_view, row_mask,
         logit0_view, logit1_view, resid0_view, resid1_view,
         arg6_1, arg7_1, view3, rows, int(view_shape[1]), BLOCK_ROWS),
    )
    view_flat = view3.view(flat_shape)
    return full, view_flat, view_flat.permute(1, 0)
