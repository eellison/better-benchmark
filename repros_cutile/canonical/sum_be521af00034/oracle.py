"""cuTile port of sum_be521af00034: DeBERTa attention softmax-backward.

Single row kernel — one program per row. Reconstruct probability from logits +
row_shift + row_denom, compute row-wise reduction, apply the fma epilogue
(cuTile's default arithmetic is round-to-nearest-even, so we just use *,+,-)
and where-mask against arg5_1.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE_F32 = 1.1111111640930176


@ct.kernel
def _softmax_backward_row_kernel(
    grad_ptr,          # bf16 [n_rows, N_COLS]
    keep_ptr,          # b8   [n_rows, N_COLS]
    logits_ptr,        # bf16 [n_rows, N_COLS]
    row_shift_ptr,     # f32  [n_rows]
    row_denom_ptr,     # f32  [n_rows]
    final_mask_ptr,    # b8   [batch, QUERY, N_COLS] flat as [batch*QUERY, N_COLS]
    fill_ptr,          # bf16 [1]
    out_ptr,           # bf16 [n_rows, N_COLS]
    HEADS: ct.Constant[int],
    QUERY: ct.Constant[int],
    N_COLS: ct.Constant[int],
    SCALE: ct.Constant[float],
):
    row = ct.bid(0)

    grad = ct.astype(
        ct.load(grad_ptr, index=(row, 0), shape=(1, N_COLS)),
        ct.float32,
    )
    keep = ct.astype(
        ct.load(keep_ptr, index=(row, 0), shape=(1, N_COLS)),
        ct.float32,
    )
    logits = ct.astype(
        ct.load(logits_ptr, index=(row, 0), shape=(1, N_COLS)),
        ct.float32,
    )
    row_shift = ct.astype(
        ct.load(row_shift_ptr, index=(row,), shape=(1,)),
        ct.float32,
    )
    row_denom = ct.astype(
        ct.load(row_denom_ptr, index=(row,), shape=(1,)),
        ct.float32,
    )

    # Final-mask offset: batch = row // (HEADS * QUERY), query = row % QUERY
    # final_mask is [batch, 1, QUERY, N_COLS] contiguous → row idx = batch*QUERY + query
    batch = row // (HEADS * QUERY)
    query = row % QUERY
    final_row = batch * QUERY + query
    final_mask = ct.load(final_mask_ptr, index=(final_row, 0), shape=(1, N_COLS))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    row_shift_2d = ct.reshape(row_shift, (1, 1))
    row_denom_2d = ct.reshape(row_denom, (1, 1))
    probs = ct.exp(logits - row_shift_2d) / row_denom_2d
    scaled_grad = grad * keep * SCALE
    product = scaled_grad * probs
    row_sum = ct.sum(product, axis=1, keepdims=True)
    fma = -probs * row_sum + product
    rounded = ct.astype(fma, ct.bfloat16)
    fill_bcast = ct.reshape(fill, (1, 1))
    result = ct.where(final_mask, fill_bcast, rounded)

    ct.store(out_ptr, index=(row, 0), tile=result)


@oracle_impl(hardware="B200", point="1e831143", BLOCK_M=8, BLOCK_N=512)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    del BLOCK_M  # per-row kernel; unused in cuTile port
    (
        arg0_1,  # bf16 [192, 512, 512] — grad
        arg1_1,  # b8   [8, 24, 512, 512] — keep
        arg2_1,  # bf16 [8, 24, 512, 512] — logits
        arg3_1,  # f32  [8, 24, 512, 1] — row_shift
        arg4_1,  # f32  [8, 24, 512, 1] — row_denom
        arg5_1,  # b8   [8, 1, 512, 512] — final_mask
        arg6_1,  # bf16 [] — fill
        _shape_param_0,
        _shape_param_1,
    ) = inputs

    n_cols = int(arg2_1.shape[-1])
    n_rows = int(arg2_1.numel() // n_cols)
    heads = int(arg2_1.shape[1])
    query = int(arg2_1.shape[2])
    batch = int(arg2_1.shape[0])
    del BLOCK_N

    device = arg0_1.device
    out = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=device,
        dtype=torch.bfloat16,
    )

    grad_2d = arg0_1.view(n_rows, n_cols)
    keep_2d = arg1_1.reshape(n_rows, n_cols)
    logits_2d = arg2_1.reshape(n_rows, n_cols)
    row_shift_1d = arg3_1.reshape(n_rows)
    row_denom_1d = arg4_1.reshape(n_rows)
    # arg5_1 is [batch, 1, query, n_cols] — flatten to [batch*query, n_cols]
    final_mask_2d = arg5_1.reshape(batch * query, n_cols)
    fill_1d = arg6_1.view(1)
    out_2d = out.view(n_rows, n_cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows, 1, 1),
        _softmax_backward_row_kernel,
        (grad_2d, keep_2d, logits_2d, row_shift_1d, row_denom_1d,
         final_mask_2d, fill_1d, out_2d,
         heads, query, n_cols, DROPOUT_SCALE_F32),
    )
    return out
