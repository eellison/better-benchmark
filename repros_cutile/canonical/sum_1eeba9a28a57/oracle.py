"""cuTile port of sum_1eeba9a28a57: BERT attention softmax-backward.

Multi-row: BLOCK_M rows per block to match Triton's BLOCK_M=8.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HEADS = 12
QUERY = 128
N_COLS = 128
DROPOUT_SCALE = 1.1111111640930176
POST_SCALE = 0.125


@ct.kernel
def _bert_softmax_bwd_kernel(
    grad_ptr,          # bf16 [rows, N_COLS]
    keep_ptr,          # b8   [rows, N_COLS]
    logits_ptr,        # bf16 [rows, N_COLS]
    row_shift_ptr,     # f32  [rows]
    row_denom_ptr,     # f32  [rows]
    mask_src_ptr,      # b8   [BATCH*QUERY, N_COLS] (viewed)
    eq_out_ptr,        # b8   [BATCH*QUERY, N_COLS]
    out_ptr,           # bf16 [rows, N_COLS]
    HEADS_C: ct.Constant[int],
    QUERY_C: ct.Constant[int],
    N_COLS_C: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    block = ct.bid(0)
    rows_1d = block * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
    batch = rows_1d // (HEADS_C * QUERY_C)
    head_query = rows_1d - batch * (HEADS_C * QUERY_C)
    head = head_query // QUERY_C
    query = head_query - head * QUERY_C

    grad = ct.astype(ct.load(grad_ptr, index=(block, 0), shape=(BLOCK_M, BLOCK_N)),
                     ct.float32)
    keep = ct.astype(ct.load(keep_ptr, index=(block, 0), shape=(BLOCK_M, BLOCK_N)),
                     ct.float32)
    logits = ct.astype(ct.load(logits_ptr, index=(block, 0), shape=(BLOCK_M, BLOCK_N)),
                       ct.float32)
    row_shift = ct.astype(ct.load(row_shift_ptr, index=(block,), shape=(BLOCK_M,)),
                          ct.float32)
    row_denom = ct.astype(ct.load(row_denom_ptr, index=(block,), shape=(BLOCK_M,)),
                          ct.float32)

    # mask_offsets = (batch*QUERY + query)*N_COLS + cols
    mask_row = batch * QUERY_C + query  # [BLOCK_M]
    cols_1d = ct.arange(BLOCK_N, dtype=ct.int32)
    rows_2d = ct.reshape(mask_row, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols_1d, (1, BLOCK_N))
    rows_bc = ct.broadcast_to(rows_2d, (BLOCK_M, BLOCK_N))
    cols_bc = ct.broadcast_to(cols_2d, (BLOCK_M, BLOCK_N))
    mask_src = ct.gather(mask_src_ptr, (rows_bc, cols_bc))
    final_mask = mask_src == ct.zeros((BLOCK_M, BLOCK_N), dtype=mask_src.dtype)

    shift_2d = ct.reshape(row_shift, (BLOCK_M, 1))
    denom_2d = ct.reshape(row_denom, (BLOCK_M, 1))
    probs = ct.exp(logits - shift_2d) / denom_2d
    scaled_grad = grad * (keep * DROPOUT_SCALE)
    product = scaled_grad * probs
    row_sum = ct.sum(product, axis=1, keepdims=True)
    fma = -probs * row_sum + product
    rounded = ct.astype(ct.astype(fma, ct.bfloat16), ct.float32)
    zero = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    selected = ct.where(final_mask, zero, rounded)
    out = ct.astype(selected * POST_SCALE, ct.bfloat16)
    ct.store(out_ptr, index=(block, 0), tile=out)

    # eq_out: only for head == 0 rows
    head_0 = head == 0  # [BLOCK_M]
    head_0_2d = ct.broadcast_to(ct.reshape(head_0, (BLOCK_M, 1)),
                                (BLOCK_M, BLOCK_N))
    ct.scatter(eq_out_ptr, (rows_bc, cols_bc), final_mask, mask=head_0_2d)


@oracle_impl(hardware="B200", point="83b2a800", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N):
    (
        arg0_1,  # bf16 (192, 128, 128)
        arg1_1,  # b8 (16, 12, 128, 128)
        arg2_1,  # bf16 (16, 12, 128, 128)
        arg3_1,  # f32 (16, 12, 128, 1)
        arg4_1,  # f32 (16, 12, 128, 1)
        arg5_1,  # b8 (16, 1, 128, 128)
        _shape_param_0,
        _shape_param_1,
    ) = inputs

    out_shape = tuple(int(dim) for dim in _shape_param_1)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    zero = torch.zeros((), device=arg0_1.device, dtype=torch.bfloat16)
    eq_out = torch.empty_strided(
        tuple(arg5_1.shape),
        tuple(arg5_1.stride()),
        device=arg0_1.device,
        dtype=torch.bool,
    )

    n_cols = int(arg2_1.shape[-1])
    n_rows = arg2_1.numel() // n_cols
    batch = int(arg2_1.shape[0])
    grad_2d = arg0_1.view(n_rows, n_cols)
    keep_2d = arg1_1.view(n_rows, n_cols)
    logits_2d = arg2_1.view(n_rows, n_cols)
    row_shift_1d = arg3_1.view(n_rows)
    row_denom_1d = arg4_1.view(n_rows)
    mask_src_2d = arg5_1.view(batch * QUERY, n_cols)
    eq_out_2d = eq_out.view(batch * QUERY, n_cols)
    out_2d = out.view(n_rows, n_cols)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _bert_softmax_bwd_kernel,
        (
            grad_2d, keep_2d, logits_2d, row_shift_1d, row_denom_1d,
            mask_src_2d, eq_out_2d, out_2d,
            HEADS, QUERY, N_COLS, BLOCK_M, BLOCK_N,
        ),
    )
    return zero, eq_out, out
