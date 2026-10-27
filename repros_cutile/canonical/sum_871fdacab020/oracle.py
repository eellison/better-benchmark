"""cuTile port of sum_871fdacab020: BERT masked scaled softmax-backward.

For each row: dropout-scale grad, bf16-rounded scale of logits by 1/8, mask
fill to -998244352 where final_mask is True, exp((masked - row_shift)/row_denom),
grad*probs sum reduction, FMA epilogue, bf16 rounding, mask fill again, /8.

cuTile fp32/bf16 native rounding is RNE, matching the Triton PTX `add.rn`,
`mul.rn`, `cvt.rn.bf16.f32` intrinsics.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


DROPOUT_SCALE_F32 = 1.1111111640930176
MASK_FILL_F32 = -998244352.0
POST_SCALE_F32 = 0.125


def _round_bf16(x_f):
    return ct.astype(ct.astype(x_f, ct.bfloat16), ct.float32)


@ct.kernel
def _bert_softmax_bwd_kernel(
    grad_ptr,        # bf16 [N_ROWS, N_COLS]
    keep_ptr,        # bool [N_ROWS, N_COLS]
    logits_ptr,      # bf16 [N_ROWS, N_COLS]
    final_mask_ptr,  # bool [BATCH, 1, QUERY, N_COLS] flattened [BATCH*QUERY, N_COLS]
    row_shift_ptr,   # f32 [N_ROWS]
    row_denom_ptr,   # f32 [N_ROWS]
    fill_ptr,        # bf16 scalar
    out_ptr,         # bf16 [N_ROWS, N_COLS]
    HEADS: ct.Constant[int],
    QUERY: ct.Constant[int],
    N_COLS: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    # BLOCK_N == N_COLS assumed (128 == 128)

    grad = ct.load(grad_ptr, index=(row_tile, 0), shape=(BLOCK_M, BLOCK_N))
    keep = ct.load(keep_ptr, index=(row_tile, 0), shape=(BLOCK_M, BLOCK_N))
    logits = ct.load(logits_ptr, index=(row_tile, 0), shape=(BLOCK_M, BLOCK_N))

    # rows for this tile
    row_base = row_tile * BLOCK_M
    row_ids = row_base + ct.arange(BLOCK_M, dtype=ct.int32)
    # final_mask index: batch = row // (HEADS*QUERY); query = row % QUERY
    # final_mask_offsets = (batch * QUERY + query) * N_COLS + col
    #                    = batch*QUERY*N_COLS + query*N_COLS + col
    batch = row_ids // (HEADS * QUERY)
    query = row_ids % QUERY
    final_row = batch * QUERY + query  # shape (BLOCK_M,)
    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    final_row_2d = ct.reshape(final_row, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    final_row_bcast = ct.broadcast_to(final_row_2d, (BLOCK_M, BLOCK_N))
    cols_bcast = ct.broadcast_to(cols_2d, (BLOCK_M, BLOCK_N))
    final_mask = ct.gather(final_mask_ptr, (final_row_bcast, cols_bcast))

    row_shift = ct.load(row_shift_ptr, index=(row_tile,), shape=(BLOCK_M,))
    row_denom = ct.load(row_denom_ptr, index=(row_tile,), shape=(BLOCK_M,))
    fill_tile = ct.load(fill_ptr, index=(0,), shape=(1,))

    grad_f = ct.astype(grad, ct.float32)
    keep_f = ct.astype(keep, ct.float32)
    logits_f = ct.astype(logits, ct.float32)

    # bf16-round: logits/8 as bf16
    scaled_logits = _round_bf16(logits_f * POST_SCALE_F32)
    # Convert final_mask (bool) via where.
    masked_logits = ct.where(final_mask, MASK_FILL_F32, scaled_logits)
    row_shift_2d = ct.reshape(row_shift, (BLOCK_M, 1))
    row_denom_2d = ct.reshape(row_denom, (BLOCK_M, 1))
    probs = ct.exp(masked_logits - row_shift_2d) / row_denom_2d

    scaled_grad = grad_f * (keep_f * DROPOUT_SCALE_F32)
    product = scaled_grad * probs
    row_sum = ct.sum(product, axis=1, keepdims=True)
    fma = -probs * row_sum + product
    rounded_bf16 = ct.astype(fma, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf16, ct.float32)
    fill_f = ct.astype(fill_tile, ct.float32)
    selected = ct.where(final_mask, fill_f, rounded_f)
    out = ct.astype(selected * POST_SCALE_F32, ct.bfloat16)
    ct.store(out_ptr, index=(row_tile, 0), tile=out)


@oracle_impl(hardware="B200", point="202893ec", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    (
        arg0_1,  # grad bf16 [192,128,128]
        arg1_1,  # keep bool [16,12,128,128]
        arg2_1,  # logits bf16 [192,128,128]
        arg3_1,  # final_mask bool [16,1,128,128]
        arg4_1,  # row_shift f32 [16,12,128,1]
        arg5_1,  # row_denom f32 [16,12,128,1]
        arg6_1,  # fill bf16 []
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
    n_cols = int(out_shape[-1])
    n_rows = arg0_1.numel() // n_cols

    # Reshape everything to 2D [n_rows, n_cols].
    grad_2d = arg0_1.view(n_rows, n_cols)
    keep_2d = arg1_1.view(n_rows, n_cols)
    logits_2d = arg2_1.view(n_rows, n_cols)
    # final_mask is [16,1,128,128]; broadcast dim is 1. Its stride[1] is 0
    # (broadcast). Flatten to [16*128, 128]: batch,query,col.
    # Verify: (batch*QUERY + query) * N_COLS + col. That matches
    # arg3_1.view(16*128, 128).
    fmask_2d = arg3_1.view(int(arg3_1.shape[0]) * int(arg3_1.shape[2]), n_cols)
    row_shift_1d = arg4_1.view(-1)
    row_denom_1d = arg5_1.view(-1)
    fill_1d = arg6_1.view(1)

    heads = int(arg1_1.shape[1])
    query = int(arg1_1.shape[2])

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_rows // BLOCK_M, 1, 1),
        _bert_softmax_bwd_kernel,
        (
            grad_2d, keep_2d, logits_2d, fmask_2d,
            row_shift_1d, row_denom_1d, fill_1d, out.view(n_rows, n_cols),
            heads, query, n_cols, BLOCK_M, BLOCK_N,
        ),
    )
    return out
