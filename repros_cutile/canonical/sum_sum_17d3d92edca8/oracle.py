"""cuTile port of sum_sum_17d3d92edca8: MegatronBERT LM-head softmax backward.

For each (row, col) in [ROWS=8192, VOCAB=29056]:
  probs = exp(logits[r, c] - row_shift0[r] - row_shift1[r])
  neg_scale = -numerator/denominator
  target = (col == labels_shifted[r]) & (label != -100) ? neg_scale : 0
  row_sum = (label != -100) ? neg_scale : 0
  softmax_grad = target - probs * row_sum   (bf16 rounded)
  out[r, c] = residual[r, c] + softmax_grad (bf16 rounded)

Then out_sum[c] = sum_r(out[r, c]).to(bf16).to(f32) — done in a second
finalize kernel (matches Triton's two-kernel structure: materialize +
partial column-reduce, then finalize).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
SEQ_IN = 513
SEQ_OUT = 512
ROWS = BATCH * SEQ_OUT
VOCAB = 29056


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _materialize_partial_kernel(
    labels_shifted_ptr,  # i64 [ROWS]
    logits_ptr,          # bf16 [ROWS, VOCAB]
    row_shift0_ptr,      # f32 [ROWS]
    row_shift1_ptr,      # f32 [ROWS]
    residual_ptr,        # bf16 [ROWS, VOCAB]
    neg_scale_row_ptr,   # f32 [ROWS]
    out_ptr,             # bf16 [ROWS, VOCAB]
    partial_ptr,         # f32 [VOCAB, NUM_ROW_TILES] strides (NUM_ROW_TILES, 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)

    cols = col_tile * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int64)
    cols_2d = ct.reshape(cols, (1, BLOCK_N))

    labels = ct.load(labels_shifted_ptr, index=(row_tile,), shape=(BLOCK_M,))
    labels_2d = ct.reshape(labels, (BLOCK_M, 1))
    label_active = labels_2d != -100

    neg_scale = ct.load(neg_scale_row_ptr, index=(row_tile,), shape=(BLOCK_M,))
    neg_scale_2d = ct.reshape(neg_scale, (BLOCK_M, 1))

    zero_f = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
    is_target = (cols_2d == labels_2d) & label_active
    target_grad = ct.where(is_target, neg_scale_2d + zero_f, zero_f)
    row_sum = neg_scale_2d + zero_f

    logits = ct.load(
        logits_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N),
    )
    logits_f = ct.astype(logits, ct.float32)
    row_shift0 = ct.load(row_shift0_ptr, index=(row_tile,), shape=(BLOCK_M,))
    row_shift1 = ct.load(row_shift1_ptr, index=(row_tile,), shape=(BLOCK_M,))
    row_shift0_2d = ct.reshape(row_shift0, (BLOCK_M, 1))
    row_shift1_2d = ct.reshape(row_shift1, (BLOCK_M, 1))

    shifted = logits_f - row_shift0_2d - row_shift1_2d
    probs = ct.exp(shifted)

    correction = target_grad - probs * row_sum
    correction_bf16 = ct.astype(correction, ct.bfloat16)

    residual = ct.load(
        residual_ptr, index=(row_tile, col_tile), shape=(BLOCK_M, BLOCK_N),
    )
    out_val = ct.astype(
        ct.astype(residual, ct.float32) + ct.astype(correction_bf16, ct.float32),
        ct.bfloat16,
    )
    ct.store(out_ptr, index=(row_tile, col_tile), tile=out_val)

    # Partial column-wise sum for this tile (BLOCK_N,) -> store into partial
    # buffer at row range [col_tile*BLOCK_N : (col_tile+1)*BLOCK_N], col row_tile.
    partial = ct.sum(ct.astype(out_val, ct.float32), axis=0)
    partial_col = ct.reshape(partial, (BLOCK_N, 1))
    ct.store(partial_ptr, index=(col_tile, row_tile), tile=partial_col)


@ct.kernel
def _finalize_sum_kernel(
    partial_ptr,     # f32 [VOCAB, NUM_ROW_TILES]
    out_sum_ptr,     # f32 [VOCAB]
    BLOCK_C: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
):
    col_tile = ct.bid(0)
    values = ct.load(
        partial_ptr, index=(col_tile, 0), shape=(BLOCK_C, BLOCK_TILES),
    )
    sums = ct.sum(values, axis=1)
    rounded = ct.astype(ct.astype(sums, ct.bfloat16), ct.float32)
    ct.store(out_sum_ptr, index=(col_tile,), tile=rounded)


@oracle_impl(hardware="B200", point="7b151c28", BLOCK_M=64, BLOCK_N=128, FINAL_BLOCK_C=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, FINAL_BLOCK_C: int):
    (
        arg0_1,   # f32 [] numerator
        arg1_1,   # f32 [] denominator
        arg2_1,   # i64 [16, 513] labels
        arg3_1,   # bf16 [16, 512, 29056] logits
        arg4_1,   # f32 [8192, 1] row_shift0
        arg5_1,   # f32 [8192, 1] row_shift1
        arg6_1,   # bf16 [16, 512, 29056] residual
        *_shape_params,
    ) = inputs
    device = arg3_1.device

    scale = arg0_1 / arg1_1
    neg_scale_scalar = -scale

    labels_shifted = arg2_1[:, 1:].contiguous().view(ROWS)
    label_active = labels_shifted != -100
    neg_scale_row = torch.where(
        label_active,
        neg_scale_scalar.expand(ROWS),
        torch.zeros((), device=device, dtype=torch.float32).expand(ROWS),
    ).contiguous()

    logits_2d = arg3_1.view(ROWS, VOCAB).contiguous() if not arg3_1.is_contiguous() else arg3_1.view(ROWS, VOCAB)
    residual_2d = arg6_1.view(ROWS, VOCAB).contiguous() if not arg6_1.is_contiguous() else arg6_1.view(ROWS, VOCAB)
    row_shift0_1d = arg4_1.view(ROWS)
    row_shift1_1d = arg5_1.view(ROWS)

    out_2d = torch.empty(
        (ROWS, VOCAB), device=device, dtype=torch.bfloat16,
    )

    num_row_tiles = ROWS // BLOCK_M
    n_col_tiles = VOCAB // BLOCK_N

    partial = torch.empty(
        (VOCAB, num_row_tiles), device=device, dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_row_tiles, n_col_tiles, 1),
        _materialize_partial_kernel,
        (
            labels_shifted, logits_2d, row_shift0_1d, row_shift1_1d, residual_2d,
            neg_scale_row, out_2d, partial, BLOCK_M, BLOCK_N,
        ),
    )

    out_sum = torch.empty((VOCAB,), device=device, dtype=torch.float32)
    block_tiles = _next_pow2(num_row_tiles)
    ct.launch(
        stream,
        (VOCAB // FINAL_BLOCK_C, 1, 1),
        _finalize_sum_kernel,
        (partial, out_sum, FINAL_BLOCK_C, block_tiles),
    )

    return out_2d, out_2d.permute(1, 0), out_sum
