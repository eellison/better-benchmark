"""cuTile port of sum_sum_2aec3d12d3d5: RobertaForCausalLM loss-backward fragment.

Ports the three Triton kernels:
  _materialize_kernel: per-row scale = numerator/denominator, one-hot label
     gradient minus exp(shifted)*row_sum correction, bf16-add residual,
     stores both rows and transposed layouts.
  _zero_pad_kernel: writes zeros to the tail-padded region (cols in
     [VOCAB, VOCAB_PAD)).
  _column_sum_kernel: reduces the row-materialized outputs across rows to
     produce a bf16-rounded f32 vocabulary sum.

RTNE downcasts and add.rn/sub.rn/mul.rn/div.rn are the cuTile defaults, so
we replace inline PTX and `.to(tl.bfloat16, fp_downcast_rounding="rtne")`
with regular arithmetic and `ct.astype(_, ct.bfloat16)`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
SEQ = 512
VOCAB = 50265
VOCAB_PAD = 50272
PAD = 7


@ct.kernel
def _materialize_kernel(
    numerator_ptr,   # f32 []
    denominator_ptr, # f32 []
    labels_ptr,      # i64 [32, 513]
    logits_ptr,      # bf16 [16384, VOCAB_PAD]
    row_shift0_ptr,  # f32 [ROWS, 1]
    row_shift1_ptr,  # f32 [ROWS, 1]
    residual_ptr,    # bf16 [ROWS, VOCAB]
    out_rows_ptr,    # bf16 [ROWS, VOCAB_PAD]
    out_trans_ptr,   # bf16 [VOCAB_PAD, ROWS]
    ROWS_C: ct.Constant[int],
    SEQ_C: ct.Constant[int],
    VOCAB_C: ct.Constant[int],
    VOCAB_PAD_C: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    row_idx = ct.arange(BLOCK_M, dtype=ct.int32) + row_block * BLOCK_M
    col_idx = ct.arange(BLOCK_N, dtype=ct.int32) + col_block * BLOCK_N
    row_active = row_idx < ROWS_C
    col_active = col_idx < VOCAB_C

    # Load labels — labels_ptr is [32, 513]; label_offsets is rows + rows//SEQ + 1
    # in flattened element space, so we use gather from a flattened view.
    label_offsets_32 = row_idx + (row_idx // SEQ_C) + 1
    label_offsets = ct.astype(label_offsets_32, ct.int64)
    labels = ct.gather(labels_ptr, label_offsets, mask=row_active, padding_value=-100)
    active_label = labels != -100
    safe_label = ct.where(active_label, labels, ct.astype(0, ct.int64))
    in_vocab = (safe_label >= 0) & (safe_label < ct.astype(VOCAB_C, ct.int64))

    numerator_1 = ct.load(numerator_ptr, index=(0,), shape=(1,))
    denominator_1 = ct.load(denominator_ptr, index=(0,), shape=(1,))
    scale_1 = numerator_1 / denominator_1
    # Reduce to a true scalar so it broadcasts naturally.
    scale = ct.sum(scale_1)

    zero_f32 = ct.astype(0.0, ct.float32)
    neg_one_f32 = ct.astype(-1.0, ct.float32)
    row_scale = ct.where(active_label, scale, zero_f32)

    scale_delta = scale - scale
    scale_is_finite = scale_delta == zero_f32
    finite_row_sum = ct.where(in_vocab, neg_one_f32 * row_scale, zero_f32)
    active_row_sum = ct.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = ct.where(active_label, active_row_sum, zero_f32)

    # one_hot[i, j] = -1.0 if safe_label[i]==col_idx[j] else 0.0.
    safe_label_2d = ct.reshape(safe_label, (BLOCK_M, 1))
    col_idx_2d = ct.reshape(ct.astype(col_idx, ct.int64), (1, BLOCK_N))
    one_hot = ct.where(safe_label_2d == col_idx_2d, neg_one_f32, zero_f32)
    row_scale_2d = ct.reshape(row_scale, (BLOCK_M, 1))
    sparse_grad = one_hot * row_scale_2d

    # Load logits — 2D tile shape (BLOCK_M, BLOCK_N).
    logits_bf = ct.load(
        logits_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    logits_f = ct.astype(logits_bf, ct.float32)

    row_shift0 = ct.load(
        row_shift0_ptr, index=(row_block, 0), shape=(BLOCK_M, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    row_shift1 = ct.load(
        row_shift1_ptr, index=(row_block, 0), shape=(BLOCK_M, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    centered = logits_f - row_shift0
    centered = centered - row_shift1
    exp_values = ct.exp(centered)

    row_sum_2d = ct.reshape(row_sum, (BLOCK_M, 1))
    correction = sparse_grad - exp_values * row_sum_2d
    correction_bf = ct.astype(correction, ct.bfloat16)

    # residual_ptr is [ROWS, VOCAB], not padded, and the store is to VOCAB_PAD.
    # For cols in [VOCAB, VOCAB_PAD), residual is OOB — zero padding.
    residual = ct.load(
        residual_ptr, index=(row_block, col_block), shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
    )
    out_value_f = ct.astype(residual, ct.float32) + ct.astype(correction_bf, ct.float32)
    out_value = ct.astype(out_value_f, ct.bfloat16)

    # Zero out any position whose col_idx is >= VOCAB (padded columns).
    col_active_2d = ct.reshape(col_active, (1, BLOCK_N))
    row_active_2d = ct.reshape(row_active, (BLOCK_M, 1))
    active_2d = row_active_2d & col_active_2d
    zero_bf = ct.full(shape=(BLOCK_M, BLOCK_N), fill_value=0.0, dtype=ct.bfloat16)
    out_value = ct.where(active_2d, out_value, zero_bf)

    ct.store(out_rows_ptr, index=(row_block, col_block), tile=out_value)
    out_value_t = ct.transpose(out_value)  # (BLOCK_N, BLOCK_M)
    ct.store(out_trans_ptr, index=(col_block, row_block), tile=out_value_t)


@ct.kernel
def _column_sum_kernel(
    out_rows_ptr,    # bf16 [ROWS, VOCAB_PAD]
    out_sum_ptr,     # f32 [VOCAB]
    ROWS_C: ct.Constant[int],
    VOCAB_C: ct.Constant[int],
    VOCAB_PAD_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    N_ROW_TILES: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    col_block = ct.bid(0)
    acc = ct.full(shape=(BLOCK_N,), fill_value=0.0, dtype=ct.float32)
    for r_tile in range(N_ROW_TILES):
        vals = ct.load(
            out_rows_ptr, index=(r_tile, col_block), shape=(BLOCK_R, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = acc + ct.sum(ct.astype(vals, ct.float32), axis=0)
    rounded = ct.astype(ct.astype(acc, ct.bfloat16), ct.float32)
    ct.store(out_sum_ptr, index=(col_block,), tile=rounded)


@oracle_impl(
    hardware="B200",
    point="020b9e29",
    BLOCK_M=16,
    BLOCK_N=256,
    FINAL_BLOCK_N=128,
    FINAL_BLOCK_R=64,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    FINAL_BLOCK_N: int,
    FINAL_BLOCK_R: int,
):
    (
        numerator,
        denominator,
        labels,
        logits,
        row_shift0,
        row_shift1,
        residual,
        _s0, _s1, _s2, _s3, _s4, _s5, _s6, _s7,
    ) = inputs
    device = logits.device

    out_rows = torch.empty_strided(
        (ROWS, VOCAB_PAD), (VOCAB_PAD, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_trans = torch.empty_strided(
        (VOCAB_PAD, ROWS), (ROWS, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_sum = torch.empty_strided(
        (VOCAB,), (1,),
        device=device, dtype=torch.float32,
    )

    # residual is [32, 512, VOCAB] contiguous but ROWS=32*512=16384.
    residual_2d = residual.reshape(ROWS, VOCAB)
    # logits has shape [32, 512, VOCAB=50265] with stride [25739264, 50272, 1] —
    # the underlying storage is exactly enough for [ROWS, VOCAB] with stride
    # [VOCAB_PAD, 1] (last valid offset = 823,656,440). The last 7 elements
    # in each row are padding that isn't backed by storage, but our kernel
    # uses padding_mode=ZERO for cols >= VOCAB, so we never dereference them.
    logits_2d = torch.as_strided(logits, (ROWS, VOCAB), (VOCAB_PAD, 1))
    # labels: [32, 513]; the Triton kernel indexes labels_ptr as a flat 1D
    # array. Provide a 1D flattened view.
    labels_1d = labels.reshape(-1)
    # row_shift0/1 are [ROWS, 1] already.

    stream = torch.cuda.current_stream()
    # Reshape 0-dim scalars to length-1 so cuTile can load them.
    numerator_1d = numerator.view(1)
    denominator_1d = denominator.view(1)
    ct.launch(
        stream,
        ((ROWS + BLOCK_M - 1) // BLOCK_M, (VOCAB + BLOCK_N - 1) // BLOCK_N, 1),
        _materialize_kernel,
        (
            numerator_1d, denominator_1d, labels_1d, logits_2d,
            row_shift0, row_shift1, residual_2d,
            out_rows, out_trans,
            ROWS, SEQ, VOCAB, VOCAB_PAD, BLOCK_M, BLOCK_N,
        ),
    )
    # Column reduction.
    n_row_tiles = (ROWS + FINAL_BLOCK_R - 1) // FINAL_BLOCK_R
    ct.launch(
        stream,
        ((VOCAB + FINAL_BLOCK_N - 1) // FINAL_BLOCK_N, 1, 1),
        _column_sum_kernel,
        (
            out_rows, out_sum,
            ROWS, VOCAB, VOCAB_PAD,
            FINAL_BLOCK_N, n_row_tiles, FINAL_BLOCK_R,
        ),
    )
    return out_rows, out_trans, out_sum
