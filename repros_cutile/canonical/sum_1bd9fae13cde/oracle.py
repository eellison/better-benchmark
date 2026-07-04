"""cuTile port of sum_1bd9fae13cde: DistillGPT2 cross-entropy backward.

For each row block (r in [0, 16384)) and column block (c in [0, 50257/2048)):
  scale = numerator/denominator; row_scale = scale if label != -100 else 0
  logits (bf16) - shift0 - shift1 -> exp
  sparse = (col == label && valid) ? -scale : 0
  out = sparse - exp * row_scale
Both logits and out are 3D [32, 512, VOCAB]. logits has stride 50264 (padded),
out has stride VOCAB (contiguous).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
SEQ_OUT = 512
SEQ_IN = 513
ROWS = BATCH * SEQ_OUT
VOCAB = 50257
LOGITS_ROW_STRIDE = 50264


@ct.kernel
def _xent_bwd_kernel(
    numerator_ptr,     # f32 () -> (1,)
    denominator_ptr,   # f32 () -> (1,)
    labels_ptr,        # i64 flat (rows,) — post shift
    logits_flat_ptr,   # bf16 flat (rows * LOGITS_ROW_STRIDE,)
    row_shift0_ptr,    # f32 (rows,)
    row_shift1_ptr,    # f32 (rows,)
    out_flat_ptr,      # bf16 flat (rows * VOCAB,)
    BLOCK_N: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)

    num_tile = ct.load(numerator_ptr, index=(0,), shape=(1,))
    den_tile = ct.load(denominator_ptr, index=(0,), shape=(1,))
    num_scalar = ct.reshape(ct.astype(num_tile, ct.float32), ())
    den_scalar = ct.reshape(ct.astype(den_tile, ct.float32), ())
    scale = num_scalar / den_scalar

    label_tile = ct.load(labels_ptr, index=(row,), shape=(1,))
    label_scalar = ct.reshape(label_tile, ())
    active = label_scalar != -100
    zero_f = ct.astype(ct.zeros((1,), dtype=ct.float32), ct.float32)
    zero_f_scalar = ct.reshape(zero_f, ())
    # row_sum = -scale if active else 0
    neg_scale = -scale
    row_sum_scalar = ct.where(active, neg_scale, zero_f_scalar)
    # one_hot_scaled = -scale if match else 0 (row_scale = scale if active else 0)
    row_scale = ct.where(active, scale, zero_f_scalar)

    shift0_tile = ct.load(row_shift0_ptr, index=(row,), shape=(1,))
    shift0 = ct.reshape(ct.astype(shift0_tile, ct.float32), ())
    shift1_tile = ct.load(row_shift1_ptr, index=(row,), shape=(1,))
    shift1 = ct.reshape(ct.astype(shift1_tile, ct.float32), ())

    cols = ct.arange(BLOCK_N, dtype=ct.int32) + col_block * BLOCK_N
    col_mask = cols < VOCAB
    safe_cols = ct.where(col_mask, cols, ct.zeros((BLOCK_N,), dtype=ct.int32))
    logits_offsets = row * LOGITS_ROW_STRIDE + safe_cols
    logits = ct.gather(logits_flat_ptr, logits_offsets, mask=col_mask, padding_value=0.0)
    logits_f = ct.astype(logits, ct.float32)
    shifted = logits_f - shift0 - shift1
    exp_val = ct.exp(shifted)

    label_i32 = ct.astype(label_scalar, ct.int32)
    is_label = cols == label_i32
    # one_hot_scaled = -1 * row_scale = -scale if match+active else 0
    neg_row_scale_bc = -row_scale
    sparse = ct.where(is_label & active,
                      ct.full((BLOCK_N,), 0.0, dtype=ct.float32) + neg_row_scale_bc,
                      ct.zeros((BLOCK_N,), dtype=ct.float32))
    # exp_times_sum = exp * row_sum (row_sum = -scale if active else 0)
    dense = exp_val * row_sum_scalar
    out = sparse - dense
    out_offsets = row * VOCAB + cols
    ct.scatter(out_flat_ptr, out_offsets, ct.astype(out, ct.bfloat16), mask=col_mask)


@oracle_impl(hardware="B200", point="ceaa9c1c", BLOCK_N=2048)
def oracle_forward(inputs, *, BLOCK_N: int):
    numerator, denominator, labels, logits, row_shift0, row_shift1, *_shapes = inputs
    out = torch.empty_strided(
        (BATCH, SEQ_OUT, VOCAB),
        (SEQ_OUT * VOCAB, VOCAB, 1),
        device=logits.device,
        dtype=torch.bfloat16,
    )

    # labels: shift by +1, dropping label[0], to form (rows,) i64.
    labels_shifted = labels[:, 1:].contiguous().view(-1)

    numerator_view = numerator.view(1)
    denominator_view = denominator.view(1)
    row_shift0_1d = row_shift0.view(ROWS)
    row_shift1_1d = row_shift1.view(ROWS)
    # logits storage is (ROWS-1) * LOGITS_ROW_STRIDE + VOCAB.
    storage_size = (ROWS - 1) * LOGITS_ROW_STRIDE + VOCAB
    logits_flat = logits.as_strided((storage_size,), (1,),
                                    storage_offset=logits.storage_offset())
    out_flat = out.view(-1)

    stream = torch.cuda.current_stream()
    n_col_blocks = (VOCAB + BLOCK_N - 1) // BLOCK_N
    ct.launch(
        stream,
        (ROWS, n_col_blocks, 1),
        _xent_bwd_kernel,
        (
            numerator_view, denominator_view, labels_shifted,
            logits_flat, row_shift0_1d, row_shift1_1d, out_flat, BLOCK_N,
        ),
    )
    return out
