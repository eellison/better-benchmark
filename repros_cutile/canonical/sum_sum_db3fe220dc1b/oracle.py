"""cuTile port of sum_sum_db3fe220dc1b: GPT-J shifted-label cross-entropy backward.

Mirrors Triton's single-kernel structure: for each BLOCK_N column block, load
all 128 rows and compute the backward + residual add, store bf16, and reduce
across rows to produce the per-column bf16-rounded f32 sum.

Vocab is 50400 (not a power of 2). Pad the output to a multiple of BLOCK_N and
slice afterwards.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
VOCAB = 50400


@ct.kernel
def _backward_and_sum_kernel(
    numerator_ptr,     # f32 ()
    denominator_ptr,   # f32 ()
    labels_ptr,        # i64 (129,)
    logits_ptr,        # bf16 (ROWS, VOCAB_PAD)
    row_shift0_ptr,    # f32 (ROWS,)
    row_shift1_ptr,    # f32 (ROWS,)
    residual_ptr,      # bf16 (ROWS, VOCAB_PAD)
    out_base_ptr,      # bf16 (ROWS, VOCAB_PAD)
    out_sum_ptr,       # f32 (VOCAB_PAD,)
    ROWS_C: ct.Constant[int],
    VOCAB_C: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    col_block = ct.bid(0)
    col_base = col_block * BLOCK_N

    # Column indices, shape (BLOCK_N,)
    cols_1d = ct.arange(BLOCK_N, dtype=ct.int64) + col_base

    # Row indices, shape (ROWS_C,)
    rows_1d = ct.arange(ROWS_C, dtype=ct.int64)
    # Labels: labels_ptr has shape (129,); we want labels[1 + row]
    raw_label = ct.gather(labels_ptr, rows_1d + 1)
    minus100 = ct.full((ROWS_C,), -100, dtype=ct.int64)
    zero_i64 = ct.zeros((ROWS_C,), dtype=ct.int64)
    active = raw_label != minus100
    safe_label = ct.where(active, raw_label, zero_i64)  # (ROWS_C,)

    # Scalar loads (scalar tensors viewed as (1,))
    numerator = ct.load(numerator_ptr, index=(0,), shape=(1,))
    denominator = ct.load(denominator_ptr, index=(0,), shape=(1,))
    numerator_f = ct.astype(numerator, ct.float32)
    denominator_f = ct.astype(denominator, ct.float32)
    scale_1 = numerator_f / denominator_f  # (1,)
    # Broadcast to (ROWS_C,)
    zero_f = ct.zeros((ROWS_C,), dtype=ct.float32)
    scale_bcast = zero_f + scale_1  # broadcast scalar
    row_scale = ct.where(active, scale_bcast, zero_f)

    # row_sum handling — replicate the finite/NaN handling from Triton.
    scale_delta = scale_bcast - scale_bcast
    scale_is_finite = scale_delta == zero_f
    neg_one_f = ct.full((ROWS_C,), -1.0, dtype=ct.float32)
    vocab_full = ct.full((ROWS_C,), VOCAB_C, dtype=ct.int64)
    zero_l = ct.zeros((ROWS_C,), dtype=ct.int64)
    in_vocab = (safe_label >= zero_l) & (safe_label < vocab_full)
    finite_row_sum = ct.where(in_vocab, neg_one_f * row_scale, zero_f)
    active_row_sum = ct.where(scale_is_finite, finite_row_sum, scale_delta)
    row_sum = ct.where(active, active_row_sum, zero_f)

    # Broadcast to (ROWS_C, BLOCK_N)
    rows_col = ct.reshape(rows_1d, (ROWS_C, 1))
    cols_row = ct.reshape(cols_1d, (1, BLOCK_N))
    safe_label_col = ct.reshape(safe_label, (ROWS_C, 1))
    row_scale_col = ct.reshape(row_scale, (ROWS_C, 1))
    row_sum_col = ct.reshape(row_sum, (ROWS_C, 1))

    # one_hot: (safe_label == cols) ? -1 : 0
    one_hot = ct.where(safe_label_col == cols_row,
                       ct.full((ROWS_C, BLOCK_N), -1.0, dtype=ct.float32),
                       ct.zeros((ROWS_C, BLOCK_N), dtype=ct.float32))
    one_hot_scaled = one_hot * row_scale_col

    # Load logits (bf16). Use padding_mode=ZERO for OOB tail (VOCAB not multiple of BLOCK_N).
    logits = ct.load(logits_ptr, index=(0, col_block), shape=(ROWS_C, BLOCK_N),
                     padding_mode=ct.PaddingMode.ZERO)
    logits_f = ct.astype(logits, ct.float32)

    row_shift0 = ct.load(row_shift0_ptr, index=(0,), shape=(ROWS_C,))
    row_shift1 = ct.load(row_shift1_ptr, index=(0,), shape=(ROWS_C,))
    row_shift0_col = ct.reshape(row_shift0, (ROWS_C, 1))
    row_shift1_col = ct.reshape(row_shift1, (ROWS_C, 1))

    centered = logits_f - row_shift0_col - row_shift1_col
    exp_values = ct.exp(centered)

    correction = one_hot_scaled - exp_values * row_sum_col
    correction_bf = ct.astype(correction, ct.bfloat16)

    residual = ct.load(residual_ptr, index=(0, col_block), shape=(ROWS_C, BLOCK_N),
                       padding_mode=ct.PaddingMode.ZERO)
    # bf16 add via f32 boundary
    out_f = ct.astype(residual, ct.float32) + ct.astype(correction_bf, ct.float32)
    out_bf = ct.astype(out_f, ct.bfloat16)

    ct.store(out_base_ptr, index=(0, col_block), tile=out_bf)

    # Column sum: sum along rows -> (BLOCK_N,), bf16-round then to f32.
    col_sum = ct.sum(ct.astype(out_bf, ct.float32), axis=0)
    rounded_sum = ct.astype(ct.astype(col_sum, ct.bfloat16), ct.float32)
    ct.store(out_sum_ptr, index=(col_block,), tile=rounded_sum)


def _shape_tuple(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="4e3e4a7e", BLOCK_N=64)
def oracle_forward(inputs, *, BLOCK_N: int):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6,
        _shape0, _shape1, _shape2, _shape3, _shape4, shape5,
    ) = inputs
    device = arg3.device

    # Pad logits/residual/out to multiple of BLOCK_N along vocab axis.
    n_tiles = (VOCAB + BLOCK_N - 1) // BLOCK_N
    padded_vocab = n_tiles * BLOCK_N

    x_flat = arg3.view(ROWS, VOCAB)
    residual_flat = arg6.view(ROWS, VOCAB)

    padded_x = torch.zeros((ROWS, padded_vocab), device=device, dtype=torch.bfloat16)
    padded_x[:, :VOCAB].copy_(x_flat)
    padded_residual = torch.zeros((ROWS, padded_vocab), device=device, dtype=torch.bfloat16)
    padded_residual[:, :VOCAB].copy_(residual_flat)

    row_shift0 = arg4.view(ROWS)
    row_shift1 = arg5.view(ROWS)

    labels_flat = arg2.view(-1)  # (129,)

    padded_out = torch.empty((ROWS, padded_vocab), device=device, dtype=torch.bfloat16)
    padded_out_sum = torch.empty((padded_vocab,), device=device, dtype=torch.float32)

    numerator_1d = arg0.view(1)
    denominator_1d = arg1.view(1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n_tiles, 1, 1),
        _backward_and_sum_kernel,
        (numerator_1d, denominator_1d, labels_flat, padded_x, row_shift0, row_shift1,
         padded_residual, padded_out, padded_out_sum,
         ROWS, VOCAB, BLOCK_N),
    )
    out_base = padded_out[:, :VOCAB].contiguous()
    out_sum = padded_out_sum[:VOCAB].contiguous()

    return out_base, out_base.permute(1, 0), out_sum
