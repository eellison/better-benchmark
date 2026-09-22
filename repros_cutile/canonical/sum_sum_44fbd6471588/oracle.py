"""cuTile port of sum_sum_44fbd6471588: Electra shifted causal-LM xent-backward.

Mirrors Triton's 3-kernel structure with `ct.sum` inside kernels:
  - `_materialize_kernel`: materialize padded output and dense output; produces
    per-row-block per-col-block column partials via `ct.sum(..., axis=0)`.
  - `_zero_pad_kernel`: zero the 6-col right pad on out_padded.
  - `_column_finalize_kernel`: sum partials across row-blocks via `ct.sum`.

BLOCK_M=8, BLOCK_N=512 match Triton.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
SEQ_IN = 513
SEQ_OUT = 512
ROWS = BATCH * SEQ_OUT  # 32768
VOCAB = 30522
VOCAB_PAD = 30528
PAD = VOCAB_PAD - VOCAB


def _next_p2(v):
    return 1 << (int(v) - 1).bit_length()


@ct.kernel
def _materialize_kernel(
    numerator_ptr,     # f32 (1,)
    denominator_ptr,   # f32 (1,)
    labels_ptr,        # i64 (BATCH * SEQ_IN,) — arg2 flat
    logits_ptr,        # bf16 (ROWS, VOCAB_PAD) — arg3 storage-padded
    row_shift0_ptr,    # f32 (ROWS,)
    row_shift1_ptr,    # f32 (ROWS,)
    residual_ptr,      # bf16 (ROWS, VOCAB) — arg6
    out_padded_ptr,    # bf16 (ROWS, VOCAB_PAD)
    partial_col_ptr,   # f32 (num_row_tiles, VOCAB_PAD)  — partials over rows-of-tile
    out_base_ptr,      # bf16 (ROWS, VOCAB)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    row_tile = ct.bid(0)
    col_tile = ct.bid(1)

    rows_1d = row_tile * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
    cols_1d = col_tile * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)
    row_mask = ct.reshape(rows_1d < ROWS, (BLOCK_M, 1))
    col_mask = ct.reshape(cols_1d < VOCAB, (1, BLOCK_N))
    mask = row_mask & col_mask

    num_tile = ct.load(numerator_ptr, index=(0,), shape=(1,))
    den_tile = ct.load(denominator_ptr, index=(0,), shape=(1,))
    scale = ct.astype(num_tile, ct.float32) / ct.astype(den_tile, ct.float32)

    # label offsets: arg2 is [BATCH, SEQ_IN]. rows -> (batch, seq_out) -> label_idx
    #   batch = rows // SEQ_OUT
    #   seq_out = rows - batch * SEQ_OUT
    #   label = arg2[batch, seq_out + 1]
    batch = rows_1d // SEQ_OUT
    seq_out = rows_1d - batch * SEQ_OUT
    label_offsets = batch * SEQ_IN + seq_out + 1
    raw_label = ct.gather(labels_ptr, label_offsets)
    active_label = raw_label != -100
    safe_label = ct.where(active_label, raw_label, 0)
    in_vocab = (safe_label >= 0) & (safe_label < VOCAB)

    row_scale = ct.where(active_label, ct.broadcast_to(scale, (BLOCK_M,)), 0.0)
    # scale_delta = scale - scale = 0 (approx; assume finite scale)
    finite_row_sum = ct.where(active_label & in_vocab, -row_scale, 0.0)
    row_sum = ct.where(active_label, finite_row_sum, 0.0)

    safe_label_2d = ct.reshape(safe_label, (BLOCK_M, 1))
    row_scale_2d = ct.reshape(row_scale, (BLOCK_M, 1))
    row_sum_2d = ct.reshape(row_sum, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols_1d, (1, BLOCK_N))
    one_hot = ct.where(safe_label_2d == cols_2d, -1.0, 0.0)
    one_hot_scaled = one_hot * row_scale_2d

    # logits are stored with row stride VOCAB_PAD; the (ROWS, VOCAB_PAD) view
    # covers valid storage entirely. Load and use padding-zero for OOB cols>=VOCAB.
    logits = ct.load(logits_ptr, index=(row_tile, col_tile),
                     shape=(BLOCK_M, BLOCK_N),
                     padding_mode=ct.PaddingMode.ZERO)
    logits_f = ct.astype(logits, ct.float32)
    row_shift0 = ct.load(row_shift0_ptr, index=(row_tile,), shape=(BLOCK_M,))
    row_shift1 = ct.load(row_shift1_ptr, index=(row_tile,), shape=(BLOCK_M,))
    rs0_2d = ct.reshape(row_shift0, (BLOCK_M, 1))
    rs1_2d = ct.reshape(row_shift1, (BLOCK_M, 1))

    centered = logits_f - rs0_2d - rs1_2d
    exp_values = ct.exp(centered)
    correction = one_hot_scaled - exp_values * row_sum_2d
    correction_bf16 = ct.astype(correction, ct.bfloat16)

    residual = ct.load(residual_ptr, index=(row_tile, col_tile),
                       shape=(BLOCK_M, BLOCK_N),
                       padding_mode=ct.PaddingMode.ZERO)
    out_f = ct.astype(residual, ct.float32) + ct.astype(correction_bf16, ct.float32)
    out_value = ct.astype(out_f, ct.bfloat16)

    # Store out_base (ROWS, VOCAB) — right-edge col_tile has OOB (VOCAB is not
    # multiple of BLOCK_N). Use scatter with mask.
    # Compute flat offsets and scatter.
    row_idx = ct.reshape(rows_1d, (BLOCK_M, 1))
    col_idx = ct.reshape(cols_1d, (1, BLOCK_N))
    base_offsets = row_idx * VOCAB + col_idx
    ct.scatter(out_base_ptr, base_offsets, out_value, mask=mask)

    # Store out_padded (ROWS, VOCAB_PAD) — right pad handled by _zero_pad_kernel.
    padded_offsets = row_idx * VOCAB_PAD + col_idx
    ct.scatter(out_padded_ptr, padded_offsets, out_value, mask=mask)

    # Column partial via ct.sum (mask OOB rows/cols).
    out_f_masked = ct.where(mask, ct.astype(out_value, ct.float32), 0.0)
    col_partial = ct.sum(out_f_masked, axis=0)  # (BLOCK_N,)
    partial_offsets = row_tile * VOCAB + cols_1d
    partial_mask = cols_1d < VOCAB
    ct.scatter(partial_col_ptr, partial_offsets, col_partial, mask=partial_mask)


@ct.kernel
def _zero_pad_kernel(
    out_padded_ptr,    # bf16 flat, viewed as (ROWS, VOCAB_PAD)
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    active = offsets < (ROWS * PAD)
    row = offsets // PAD
    pad_col = VOCAB + (offsets - row * PAD)
    real_offsets = row * VOCAB_PAD + pad_col
    zero = ct.zeros((BLOCK,), dtype=ct.bfloat16)
    ct.scatter(out_padded_ptr, real_offsets, zero, mask=active)


@ct.kernel
def _column_finalize_kernel(
    partial_ptr,       # f32 (NUM_ROW_TILES, VOCAB)
    out_sum_ptr,       # f32 (VOCAB,)
    NUM_ROW_TILES: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    col_tile = ct.bid(0)
    # Reduce partials over row-tiles. NUM_ROW_TILES is small (ROWS/BLOCK_M=4096).
    # Use iterative load with padding-zero.
    cols_1d = col_tile * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    col_active = cols_1d < VOCAB

    acc = ct.zeros((BLOCK_C,), dtype=ct.float32)
    # Iterate over NUM_ROW_TILES in chunks of BLOCK_R
    # (Compile-time loop unrolled via python range.)
    n_chunks = (NUM_ROW_TILES + BLOCK_R - 1) // BLOCK_R
    for chunk in range(n_chunks):
        row_start = chunk * BLOCK_R
        rows_1d = row_start + ct.arange(BLOCK_R, dtype=ct.int32)
        row_active_1d = rows_1d < NUM_ROW_TILES
        row_active = ct.reshape(row_active_1d, (BLOCK_R, 1))
        col_active_2d = ct.reshape(col_active, (1, BLOCK_C))
        active = row_active & col_active_2d

        rr = ct.reshape(rows_1d, (BLOCK_R, 1))
        cc = ct.reshape(cols_1d, (1, BLOCK_C))
        offsets = rr * VOCAB + cc
        values = ct.gather(partial_ptr, offsets)
        vals = ct.where(active, values, 0.0)
        chunk_sum = ct.sum(vals, axis=0)
        acc = acc + chunk_sum

    rounded = ct.astype(ct.astype(acc, ct.bfloat16), ct.float32)
    ct.scatter(out_sum_ptr, cols_1d, rounded, mask=col_active)


@oracle_impl(hardware="B200", point="d82d2bc6")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1,
     *_shape_params) = inputs

    device = arg3_1.device

    # Views
    labels_flat = arg2_1.contiguous().view(-1)  # (BATCH * SEQ_IN,)
    num_scalar = arg0_1.view(1)
    den_scalar = arg1_1.view(1)
    # arg3 storage stride is (VOCAB_PAD, 1). The valid element count is
    # 32767*30528 + 30522, so (ROWS, VOCAB) is a legal strided view (the last
    # row's 30522nd element ends exactly at storage end); (ROWS, VOCAB_PAD)
    # would read past storage. Use (ROWS, VOCAB) with the padded row stride.
    arg3_2d = arg3_1.as_strided((ROWS, VOCAB), (VOCAB_PAD, 1),
                                storage_offset=arg3_1.storage_offset())
    arg4_1d = arg4_1.view(ROWS).contiguous()
    arg5_1d = arg5_1.view(ROWS).contiguous()
    arg6_2d = arg6_1.reshape(ROWS, VOCAB)

    BLOCK_M = 8
    BLOCK_N = 512
    num_row_tiles = (ROWS + BLOCK_M - 1) // BLOCK_M  # 4096
    num_col_tiles = (VOCAB + BLOCK_N - 1) // BLOCK_N  # 60

    out_padded = torch.empty_strided(
        (ROWS, VOCAB_PAD), (VOCAB_PAD, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_base = torch.empty_strided(
        (ROWS, VOCAB), (VOCAB, 1),
        device=device, dtype=torch.bfloat16,
    )
    out_sum = torch.empty((VOCAB,), device=device, dtype=torch.float32)
    partial = torch.empty((num_row_tiles, VOCAB), device=device, dtype=torch.float32)

    # Flat views for scatter access
    def _flat(t):
        return t.as_strided((t.numel(),), (1,))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_row_tiles, num_col_tiles, 1),
        _materialize_kernel,
        (num_scalar, den_scalar, labels_flat, arg3_2d,
         arg4_1d, arg5_1d, arg6_2d,
         _flat(out_padded), _flat(partial), _flat(out_base),
         BLOCK_M, BLOCK_N),
    )
    ZERO_PAD_BLOCK = 1024
    ct.launch(
        stream, (((ROWS * PAD) + ZERO_PAD_BLOCK - 1) // ZERO_PAD_BLOCK, 1, 1),
        _zero_pad_kernel,
        (_flat(out_padded), ZERO_PAD_BLOCK),
    )
    FINAL_BLOCK_C = 128
    FINAL_BLOCK_R = 64
    ct.launch(
        stream, ((VOCAB + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _column_finalize_kernel,
        (_flat(partial), out_sum,
         num_row_tiles, FINAL_BLOCK_C, FINAL_BLOCK_R),
    )

    return out_padded, out_base.permute(1, 0), out_sum
