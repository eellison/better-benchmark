"""cuTile port of pointwise_faf725b387f9: MobileBERT vocab-layout cat/pad/transpose.

Two outputs both derived from concatenation:
  padded[row, col]     : bf16 [ROWS=512, PADDED_VOCAB=30528]
  unpadded_t[col, row] : bf16 [VOCAB=30522, ROWS=512]   (stride (1, VOCAB))

Source region:
  rows [0, 128)   x cols [0, 30522): embed[col, row]  (i.e. transpose of embed[30522, 128])
  rows [128, 512) x cols [0, 30522): logits[row-128, col]
  cols [30522, 30528): 0
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


VOCAB = 30522
PADDED_VOCAB = 30528
FIRST_ROWS = 128
SECOND_ROWS = 384
ROWS = FIRST_ROWS + SECOND_ROWS  # 512


@ct.kernel
def _fill_zero_pad_kernel(padded_ptr, PADDED_VOCAB: ct.Constant[int], PAD_COL_START: ct.Constant[int], PAD_COLS: ct.Constant[int]):
    r = ct.bid(0)
    c_off = ct.bid(1)
    # Store 6 columns of zeros at cols [PAD_COL_START, PAD_COL_START+PAD_COLS)
    zero = ct.zeros(shape=(1, PAD_COLS), dtype=ct.bfloat16)
    ct.store(padded_ptr, index=(r, 0), tile=ct.reshape(zero, (1, PAD_COLS)))


@ct.kernel
def _embed_transpose_kernel(
    embed_ptr,        # f32 [VOCAB, FIRST_ROWS]
    padded_ptr,       # bf16 [ROWS, PADDED_VOCAB]
    unpadded_t_ptr,   # bf16 [VOCAB, ROWS]  strided (1, VOCAB)
    BLOCK_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    FIRST_ROWS: ct.Constant[int],
):
    c_blk = ct.bid(0)
    r_blk = ct.bid(1)
    # Load tile embed[c_blk*BLOCK_C:(c_blk+1)*BLOCK_C, r_blk*BLOCK_R:(r_blk+1)*BLOCK_R]
    tile = ct.load(embed_ptr, index=(c_blk, r_blk), shape=(BLOCK_C, BLOCK_R))
    tile_bf = ct.astype(tile, ct.bfloat16)
    # padded[r_blk*BLOCK_R:..., c_blk*BLOCK_C:...] = transpose
    tile_t = ct.transpose(tile_bf)  # (BLOCK_R, BLOCK_C)
    ct.store(padded_ptr, index=(r_blk, c_blk), tile=tile_t)
    # unpadded_t[c_blk*BLOCK_C:, r_blk*BLOCK_R:] = tile_bf directly
    ct.store(unpadded_t_ptr, index=(c_blk, r_blk), tile=tile_bf)


@ct.kernel
def _logits_copy_kernel(
    logits_ptr,       # f32 [SECOND_ROWS, VOCAB]
    padded_ptr,       # bf16 [ROWS, PADDED_VOCAB]
    unpadded_t_ptr,   # bf16 [VOCAB, ROWS] strided (1, VOCAB)
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    FIRST_ROWS: ct.Constant[int],
    VOCAB: ct.Constant[int],
):
    r_blk = ct.bid(0)  # in [0, SECOND_ROWS/BLOCK_R)
    c_blk = ct.bid(1)  # in [0, VOCAB/BLOCK_C)
    tile = ct.load(logits_ptr, index=(r_blk, c_blk), shape=(BLOCK_R, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_bf = ct.astype(tile, ct.bfloat16)
    # Mask out OOB cols beyond VOCAB
    cols = ct.arange(BLOCK_C, dtype=ct.int32) + c_blk * BLOCK_C
    rows = ct.arange(BLOCK_R, dtype=ct.int32) + r_blk * BLOCK_R
    col_valid = ct.reshape(cols < VOCAB, (1, BLOCK_C))
    row_valid = ct.reshape(rows < SECOND_ROWS, (BLOCK_R, 1))
    valid = row_valid & col_valid
    zero = ct.full(shape=(BLOCK_R, BLOCK_C), fill_value=0.0, dtype=ct.bfloat16)
    tile_bf = ct.where(valid, tile_bf, zero)
    # Store to padded[FIRST_ROWS + r_blk*BLOCK_R:, c_blk*BLOCK_C:]
    # Use scatter for masked store.
    row_idx = rows + FIRST_ROWS
    row_2d = ct.reshape(row_idx, (BLOCK_R, 1)) + ct.full(shape=(BLOCK_R, BLOCK_C), fill_value=0, dtype=ct.int32)
    col_2d = ct.reshape(cols, (1, BLOCK_C)) + ct.full(shape=(BLOCK_R, BLOCK_C), fill_value=0, dtype=ct.int32)
    ct.scatter(padded_ptr, (row_2d, col_2d), tile_bf, mask=valid)
    # unpadded_t[c_blk*BLOCK_C+cols, FIRST_ROWS + r_blk*BLOCK_R+rows] = tile_bf
    col_2d_u = ct.reshape(cols, (1, BLOCK_C)) + ct.full(shape=(BLOCK_R, BLOCK_C), fill_value=0, dtype=ct.int32)
    row_2d_u = ct.reshape(row_idx, (BLOCK_R, 1)) + ct.full(shape=(BLOCK_R, BLOCK_C), fill_value=0, dtype=ct.int32)
    ct.scatter(unpadded_t_ptr, (col_2d_u, row_2d_u), tile_bf, mask=valid)


@ct.kernel
def _pad_zero_cols_kernel(
    padded_ptr,       # bf16 [ROWS, PADDED_VOCAB]
    ROWS_C: ct.Constant[int],
    PAD_START: ct.Constant[int],
    PAD_LEN: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    r_blk = ct.bid(0)
    # Store zeros at padded[r_blk*BLOCK_R:(r_blk+1)*BLOCK_R, PAD_START:PAD_START+PAD_LEN]
    rows = ct.arange(BLOCK_R, dtype=ct.int32) + r_blk * BLOCK_R
    cols = ct.arange(PAD_LEN, dtype=ct.int32) + PAD_START
    zero = ct.zeros(shape=(BLOCK_R, PAD_LEN), dtype=ct.bfloat16)
    row_valid = ct.reshape(rows < ROWS_C, (BLOCK_R, 1))
    col_valid = ct.reshape(cols < PADDED_VOCAB, (1, PAD_LEN))
    valid = row_valid & col_valid
    row_2d = ct.reshape(rows, (BLOCK_R, 1)) + ct.full(shape=(BLOCK_R, PAD_LEN), fill_value=0, dtype=ct.int32)
    col_2d = ct.reshape(cols, (1, PAD_LEN)) + ct.full(shape=(BLOCK_R, PAD_LEN), fill_value=0, dtype=ct.int32)
    ct.scatter(padded_ptr, (row_2d, col_2d), zero, mask=valid)


@oracle_impl(hardware="B200", point="6d651bcd")
def oracle_forward(inputs):
    embed, logits, _pad = inputs
    padded = torch.empty_strided(
        (ROWS, PADDED_VOCAB),
        (PADDED_VOCAB, 1),
        device=embed.device,
        dtype=torch.bfloat16,
    )
    unpadded_t = torch.empty_strided(
        (VOCAB, ROWS),
        (1, VOCAB),
        device=embed.device,
        dtype=torch.bfloat16,
    )

    # 1) Copy embed transposed into padded[0:FIRST_ROWS, 0:VOCAB] and unpadded_t[:, 0:FIRST_ROWS]
    BLOCK_C = 32
    BLOCK_R = 128  # FIRST_ROWS
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(VOCAB, BLOCK_C), 1, 1),
        _embed_transpose_kernel,
        (embed, padded, unpadded_t, BLOCK_C, BLOCK_R, FIRST_ROWS),
    )
    # 2) Copy logits into padded[FIRST_ROWS:, 0:VOCAB] and unpadded_t[:, FIRST_ROWS:]
    BLOCK_R_LOG = 32
    BLOCK_C_LOG = 128
    ct.launch(
        stream,
        (ct.cdiv(SECOND_ROWS, BLOCK_R_LOG), ct.cdiv(VOCAB, BLOCK_C_LOG), 1),
        _logits_copy_kernel,
        (logits, padded, unpadded_t, BLOCK_R_LOG, BLOCK_C_LOG, FIRST_ROWS, VOCAB),
    )
    # 3) Zero pad the last 6 columns of padded [ROWS x 6]
    BLOCK_R_PAD = 32
    PAD_LEN = 8  # next power of 2 >= 6
    ct.launch(
        stream,
        (ct.cdiv(ROWS, BLOCK_R_PAD), 1, 1),
        _pad_zero_cols_kernel,
        (padded, ROWS, VOCAB, PAD_LEN, BLOCK_R_PAD),
    )
    return padded, unpadded_t
