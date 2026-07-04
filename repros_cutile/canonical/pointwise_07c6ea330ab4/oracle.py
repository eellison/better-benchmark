"""cuTile port of pointwise_07c6ea330ab4: Longformer padded-chunk layout.

Materializes bf16[384, 768, 64] output from bf16[8192, 768] input + bf16[768] bias.
For each output entry (chunk, pos, dim):
  head_batch = chunk // 4, window = chunk % 4
  batch = head_batch // 12, head = head_batch % 12
  source_seq = pos + window*256 - 256
  source_feature = head*64 + dim
  if 0 <= source_seq < 1024:
    out = arg0[source_seq * BATCH + batch, source_feature] + bias[source_feature]
  else:
    out = -1.0
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 1024
BATCH = 8
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
WINDOWS = 4
WINDOW_SIZE = 768
WINDOW_STEP = 256
PAD_BEFORE = 256
TOTAL_CHUNKS = BATCH * HEADS * WINDOWS
OUTPUT_SHAPE = (TOTAL_CHUNKS, WINDOW_SIZE, HEAD_DIM)
OUTPUT_STRIDE = (WINDOW_SIZE * HEAD_DIM, HEAD_DIM, 1)


@ct.kernel
def _longformer_chunk_kernel(
    input_flat_ptr,   # bf16 (SEQ * BATCH * HIDDEN,)
    bias_ptr,         # bf16 (HIDDEN,)
    out_flat_ptr,     # bf16 (TOTAL_CHUNKS * WINDOW_SIZE * HEAD_DIM,)
    BLOCK_P: ct.Constant[int],
):
    chunk = ct.bid(0)
    pos_block = ct.bid(1)

    head_batch = chunk // WINDOWS
    window = chunk - head_batch * WINDOWS
    batch = head_batch // HEADS
    head = head_batch - batch * HEADS

    pos = ct.arange(BLOCK_P, dtype=ct.int32) + pos_block * BLOCK_P
    dims = ct.arange(HEAD_DIM, dtype=ct.int32)
    source_seq = pos + window * WINDOW_STEP - PAD_BEFORE
    source_feature = head * HEAD_DIM + dims

    pos_2d = ct.reshape(pos, (BLOCK_P, 1))
    dim_2d = ct.reshape(dims, (1, HEAD_DIM))
    src_seq_2d = ct.reshape(source_seq, (BLOCK_P, 1))
    src_feat_2d = ct.reshape(source_feature, (1, HEAD_DIM))

    valid = (src_seq_2d >= 0) & (src_seq_2d < SEQ)
    safe_seq = ct.where(valid, src_seq_2d, ct.zeros((BLOCK_P, 1), dtype=ct.int32))

    # Ones-broadcast to full (BLOCK_P, HEAD_DIM)
    seq_offsets_2d = safe_seq * BATCH + batch  # (BLOCK_P, 1)
    load_offsets = seq_offsets_2d * HIDDEN + src_feat_2d  # (BLOCK_P, HEAD_DIM)
    values = ct.gather(input_flat_ptr, load_offsets)  # bf16 tile
    bias_1d = ct.load(bias_ptr, index=(head,), shape=(HEAD_DIM,))
    bias_2d = ct.reshape(bias_1d, (1, HEAD_DIM))

    values_f = ct.astype(values, ct.float32)
    bias_f = ct.astype(bias_2d, ct.float32)
    added = ct.astype(values_f + bias_f, ct.bfloat16)
    neg_one = ct.full((BLOCK_P, HEAD_DIM), -1.0, dtype=ct.bfloat16)
    out = ct.where(valid, added, neg_one)

    # store to out[chunk, pos, dim]
    store_offsets = (chunk * WINDOW_SIZE * HEAD_DIM
                     + pos_2d * HEAD_DIM + dim_2d)
    ct.scatter(out_flat_ptr, store_offsets, out)


@oracle_impl(hardware="B200", point="5fa3702b", BLOCK_P=32)
def oracle_forward(inputs, *, BLOCK_P: int):
    arg0_1, arg1_1, *_shape_params = inputs
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    input_flat = arg0_1.view(-1)
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (TOTAL_CHUNKS, WINDOW_SIZE // BLOCK_P, 1),
        _longformer_chunk_kernel,
        (input_flat, arg1_1, out_flat, BLOCK_P),
    )
    return out
