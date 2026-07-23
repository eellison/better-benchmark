"""cuTile port of pointwise_c7832ce0ffe9: Longformer bias+padded chunk layout.

For each output element in [chunk, pos, dim]:
- source_seq = pos + window*256 - 256; valid if 0..1023
- source_feature = head*64 + dim
- load activation[source_seq, batch, source_feature], add bias[source_feature]
- if invalid, store -1.0

The output is [384, 768, 64] and grid is (384, 768/BLOCK_P, 1). Uses ct.gather
for the (possibly invalid) source_seq loads.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 1024
BATCH = 8
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
HEAD_BATCH = BATCH * HEADS
WINDOWS = 4
WINDOW_SIZE = 768
WINDOW_STEP = 256
PAD_BEFORE = 256
TOTAL_CHUNKS = HEAD_BATCH * WINDOWS
OUTPUT_SHAPE = (TOTAL_CHUNKS, WINDOW_SIZE, HEAD_DIM)


@ct.kernel
def _longformer_kernel(
    activation_ptr,  # bf16 flattened [SEQ * BATCH * HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    out_ptr,         # bf16 [TOTAL_CHUNKS, WINDOW_SIZE, HEAD_DIM]
    BLOCK_P: ct.Constant[int],
    BLOCK_D: ct.Constant[int],
    HEADS_C: ct.Constant[int],
    HEAD_DIM_C: ct.Constant[int],
    BATCH_C: ct.Constant[int],
    HIDDEN_C: ct.Constant[int],
    WINDOWS_C: ct.Constant[int],
    WINDOW_STEP_C: ct.Constant[int],
    PAD_BEFORE_C: ct.Constant[int],
    SEQ_C: ct.Constant[int],
):
    chunk = ct.bid(0)
    pos_block = ct.bid(1)
    # BLOCK_D = HEAD_DIM = 64; grid dim 2 is 1
    # Compute (chunk, pos, dim) coordinates
    pos_offset = ct.arange(BLOCK_P, dtype=ct.int32)
    dim_offset = ct.arange(BLOCK_D, dtype=ct.int32)
    pos = pos_block * BLOCK_P + pos_offset  # (BLOCK_P,)

    head_batch = chunk // WINDOWS_C
    window = chunk - head_batch * WINDOWS_C
    batch = head_batch // HEADS_C
    head = head_batch - batch * HEADS_C

    source_seq_1d = pos + window * WINDOW_STEP_C - PAD_BEFORE_C  # (BLOCK_P,)
    source_feature_1d = head * HEAD_DIM_C + dim_offset            # (BLOCK_D,)

    # Reshape for 2D broadcasting
    source_seq = ct.reshape(source_seq_1d, (BLOCK_P, 1))          # (BLOCK_P, 1)
    source_feature = ct.reshape(source_feature_1d, (1, BLOCK_D))  # (1, BLOCK_D)

    valid_seq = (source_seq >= 0) & (source_seq < SEQ_C)
    valid_broadcast = ct.reshape(valid_seq, (BLOCK_P, 1))

    # Compute linear index into flat activation buffer.
    safe_source_seq = ct.where(valid_seq, source_seq, ct.full((BLOCK_P, 1), 0, dtype=ct.int32))
    # Address = source_seq * (BATCH * HIDDEN) + batch * HIDDEN + source_feature
    lin_index = (
        safe_source_seq * (BATCH_C * HIDDEN_C)
        + batch * HIDDEN_C
        + source_feature
    )
    values = ct.astype(
        ct.gather(activation_ptr, lin_index, mask=valid_broadcast, padding_value=0),
        ct.float32,
    )

    # Gather bias directly from source_feature_1d
    bias_gathered_1d = ct.gather(bias_ptr, source_feature_1d)  # (BLOCK_D,)
    bias_gathered = ct.reshape(bias_gathered_1d, (1, BLOCK_D))
    # cast bias to bf16 and back to f32 (mimics fp_downcast_rounding='rtne' + upcast)
    bias_bf16 = ct.astype(ct.astype(bias_gathered, ct.bfloat16), ct.float32)

    out = ct.where(valid_broadcast, values + bias_bf16, -1.0)
    ct.store(out_ptr, index=(chunk, pos_block, 0), tile=ct.astype(ct.reshape(out, (1, BLOCK_P, BLOCK_D)), ct.bfloat16))


@oracle_impl(hardware="B200", point="53c69788", BLOCK_P=32, BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_P: int, BLOCK_D: int):
    bias, activation, *_shape_params = inputs

    out = torch.empty_strided(
        OUTPUT_SHAPE,
        (OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2], OUTPUT_SHAPE[2], 1),
        device=activation.device,
        dtype=torch.bfloat16,
    )

    # Flatten activation as [SEQ * BATCH * HIDDEN] for 1D gather
    activation_flat = activation.contiguous().view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (TOTAL_CHUNKS, WINDOW_SIZE // BLOCK_P, 1),
        _longformer_kernel,
        (
            activation_flat,
            bias,
            out,
            BLOCK_P,
            BLOCK_D,
            HEADS,
            HEAD_DIM,
            BATCH,
            HIDDEN,
            WINDOWS,
            WINDOW_STEP,
            PAD_BEFORE,
            SEQ,
        ),
    )
    return out, out.permute(0, 2, 1)
