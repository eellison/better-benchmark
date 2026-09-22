"""cuTile port of pointwise_18acdd25ec9e (SCHEDULER_FUSION): MobileBERT
embedding + affine.

For each `row` in [0, ROWS): let `seq = row % SEQ_LEN`. Gather the position
embedding `word_table[token_ids[seq]]` and the token-type row-0 constant, add
to the activation, apply scale/bias affine with bf16 rounded intermediate
boundaries. cuTile's default fp32 arithmetic is round-to-nearest-even, so
the Triton inline PTX becomes plain arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 256
SEQ = 128
HIDDEN = 512
ROWS = BATCH * SEQ


@ct.kernel
def _mobilebert_embedding_affine_kernel(
    activation_ptr,       # bf16 [ROWS, HIDDEN]
    token_ids_ptr,        # i64 [SEQ]
    word_table_ptr,       # bf16 [512, HIDDEN]
    token_type_row0_ptr,  # bf16 [HIDDEN] (row 0 of token_type_table)
    scale_ptr,            # bf16 [HIDDEN]
    bias_ptr,             # bf16 [HIDDEN]
    out_ptr,              # bf16 [ROWS, HIDDEN]
    SEQ_LEN: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    seq_idx = row - (row // SEQ_LEN) * SEQ_LEN

    token_id = ct.load(token_ids_ptr, index=(seq_idx,), shape=(1,))
    tid_i32 = ct.astype(token_id, ct.int32)

    # word_table[token_id, :]
    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    tid_bcast = ct.reshape(tid_i32, (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    word_bf = ct.gather(word_table_ptr, (tid_bcast, cols))
    word = ct.astype(word_bf, ct.float32)

    activation = ct.astype(
        ct.load(activation_ptr, index=(row, 0), shape=(1, BLOCK_H)),
        ct.float32,
    )
    token_type = ct.astype(
        ct.load(token_type_row0_ptr, index=(0,), shape=(BLOCK_H,)),
        ct.float32,
    )
    scale = ct.astype(
        ct.load(scale_ptr, index=(0,), shape=(BLOCK_H,)),
        ct.float32,
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)),
        ct.float32,
    )
    word_2d = ct.reshape(word, (1, BLOCK_H))
    token_type_2d = ct.reshape(token_type, (1, BLOCK_H))
    scale_2d = ct.reshape(scale, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))

    add0 = ct.astype(ct.astype(activation + word_2d, ct.bfloat16), ct.float32)
    add1 = ct.astype(ct.astype(add0 + token_type_2d, ct.bfloat16), ct.float32)
    mul = ct.astype(ct.astype(add1 * scale_2d, ct.bfloat16), ct.float32)
    out = ct.astype(mul + bias_2d, ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=out)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="0c4f2b31", BLOCK_H=512)
def oracle_forward(inputs, *, BLOCK_H: int):
    (
        activation,     # bf16 [32768, 512]
        token_ids,      # i64 [1, 512]  (only first 128 used via slice)
        word_table,     # bf16 [512, 512]
        token_type_table,  # bf16 [2, 512]
        scale,          # bf16 [512]
        bias,           # bf16 [512]
        base_shape,     # (256, 128, 512)
        _full_shape,
        view_shape_0,
        view_shape_1,
        view_shape_2,
    ) = inputs

    base_shape = _shape_tuple(base_shape)
    view_shape_0 = _shape_tuple(view_shape_0)
    view_shape_1 = _shape_tuple(view_shape_1)
    view_shape_2 = _shape_tuple(view_shape_2)
    out = torch.empty_strided(
        base_shape,
        _contiguous_stride(base_shape),
        device=activation.device,
        dtype=activation.dtype,
    )
    out_2d = out.view(ROWS, HIDDEN)
    # Slice first SEQ token IDs
    token_ids_1d = token_ids[0, :SEQ].contiguous()
    # Extract row 0 of token_type_table
    token_type_row0 = token_type_table[0]

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _mobilebert_embedding_affine_kernel,
        (activation, token_ids_1d, word_table, token_type_row0,
         scale, bias, out_2d, SEQ, HIDDEN, BLOCK_H),
    )
    return (
        out,
        out.view(view_shape_0),
        out.view(view_shape_1),
        out.view(view_shape_2),
    )
