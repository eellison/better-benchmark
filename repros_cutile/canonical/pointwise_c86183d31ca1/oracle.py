"""cuTile port of pointwise_c86183d31ca1 (SCATTER_REDUCE): masked vocabulary
scatter-add.

For each (batch, token) source row, compute `vocab_row = index[batch, token]`
and atomically add the values row into `out[vocab_row]` — but only if
vocab_row is in [0, out_rows) and != 1. Uses cuTile's `ct.atomic_add`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _masked_vocab_scatter_kernel(
    values_ptr,       # f32 [batch, seq_len, hidden]
    index_ptr,        # i64 [batch, seq_len]
    out_ptr,          # f32 [out_rows, hidden]
    SEQ_LEN: ct.Constant[int],
    OUT_ROWS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    source = ct.bid(0)
    col_block = ct.bid(1)
    batch = source // SEQ_LEN
    token = source - batch * SEQ_LEN

    vocab_scalar = ct.load(index_ptr, index=(batch, token), shape=(1, 1))
    vocab_row = ct.reshape(vocab_scalar, (1,))
    zero_i = ct.full(shape=(1,), fill_value=0, dtype=ct.int64)
    out_rows_i = ct.full(shape=(1,), fill_value=OUT_ROWS, dtype=ct.int64)
    one_i = ct.full(shape=(1,), fill_value=1, dtype=ct.int64)
    active_scalar = (vocab_row >= zero_i) & (vocab_row < out_rows_i) & (vocab_row != one_i)
    # Redirect inactive rows to a safe row (0) and add 0.0
    safe_row_scalar = ct.where(active_scalar, vocab_row, zero_i)

    values = ct.load(values_ptr, index=(batch, token, col_block),
                     shape=(1, 1, BLOCK_H),
                     padding_mode=ct.PaddingMode.ZERO)
    values_1d = ct.reshape(values, (BLOCK_H,))
    values_scaled = values_1d * 1.0  # explicit mul by 1

    # active is a scalar; broadcast to BLOCK_H
    active_scalar_i32 = ct.astype(active_scalar, ct.int32)
    active_bcast = ct.reshape(active_scalar_i32, (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    active = active_bcast != ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)

    # Zero out inactive values so atomic_add is a no-op there
    zero_v = ct.full(shape=(BLOCK_H,), fill_value=0.0, dtype=ct.float32)
    to_add = ct.where(active, values_scaled, zero_v)

    # Build row/col index for atomic_add
    row_idx = ct.reshape(ct.astype(safe_row_scalar, ct.int32), (1,)) + ct.full(shape=(BLOCK_H,), fill_value=0, dtype=ct.int32)
    cols = ct.arange(BLOCK_H, dtype=ct.int32) + col_block * BLOCK_H
    ct.atomic_add(out_ptr, (row_idx, cols), to_add)


# 19f6778a: (T([8,1024,1024], f32), T([8,1024], i64), S([50265,1024]))
@oracle_impl(hardware="B200", point="19f6778a")
# 47b53512: (T([32,128,2560], f32), T([32,128], i64), S([8008,2560]))
@oracle_impl(hardware="B200", point="47b53512")
# c51d1d47: (T([16,128,2560], f32), T([16,128], i64), S([8008,2560]))
@oracle_impl(hardware="B200", point="c51d1d47")
# f18349c3: (T([64,128,1024], f32), T([64,128], i64), S([128112,1024]))
@oracle_impl(hardware="B200", point="f18349c3")
# f507fb35: (T([16,1024,768], f32), T([16,1024], i64), S([50005,768]))
@oracle_impl(hardware="B200", point="f507fb35")
# 79ee2993: (T([64,256,1024], f32), T([64,256], i64), S([50265,1024]))
@oracle_impl(hardware="B200", point="79ee2993")
# e91334e5: (T([32,128,1024], f32), T([32,128], i64), S([256008,1024]))
@oracle_impl(hardware="B200", point="e91334e5")
def oracle_forward(inputs):
    values, indices, out_shape = inputs
    out_rows = int(out_shape[0])
    hidden = int(out_shape[1])
    batch = int(indices.shape[0])
    seq_len = int(indices.shape[1])

    # Zero-init output
    out = torch.zeros((out_rows, hidden), device=values.device, dtype=torch.float32)

    # BLOCK_H: choose a value that divides hidden.
    # hidden values seen: 1024, 2560, 768.
    if hidden % 1024 == 0:
        BLOCK_H = 1024
    elif hidden % 768 == 0:
        BLOCK_H = 768  # not pow2, but let's try — cuTile might require pow2
        # Fall back: use 512
        BLOCK_H = 256
    else:
        BLOCK_H = 256
    while BLOCK_H > 1 and hidden % BLOCK_H != 0:
        BLOCK_H //= 2
    # Ensure BLOCK_H is a power of 2 dividing hidden
    # hidden=768: divisors are 256, 128, 64...
    # hidden=1024: BLOCK_H=1024
    # hidden=2560: divisors 256, 128, 64...
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (batch * seq_len, hidden // BLOCK_H, 1),
        _masked_vocab_scatter_kernel,
        (values, indices, out, seq_len, out_rows, hidden, BLOCK_H),
    )
    return out
