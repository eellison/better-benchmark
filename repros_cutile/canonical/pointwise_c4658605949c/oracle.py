"""cuTile port of pointwise_c4658605949c: M2M100 causal source-mask materialize.

Ports the Triton `_causal_source_mask_kernel`. For each (batch, q, k):
  out[batch, 0, q, k] = (k <= q) & (arg0[batch, k] != 0.0)

We use a full-row 128-wide load of the source and a 128x128 tile per batch.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
SEQ = 128
OUT_SHAPE = (BATCH, 1, SEQ, SEQ)
OUT_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)


@ct.kernel
def _causal_source_mask_kernel(
    in_arr,   # (BATCH, SEQ) f32
    out_arr,  # (BATCH, 1, SEQ, SEQ) bool
    SEQ_: ct.Constant[int],
):
    batch = ct.bid(0)
    # Load source row [SEQ] and broadcast against q index range.
    source = ct.load(in_arr, index=(batch, 0), shape=(1, SEQ_))
    # source has shape (1, SEQ). Non-zero float => True for valid key.
    src_ne = ct.not_equal(source, ct.full(shape=(1, SEQ_), fill_value=0.0, dtype=ct.float32))

    # Build (SEQ, SEQ) mask where mask[q, k] = k <= q.
    q_idx = ct.arange(SEQ_, dtype=ct.int32)
    k_idx = ct.arange(SEQ_, dtype=ct.int32)
    q_col = ct.reshape(q_idx, (SEQ_, 1))  # (SEQ, 1)
    k_row = ct.reshape(k_idx, (1, SEQ_))  # (1, SEQ)
    keep = ct.less_equal(k_row, q_col)  # (SEQ, SEQ) bool

    # Broadcast src_ne (1, SEQ) with keep (SEQ, SEQ).
    values = ct.bitwise_and(keep, src_ne)
    # Store to (batch, 0, :, :) — full SEQ x SEQ tile at the (batch, 0) start.
    ct.store(out_arr, index=(batch, 0, 0, 0), tile=ct.reshape(values, (1, 1, SEQ_, SEQ_)))


# 3faaaa71: (T([64,128], f32), S([64,-1,128,128]))
@oracle_impl(hardware="B200", point="3faaaa71")
def oracle_forward(inputs):
    arg0_1, expand_shape = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg0_1.device,
        dtype=torch.bool,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BATCH, 1, 1),
        _causal_source_mask_kernel,
        (arg0_1, out, SEQ),
    )
    return out.expand(tuple(int(dim) for dim in expand_shape))
