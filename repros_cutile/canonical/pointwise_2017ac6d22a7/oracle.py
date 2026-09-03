"""cuTile port of pointwise_2017ac6d22a7: MobileViT slice_scatter layout.

For each output element (batch, head, seq, dim) in a [512, 4, 256, 40] output:
  if dim < 36: load from src [131072,144] using the folded map
                (batch, seq, head, dim) -> row = batch*256+seq, col = head*36+dim
  else:        load from base [512,4,256,40] directly at the same position.

Since 512*4*256*40 = 20971520 = 1024 * 20480, BLOCK=1024 divides exactly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
HEADS = 4
SEQ = 256
ACTIVE_DIM = 36
PADDED_DIM = 40
HIDDEN = HEADS * ACTIVE_DIM  # 144
TOTAL = BATCH * HEADS * SEQ * PADDED_DIM  # 20971520


@ct.kernel
def _mobilevit_slice_scatter_kernel(
    src_ptr,      # bf16 flat [131072 * 144]
    base_ptr,     # bf16 flat [BATCH * HEADS * SEQ * PADDED_DIM]
    out_ptr,      # bf16 flat [BATCH * HEADS * SEQ * PADDED_DIM]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lane = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)

    # Decode (batch, head, seq, dim) from flat lane
    dim = lane % PADDED_DIM
    tmp = lane // PADDED_DIM
    seq = tmp % SEQ
    tmp = tmp // SEQ
    head = tmp % HEADS
    batch = tmp // HEADS

    from_src = dim < ACTIVE_DIM
    zero64 = ct.zeros(shape=(BLOCK,), dtype=ct.int64)
    # src_off = (batch * SEQ + seq) * HIDDEN + head * ACTIVE_DIM + dim
    safe_dim = ct.where(from_src, dim, zero64)
    src_off = (batch * SEQ + seq) * HIDDEN + head * ACTIVE_DIM + safe_dim
    # base_off = flat index in [BATCH, HEADS, SEQ, PADDED_DIM] which equals lane
    src_values = ct.gather(src_ptr, src_off)
    base_values = ct.gather(base_ptr, lane)
    values = ct.where(from_src, src_values, base_values)
    ct.store(out_ptr, index=(pid,), tile=values)


# (T([131072,144], bf16), T([512,4,256,40], bf16), S([512,256,144]), S([512,256,4,36]))
@oracle_impl(hardware="B200", point="259ee10a", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    src, base, _shape0, _shape1 = inputs
    out = torch.empty_strided(
        (BATCH, HEADS, SEQ, PADDED_DIM),
        (HEADS * SEQ * PADDED_DIM, SEQ * PADDED_DIM, PADDED_DIM, 1),
        device=base.device,
        dtype=base.dtype,
    )
    src_flat = src.contiguous().view(-1)
    base_flat = base.contiguous().view(-1)
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _mobilevit_slice_scatter_kernel,
        (src_flat, base_flat, out_flat, BLOCK),
    )
    return out
