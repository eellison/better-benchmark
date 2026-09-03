"""cuTile port of pointwise_ef7896396d78 (SCHEDULER_FUSION):

Bart/MBart bf16 causal mask fanout — materialize 12 identical `[8,1,1024,1024]`
tensors where value is 0.0 if col<=row else -inf.
"""

import math

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


NEG_INF = float("-inf")


@ct.kernel
def _causal_mask_12_kernel(
    out0, out1, out2, out3, out4, out5,
    out6, out7, out8, out9, out10, out11,
    S: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    batch = ct.bid(0)
    row = ct.bid(1)
    # cols in [0, BLOCK)
    cols = ct.arange(BLOCK, dtype=ct.int32)
    # Row scalar broadcast
    row_i32 = ct.astype(ct.full(shape=(BLOCK,), fill_value=row, dtype=ct.int32), ct.int32)
    keep = cols <= row_i32
    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    ninf = ct.full(shape=(BLOCK,), fill_value=NEG_INF, dtype=ct.bfloat16)
    values = ct.where(keep, zero, ninf)
    # Store to all 12 outputs at (batch, 0, row, 0)
    ct.store(out0, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out1, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out2, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out3, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out4, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out5, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out6, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out7, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out8, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out9, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out10, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))
    ct.store(out11, index=(batch, 0, row, 0), tile=ct.reshape(values, (1, 1, 1, BLOCK)))


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch, heads, seq, _ = expand_shape
    stride = (heads * seq * seq, seq * seq, seq, 1)
    device = torch.device("cuda", 0)
    outputs = tuple(
        torch.empty_strided(expand_shape, stride, device=device, dtype=torch.bfloat16)
        for _ in range(12)
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, seq, 1),
        _causal_mask_12_kernel,
        (*outputs, seq, BLOCK),
    )
    return outputs
