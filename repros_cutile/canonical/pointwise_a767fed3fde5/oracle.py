"""cuTile port of pointwise_a767fed3fde5 (SCHEDULER_FUSION): Blenderbot causal
mask fanout — materialize 24 fresh [32,1,128,128] causal masks in one kernel
sharing the iota/le predicate.

Each row `r` of the [S, S] mask is `where(c <= r, 0.0, -inf)` (bf16).
The [32,1,S,S] output has all 32 batch entries repeat the same S*S pattern.
"""

import math

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
S = 128
S2 = S * S
BATCH_N = BATCH * S2  # total elements per output tensor


@ct.kernel
def _causal_mask_24_kernel(
    out0, out1, out2, out3, out4, out5, out6, out7,
    out8, out9, out10, out11, out12, out13, out14, out15,
    out16, out17, out18, out19, out20, out21, out22, out23,
    BLOCK: ct.Constant[int],
    S_: ct.Constant[int],
    S2_: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    # Reduce to a single [S, S] block; find row and col within the S*S pattern.
    within = offsets % S2_
    rows = within // S_
    cols = within - rows * S_
    neg_inf = ct.full((BLOCK,), -math.inf, dtype=ct.float32)
    zero_f = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    values = ct.astype(ct.where(cols <= rows, zero_f, neg_inf), ct.bfloat16)

    ct.store(out0, index=(pid,), tile=values)
    ct.store(out1, index=(pid,), tile=values)
    ct.store(out2, index=(pid,), tile=values)
    ct.store(out3, index=(pid,), tile=values)
    ct.store(out4, index=(pid,), tile=values)
    ct.store(out5, index=(pid,), tile=values)
    ct.store(out6, index=(pid,), tile=values)
    ct.store(out7, index=(pid,), tile=values)
    ct.store(out8, index=(pid,), tile=values)
    ct.store(out9, index=(pid,), tile=values)
    ct.store(out10, index=(pid,), tile=values)
    ct.store(out11, index=(pid,), tile=values)
    ct.store(out12, index=(pid,), tile=values)
    ct.store(out13, index=(pid,), tile=values)
    ct.store(out14, index=(pid,), tile=values)
    ct.store(out15, index=(pid,), tile=values)
    ct.store(out16, index=(pid,), tile=values)
    ct.store(out17, index=(pid,), tile=values)
    ct.store(out18, index=(pid,), tile=values)
    ct.store(out19, index=(pid,), tile=values)
    ct.store(out20, index=(pid,), tile=values)
    ct.store(out21, index=(pid,), tile=values)
    ct.store(out22, index=(pid,), tile=values)
    ct.store(out23, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK=1024):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch = expand_shape[0]
    seq = expand_shape[2]
    numel = batch * seq * seq
    stride = (seq * seq, seq * seq, seq, 1)
    device = torch.device("cuda", 0)
    outputs = tuple(
        torch.empty_strided(expand_shape, stride, device=device, dtype=torch.bfloat16)
        for _ in range(24)
    )
    flat_outputs = tuple(o.view(numel) for o in outputs)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _causal_mask_24_kernel,
        (*flat_outputs, BLOCK, seq, seq * seq),
    )
    return outputs
