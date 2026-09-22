"""cuTile port of pointwise_fc28348c6da6: XLNet sinusoidal positional-embedding
fanout of 24 identical bf16[16384, 1024] outputs.

The Triton kernel writes the same value to 24 output pointers. We compute the
table once with cuTile, then copy to the other 23 outputs via torch. Note that
`cat([sin, cos], -1)` operates in the hidden axis; expand+clone [1024, 16, 1024]
is equivalent to broadcasting rows across the 16-wide axis.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 1024
HALF_HIDDEN = 512
BLOCK = 1024


@ct.kernel
def _xlnet_pos_emb_kernel(
    out_ptr,  # bf16 [ROWS, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    HALF_H_C: ct.Constant[int],
    ROWS_PER_POS_C: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    # One tile per (row, half). We split by hidden-half so we can compute
    # sin for the low half and cos for the high half.
    row = ct.bid(0)
    half = ct.bid(1)  # 0 -> sin (cols 0..511), 1 -> cos (cols 512..1023)

    pos_index = row // ROWS_PER_POS_C
    pos = ct.astype(512 - pos_index, ct.float32)

    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    freq_index = ct.astype(cols, ct.float32)
    even_dim = freq_index * 2.0
    exponent = even_dim / 1024.0
    # 10000 ** exponent
    denom = ct.exp(exponent * ct.log(ct.full((BLOCK_C,), 10000.0, dtype=ct.float32)))
    inv_freq = 1.0 / denom
    phase = pos * inv_freq

    # branch: use sin for half=0, cos for half=1
    # Since half is a bid scalar, both branches evaluate lazily.
    sin_v = ct.sin(phase)
    cos_v = ct.cos(phase)
    value = ct.where(half == 0, sin_v, cos_v)
    value_bf16 = ct.astype(ct.reshape(value, (1, BLOCK_C)), ct.bfloat16)
    ct.store(out_ptr, index=(row, half), tile=value_bf16)


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    del inputs
    device = torch.device("cuda", 0)
    ROWS_PER_POS = 16  # each pos_index generates 16 rows

    outputs = tuple(
        torch.empty_strided((ROWS, HIDDEN), (HIDDEN, 1), device=device, dtype=torch.bfloat16)
        for _ in range(24)
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 2, 1),
        _xlnet_pos_emb_kernel,
        (outputs[0], HIDDEN, HALF_HIDDEN, ROWS_PER_POS, HALF_HIDDEN),
    )
    for i in range(1, 24):
        outputs[i].copy_(outputs[0])
    return outputs
