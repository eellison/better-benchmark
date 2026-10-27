"""cuTile port of pointwise_febb9b4c9d22: 12 identical causal-mask outputs.

Emits a [64, 1, 128, 128] bf16 tensor where entry (b, 0, r, c) is 0.0 if
c <= r else -inf, then broadcasts to 12 stores.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
SEQ = 128


@ct.kernel
def _causal_mask_kernel(
    out0, out1, out2, out3, out4, out5,
    out6, out7, out8, out9, out10, out11,
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    b = ct.bid(0)
    r_tile = ct.bid(1)
    c_tile = ct.bid(2)

    # Row indices and col indices for this tile.
    r_idx = r_tile * BLOCK_R + ct.arange(BLOCK_R, dtype=ct.int32)
    c_idx = c_tile * BLOCK_C + ct.arange(BLOCK_C, dtype=ct.int32)
    # Compare c_idx[None, :] <= r_idx[:, None]
    r_2d = ct.reshape(r_idx, (BLOCK_R, 1))
    c_2d = ct.reshape(c_idx, (1, BLOCK_C))
    keep = c_2d <= r_2d
    zero_v = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.float32)
    ninf_v = ct.full((BLOCK_R, BLOCK_C), -float("inf"), dtype=ct.float32)
    values = ct.astype(ct.where(keep, zero_v, ninf_v), ct.bfloat16)
    # reshape to (1, 1, BLOCK_R, BLOCK_C) for storing into 4D dst
    values4 = ct.reshape(values, (1, 1, BLOCK_R, BLOCK_C))
    ct.store(out0, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out1, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out2, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out3, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out4, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out5, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out6, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out7, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out8, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out9, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out10, index=(b, 0, r_tile, c_tile), tile=values4)
    ct.store(out11, index=(b, 0, r_tile, c_tile), tile=values4)


@oracle_impl(hardware="B200", point="d7517139", BLOCK_R=32, BLOCK_C=32)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C):
    expand_shape = tuple(1 if int(dim) == -1 else int(dim) for dim in inputs[0])
    batch = expand_shape[0]
    seq = expand_shape[2]
    stride = (seq * seq, seq * seq, seq, 1)
    device = torch.device("cuda", 0)

    outputs = tuple(
        torch.empty_strided(expand_shape, stride, device=device, dtype=torch.bfloat16)
        for _ in range(12)
    )
    stream = torch.cuda.current_stream()
    grid = (batch, seq // BLOCK_R, seq // BLOCK_C)
    ct.launch(stream, grid, _causal_mask_kernel, (*outputs, BLOCK_R, BLOCK_C))
    return outputs
