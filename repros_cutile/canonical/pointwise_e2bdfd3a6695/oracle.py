"""cuTile port of pointwise_e2bdfd3a6695: Gemma 5x causal-mask fanout."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


NUM_OUTPUTS = 5
BATCH = 1
HEADS = 8
Q_LEN = 1000
K_LEN = 1000
BASE_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
BASE_STRIDE = (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1)
EXPANDED_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
BASE_NUMEL = BATCH * Q_LEN * K_LEN  # 1_000_000


@ct.kernel
def _causal_mask_5_kernel(
    out0, out1, out2, out3, out4,
    K_LEN_C: ct.Constant[int],
    Q_LEN_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    key = offsets % K_LEN_C
    query = (offsets // K_LEN_C) % Q_LEN_C
    neg_inf = ct.full(shape=(BLOCK,), fill_value=-float("inf"), dtype=ct.float32)
    zero = ct.zeros(shape=(BLOCK,), dtype=ct.float32)
    values_f = ct.where(key <= query, zero, neg_inf)
    values = ct.astype(values_f, ct.bfloat16)
    ct.store(out0, index=(pid,), tile=values)
    ct.store(out1, index=(pid,), tile=values)
    ct.store(out2, index=(pid,), tile=values)
    ct.store(out3, index=(pid,), tile=values)
    ct.store(out4, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1000)
def oracle_forward(inputs, *, BLOCK: int):
    del inputs
    device = torch.device("cuda", torch.cuda.current_device())
    bases = tuple(
        torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device=device, dtype=torch.bfloat16)
        for _ in range(NUM_OUTPUTS)
    )
    # BASE_NUMEL = 1_000_000. Need BLOCK to divide 1_000_000 and be pow-of-2 for cuTile.
    # 1_000_000 = 2^6 * 5^6 * ... hmm 1M = 10^6. pow-of-2 divisors are 1,2,4,8,16,32,64.
    # Use BLOCK=64, grid = 15625.
    del BLOCK
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (BASE_NUMEL // 64, 1, 1),
        _causal_mask_5_kernel,
        (bases[0].view(-1), bases[1].view(-1), bases[2].view(-1),
         bases[3].view(-1), bases[4].view(-1), K_LEN, Q_LEN, 64),
    )
    return tuple(base.expand(EXPANDED_SHAPE) for base in bases)
