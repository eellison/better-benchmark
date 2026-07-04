"""cuTile port of pointwise_d29ee9cb59f4: Gemma sliding-window causal-mask fanout.

Materializes 13 bf16 `[1,1,1000,1000]` mask tensors (0 for kept, -inf for
dropped) and returns each expanded to `[1,8,1000,1000]`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 1000
BASE_SHAPE = (1, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
OUT_SHAPE = (1, 8, SEQ, SEQ)
N_ELEMENTS = SEQ * SEQ


@ct.kernel
def _causal_mask13_kernel(
    o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12,
    BLOCK: ct.Constant[int],
    SEQ_C: ct.Constant[int],
):
    pid = ct.bid(0)
    # Compute per-element key/query in the tile
    base = pid * BLOCK
    offsets = base + ct.arange(BLOCK, dtype=ct.int32)
    key = offsets % SEQ_C
    query = offsets // SEQ_C
    keep = (key > (query - 4096)) & (key <= query)
    # bf16 doesn't have inf constant directly — use float("-inf") in ct.where
    neg_inf = ct.full(shape=(BLOCK,), fill_value=-float("inf"), dtype=ct.float32)
    zero = ct.zeros(shape=(BLOCK,), dtype=ct.float32)
    values_f32 = ct.where(keep, zero, neg_inf)
    values = ct.astype(values_f32, ct.bfloat16)

    valid = offsets < 1000000
    # cuTile lacks masked stores — but our tile size divides the padded buffer
    # if we launch cdiv(N, BLOCK) with padding. Simpler: rely on BLOCK dividing.
    # N_ELEMENTS = 1_000_000; use BLOCK=1024, then 977*1024 = 1000448 > 1000000.
    # Store all writes but ensure we allocate enough space (we do — 1000*1000 = 1000000).
    # Actually, we need exact BLOCK dividing. Use BLOCK=1000000/1000 = ... hmm.
    # 1000000 = 2^6 * 5^6 = 64 * 15625. Only pow2 factor is 64. Use BLOCK=64
    # since 1000000/64 = 15625.
    ct.store(o0, index=(pid,), tile=values)
    ct.store(o1, index=(pid,), tile=values)
    ct.store(o2, index=(pid,), tile=values)
    ct.store(o3, index=(pid,), tile=values)
    ct.store(o4, index=(pid,), tile=values)
    ct.store(o5, index=(pid,), tile=values)
    ct.store(o6, index=(pid,), tile=values)
    ct.store(o7, index=(pid,), tile=values)
    ct.store(o8, index=(pid,), tile=values)
    ct.store(o9, index=(pid,), tile=values)
    ct.store(o10, index=(pid,), tile=values)
    ct.store(o11, index=(pid,), tile=values)
    ct.store(o12, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    bases = tuple(
        torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device="cuda", dtype=torch.bfloat16)
        for _ in range(13)
    )
    # 1000000 = 64 * 15625; use BLOCK=64 which divides N_ELEMENTS.
    BLOCK = 64
    grid_x = N_ELEMENTS // BLOCK
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (grid_x, 1, 1),
        _causal_mask13_kernel,
        (bases[0].view(N_ELEMENTS), bases[1].view(N_ELEMENTS), bases[2].view(N_ELEMENTS),
         bases[3].view(N_ELEMENTS), bases[4].view(N_ELEMENTS), bases[5].view(N_ELEMENTS),
         bases[6].view(N_ELEMENTS), bases[7].view(N_ELEMENTS), bases[8].view(N_ELEMENTS),
         bases[9].view(N_ELEMENTS), bases[10].view(N_ELEMENTS), bases[11].view(N_ELEMENTS),
         bases[12].view(N_ELEMENTS), BLOCK, SEQ),
    )
    return tuple(base.expand(OUT_SHAPE) for base in bases)
