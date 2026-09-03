"""cuTile port of pointwise_89c74f3242b5: bool mask -> bf16 (0, -inf) bias base + expand alias."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 16
HEADS = 12
SEQ = 1024
BASE_SHAPE = (BATCH, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
EXPAND_SHAPE = (BATCH, HEADS, SEQ, SEQ)
N_ELEMENTS = BATCH * SEQ * SEQ


@ct.kernel
def _mask_to_bias_kernel(mask_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    m = ct.load(mask_ptr, index=(pid,), shape=(BLOCK,))
    keep = m != 0
    zero = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.float32)
    neg_inf = ct.full(shape=(BLOCK,), fill_value=float("-inf"), dtype=ct.float32)
    values = ct.where(keep, zero, neg_inf)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(values, ct.bfloat16))


@oracle_impl(hardware="B200", point="4d6c86f7", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    mask, _shape0 = inputs
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mask.device,
        dtype=torch.bfloat16,
    )
    mask_flat = mask.reshape(N_ELEMENTS)
    base_flat = base.view(N_ELEMENTS)
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(N_ELEMENTS, BLOCK), 1, 1)
    ct.launch(
        stream,
        grid,
        _mask_to_bias_kernel,
        (mask_flat, base_flat, BLOCK),
    )
    return base, base.expand(EXPAND_SHAPE)
