"""cuTile port of pointwise_c1772948ebc8 (ALGEBRAIC_ELIMINATION): iota + identity
embedding gather emitting fresh contiguous f32 [128, 2560] copy plus iota output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 2560
TOTAL = ROWS * HIDDEN


@ct.kernel
def _copy_kernel(x_ptr, out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    values = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    ct.store(out_ptr, index=(pid,), tile=values)


@ct.kernel
def _iota_kernel(iota_ptr, BLOCK: ct.Constant[int]):
    # Just a single tile of length BLOCK==ROWS==128
    rng = ct.arange(BLOCK, dtype=ct.int64)
    ct.store(iota_ptr, index=(0,), tile=rng)


@oracle_impl(hardware="B200", point="800c9b3b", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (arg0_1,) = inputs
    iota = torch.empty((ROWS,), device=arg0_1.device, dtype=torch.int64)
    embedding = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    x_flat = arg0_1.contiguous().view(-1)
    emb_flat = embedding.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(TOTAL, BLOCK), 1, 1),
        _copy_kernel,
        (x_flat, emb_flat, BLOCK),
    )
    # iota: fill with [0..ROWS-1]
    ct.launch(stream, (1, 1, 1), _iota_kernel, (iota, ROWS))
    return iota, embedding
