"""cuTile port of pointwise_3c92a46da990: XLNet-style layout clone with bf16 cast.

Input f32[1024,16,64] -> output bf16[1024,1024] laid out contiguously as
[c=64, b=16, t=1024] flat (a shuffle transpose+cast).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


T = 1024
B = 16
C = 64


@ct.kernel
def _cast_clone_kernel(
    src,   # f32 (T, B, C)
    dst,   # bf16 (C, B, T)
    TBLOCK: ct.Constant[int],
    CBLOCK: ct.Constant[int],
):
    tb = ct.bid(0)  # T tile index
    cb = ct.bid(1)  # C tile index
    b = ct.bid(2)   # B index

    # Load (TBLOCK, 1, CBLOCK) from src at (tb, b, cb)
    values = ct.load(src, index=(tb, b, cb), shape=(TBLOCK, 1, CBLOCK))
    values = ct.reshape(values, (TBLOCK, CBLOCK))
    # Transpose to (CBLOCK, TBLOCK) and cast to bf16
    values_t = ct.transpose(values)
    values_bf16 = ct.astype(values_t, ct.bfloat16)
    # Store to dst at (cb, b, tb) with shape (CBLOCK, 1, TBLOCK)
    out_tile = ct.reshape(values_bf16, (CBLOCK, 1, TBLOCK))
    ct.store(dst, index=(cb, b, tb), tile=out_tile)


@oracle_impl(hardware="B200", point="d102a86e", TBLOCK=128, CBLOCK=64)
def oracle_forward(inputs, *, TBLOCK=128, CBLOCK=64):
    arg0_1, _shape_param = inputs
    # Output (1024, 1024) contiguous
    base = torch.empty_strided(
        (T, T),
        (T, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    # base viewed as (C, B, T) contiguous
    base_3d = base.view(C, B, T)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (T // TBLOCK, C // CBLOCK, B),
        _cast_clone_kernel,
        (arg0_1, base_3d, TBLOCK, CBLOCK),
    )
    return base, base.permute(1, 0)
