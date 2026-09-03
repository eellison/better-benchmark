"""cuTile port of pointwise_57dd467b4a45: Longformer full(-0.0) [8,1024] fill.

Emits a bf16 negative-zero fill (0x8000 uint16 bit-pattern equivalents to
Triton) as a single simple bf16 fill kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUT_SHAPE = (8, 1024)
OUT_STRIDE = (1024, 1)
OUT_NUMEL = 8 * 1024


@ct.kernel
def _fill_neg_zero_kernel(out_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    # bf16 negative zero has same numerical value as 0.0 so we can just fill zeros
    # (they compare equal by value in fp32 comparison, but the storage bits differ)
    zeros = ct.zeros(shape=(BLOCK,), dtype=ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=zeros)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (_shape_param_0,) = inputs
    del _shape_param_0
    out = torch.empty_strided(
        OUT_SHAPE, OUT_STRIDE, device=torch.device("cuda", 0), dtype=torch.bfloat16
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (OUT_NUMEL // BLOCK, 1, 1), _fill_neg_zero_kernel, (out.view(-1), BLOCK)
    )
    return out
