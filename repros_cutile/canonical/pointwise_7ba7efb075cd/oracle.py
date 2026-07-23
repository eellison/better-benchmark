"""cuTile port of pointwise_7ba7efb075cd: Whisper singleton mask - 4 zero scalars expanded.

Allocates four `[1,1,1,1]` bf16 tensors filled with zero and returns each
expanded to `[1,6,1,1]`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BASE_SHAPE = (1, 1, 1, 1)
BASE_STRIDE = (8, 8, 8, 1)
OUT_SHAPE = (1, 6, 1, 1)


@ct.kernel
def _zero_four_scalars_kernel(out0, out1, out2, out3):
    zero = ct.zeros(shape=(1,), dtype=ct.bfloat16)
    ct.store(out0, index=(0,), tile=zero)
    ct.store(out1, index=(0,), tile=zero)
    ct.store(out2, index=(0,), tile=zero)
    ct.store(out3, index=(0,), tile=zero)


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    device = torch.device("cuda")
    out0_base = torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device=device, dtype=torch.bfloat16)
    out1_base = torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device=device, dtype=torch.bfloat16)
    out2_base = torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device=device, dtype=torch.bfloat16)
    out3_base = torch.empty_strided(BASE_SHAPE, BASE_STRIDE, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _zero_four_scalars_kernel,
        (out0_base.view(1), out1_base.view(1), out2_base.view(1), out3_base.view(1)),
    )
    return (
        out0_base.expand(OUT_SHAPE),
        out1_base.expand(OUT_SHAPE),
        out2_base.expand(OUT_SHAPE),
        out3_base.expand(OUT_SHAPE),
    )
