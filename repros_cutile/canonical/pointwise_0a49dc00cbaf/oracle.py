"""cuTile port of the pointwise_0a49dc00cbaf oracle: allocate a scalar bf16
tensor and store 0.0 with a single one-thread kernel.

cuTile does not currently support 0-dimension tile shapes, so we view the
scalar output as a length-1 bf16 array and store a size-(1,) zero tile."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scalar_bf16_zero_kernel(out_ptr):
    ct.store(out_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.bfloat16))


@oracle_impl(hardware="B200", point="d7517139")
def oracle_forward(inputs):
    out = torch.empty((), device=torch.device("cuda", 0), dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _scalar_bf16_zero_kernel, (out.view(1),))
    return out
