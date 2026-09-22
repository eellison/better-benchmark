"""cuTile port of pointwise_9f7dce7fb407: zero-rank lift_fresh_copy for a scalar.

cuTile doesn't support 0-dimension tiles, so we view the scalar input as a
length-1 tensor and copy it via a size-(1,) load/store.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scalar_fresh_copy_kernel(input_ptr, output_ptr):
    v = ct.load(input_ptr, index=(0,), shape=(1,))
    ct.store(output_ptr, index=(0,), tile=v)


@oracle_impl(hardware="B200", point="2a837a19")
@oracle_impl(hardware="B200", point="896c6bb5")
def oracle_forward(inputs):
    (arg0_1,) = inputs
    output = torch.empty_strided((), (), device=arg0_1.device, dtype=arg0_1.dtype)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _scalar_fresh_copy_kernel,
        (arg0_1.view(1), output.view(1)),
    )
    return output
