"""cuTile port of pointwise_848f47987aea: all-true bool[1,1,128,1] base, expanded."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _all_true_128_kernel(out_ptr, BLOCK: ct.Constant[int]):
    ones = ct.full(shape=(BLOCK,), fill_value=True, dtype=ct.bool_)
    ct.store(out_ptr, index=(0,), tile=ones)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=128)
def oracle_forward(inputs, *, BLOCK):
    (expand_shape,) = inputs
    ge = torch.empty_strided(
        (1, 1, 128, 1),
        (128, 128, 1, 1),
        device=torch.device("cuda", 0),
        dtype=torch.bool,
    )
    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _all_true_128_kernel, (ge.view(128), BLOCK))
    return ge.expand(tuple(int(dim) for dim in expand_shape))
