"""cuTile port of pointwise_247d220fbbab: T5 causal-mask constant fills.

Writes iota[1024] and add[1024] (both int64 arange), and a bool [1,1,1024,1]
tensor of ones. The Repro also returns unsqueeze views + an expand alias.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _fill_iota_add_true_kernel(iota_ptr, add_ptr, true_base_ptr,
                                BLOCK: ct.Constant[int]):
    values = ct.arange(BLOCK, dtype=ct.int64)
    ct.store(iota_ptr, index=(0,), tile=values)
    ct.store(add_ptr, index=(0,), tile=values)
    ones = ct.full(shape=(BLOCK,), fill_value=True, dtype=ct.bool_)
    ct.store(true_base_ptr, index=(0,), tile=ones)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (expand_shape,) = inputs
    device = torch.device("cuda", 0)
    iota = torch.empty_strided((1024,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((1024,), (1,), device=device, dtype=torch.int64)
    ge = torch.empty_strided(
        (1, 1, 1024, 1), (1024, 1024, 1, 1), device=device, dtype=torch.bool
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (1, 1, 1), _fill_iota_add_true_kernel,
        (iota, add, ge.view(1024), BLOCK),
    )
    unsqueeze = add.unsqueeze(0)
    unsqueeze_1 = unsqueeze.unsqueeze(1)
    unsqueeze_2 = unsqueeze_1.unsqueeze(3)
    expand = ge.expand(tuple(int(dim) for dim in expand_shape))
    return iota, add, unsqueeze, unsqueeze_1, unsqueeze_2, ge, expand
