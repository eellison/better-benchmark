"""cuTile port of pointwise_e40852a34050: iota/add/ge base + metadata expand views."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _iota_add_ge_kernel(iota_ptr, add_ptr, ge_ptr, BLOCK: ct.Constant[int]):
    values = ct.astype(ct.arange(BLOCK, dtype=ct.int32), ct.int64)
    ct.store(iota_ptr, index=(0,), tile=values)
    ct.store(add_ptr, index=(0,), tile=values)
    zero = ct.zeros(shape=(BLOCK,), dtype=ct.int64)
    ct.store(ge_ptr, index=(0,), tile=values >= zero)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=128)
def oracle_forward(inputs, *, BLOCK):
    (expand_shape,) = inputs
    device = torch.device("cuda", 0)

    iota = torch.empty_strided((128,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((128,), (1,), device=device, dtype=torch.int64)
    ge = torch.empty_strided(
        (1, 1, 128, 1),
        (128, 128, 1, 1),
        device=device,
        dtype=torch.bool,
    )
    ge_flat = torch.as_strided(ge, (128,), (1,))

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _iota_add_ge_kernel, (iota, add, ge_flat, BLOCK))

    unsqueeze = torch.as_strided(add, (1, 128), (128, 1))
    unsqueeze_1 = torch.as_strided(add, (1, 1, 128), (128, 128, 1))
    unsqueeze_2 = torch.as_strided(add, (1, 1, 128, 1), (128, 128, 1, 1))
    expand = ge.expand(tuple(int(dim) for dim in expand_shape))
    return iota, add, unsqueeze, unsqueeze_1, unsqueeze_2, ge, expand
