"""cuTile port of pointwise_562d52916cc9 (ALGEBRAIC_ELIMINATION): materialize
tiny int64 iota, add=iota, and ge=(iota>=0) base, then return metadata views."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _iota_add_ge_kernel(
    iota_ptr,
    add_ptr,
    ge_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    values = ct.astype(ct.arange(BLOCK, dtype=ct.int32), ct.int64)
    ct.store(iota_ptr, index=(pid,), tile=values)
    ct.store(add_ptr, index=(pid,), tile=values)
    ct.store(ge_ptr, index=(pid,), tile=values >= 0)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=512)
def oracle_forward(inputs, *, BLOCK):
    (expand_shape,) = inputs
    device = torch.device("cuda", 0)

    iota = torch.empty_strided((512,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((512,), (1,), device=device, dtype=torch.int64)
    ge = torch.empty_strided(
        (1, 1, 512, 1),
        (512, 512, 1, 1),
        device=device,
        dtype=torch.bool,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _iota_add_ge_kernel,
        (iota, add, ge.view(-1), BLOCK),
    )

    unsqueeze = torch.as_strided(add, (1, 512), (512, 1))
    unsqueeze_1 = torch.as_strided(add, (1, 1, 512), (512, 512, 1))
    unsqueeze_2 = torch.as_strided(add, (1, 1, 512, 1), (512, 512, 1, 1))
    expand = ge.expand(tuple(int(dim) for dim in expand_shape))
    return iota, add, unsqueeze, unsqueeze_1, unsqueeze_2, ge, expand
