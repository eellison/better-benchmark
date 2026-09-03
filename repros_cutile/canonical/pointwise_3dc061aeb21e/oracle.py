"""cuTile port of pointwise_3dc061aeb21e: bool causal mask base fill (all True)."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _fill_true_mask_kernel(
    base_ptr,
    BLOCK: ct.Constant[int],
):
    # Fill a [1, 1, 1024, 1] bool tensor with True (viewed as [1024])
    one = ct.full(shape=(BLOCK,), fill_value=1, dtype=ct.bool_)
    ct.store(base_ptr, index=(0,), tile=one)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (expand_shape,) = inputs
    base = torch.empty_strided(
        (1, 1, 1024, 1),
        (1024, 1024, 1, 1),
        device=torch.device("cuda", 0),
        dtype=torch.bool,
    )
    base_flat = base.view(1024)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (1, 1, 1), _fill_true_mask_kernel, (base_flat, BLOCK))

    return base.expand(tuple(int(dim) for dim in expand_shape))
