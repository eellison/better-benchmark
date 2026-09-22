"""cuTile port of any_1918882eece2: gt-zero mask + scalar any.

Loads a flat tensor of 8192 elements, computes gt(0), stores the mask, and
reduces to a single scalar `any`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _gt_zero_mask_any_kernel(
    x_ptr,
    mask_ptr,
    any_ptr,
    BLOCK: ct.Constant[int],
):
    values = ct.load(x_ptr, index=(0,), shape=(BLOCK,))
    zero = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    values_f = ct.astype(values, ct.float32)
    keep = values_f > zero
    ct.store(mask_ptr, index=(0,), tile=keep)
    keep_int = ct.astype(keep, ct.int32)
    total = ct.sum(keep_int)
    total_1d = ct.reshape(total, (1,))
    zero_1 = ct.full((1,), 0, dtype=ct.int32)
    any_val = total_1d != zero_1
    ct.store(any_ptr, index=(0,), tile=any_val)


@oracle_impl(hardware="B200", point="a7e138d5", BLOCK=8192)
@oracle_impl(hardware="B200", point="5002b631", BLOCK=8192)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, _shape_param_0 = inputs
    gt = torch.empty_strided(
        arg0_1.shape,
        arg0_1.stride(),
        device=arg0_1.device,
        dtype=torch.bool,
    )
    any_1 = torch.empty((), device=arg0_1.device, dtype=torch.bool)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (1, 1, 1),
        _gt_zero_mask_any_kernel,
        (arg0_1.view(BLOCK), gt.view(BLOCK), any_1.view(1), BLOCK),
    )
    return gt, any_1
