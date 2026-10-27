"""cuTile port of pointwise_07f3b9d27881: T5 bf16 bool-mask scale/where pointwise.

Ports the Triton `_masked_scale_where_kernel` to cuTile: for each element,
`out = 0 if zero_mask else x * scale_mask * SCALE_BF16`. Also allocates a bf16
scalar zero and returns view/transpose aliases.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE_BF16 = 1.109375


@ct.kernel
def _masked_scale_where_kernel(
    x_ptr,          # bf16 [N]
    scale_mask_ptr, # b8 [N]
    zero_mask_ptr,  # b8 [N]
    out_ptr,        # bf16 [N]
    SCALE: ct.Constant[float],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    scale_mask = ct.load(scale_mask_ptr, index=(pid,), shape=(BLOCK,))
    zero_mask = ct.load(zero_mask_ptr, index=(pid,), shape=(BLOCK,))

    x_f = ct.astype(x, ct.float32)
    scale_mask_f = ct.astype(scale_mask, ct.float32)
    scaled = x_f * (scale_mask_f * SCALE)
    zero_mask_b = zero_mask != ct.full(shape=(BLOCK,), fill_value=0, dtype=ct.uint8)
    out = ct.where(zero_mask_b, 0.0, scaled)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="363c89c3", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    base_shape = tuple(int(dim) for dim in _shape_param_0)
    flat_shape = tuple(int(dim) for dim in _shape_param_1)
    base = torch.empty_strided(
        base_shape,
        (base_shape[1] * base_shape[2], base_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    full = torch.zeros((), device=arg0_1.device, dtype=torch.bfloat16)

    n = arg0_1.numel()
    x_flat = arg0_1.reshape(-1)
    m1_flat = arg1_1.reshape(-1)
    m2_flat = arg2_1.reshape(-1)
    base_flat = base.reshape(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK_SIZE), 1, 1),
        _masked_scale_where_kernel,
        (x_flat, m1_flat, m2_flat, base_flat, SCALE_BF16, BLOCK_SIZE),
    )
    flat = base.view(flat_shape)
    return full, flat, flat.permute(1, 0)
