"""cuTile port of pointwise_c1af8096914d: M2M100 bool mask -> bf16 attention bias.

Given b8 mask [64,1,128,128], write:
- neg_inf scalar bf16
- zero scalar bf16
- bf16 base [64,1,128,128] = where(mask, 0, -inf)
- expanded view [64,16,128,128] pointing to base with zero-stride on head axis
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 64
HEADS = 16
SEQ = 128
BASE_SHAPE = (BATCH, 1, SEQ, SEQ)
BASE_STRIDE = (SEQ * SEQ, SEQ * SEQ, SEQ, 1)
EXPAND_SHAPE = (BATCH, HEADS, SEQ, SEQ)
N_ELEMENTS = BATCH * SEQ * SEQ


@ct.kernel
def _mask_to_bias_kernel(
    mask_ptr,       # b8 [N_ELEMENTS] (flat)
    out_ptr,        # bf16 [N_ELEMENTS] (flat)
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    mask = ct.load(mask_ptr, index=(pid,), shape=(BLOCK,))
    # Mask is bool (u8 in torch); non-zero means keep
    keep = ct.astype(mask, ct.int32) != 0
    zero_bf = ct.full(shape=(BLOCK,), fill_value=0.0, dtype=ct.bfloat16)
    neginf_bf = ct.full(shape=(BLOCK,), fill_value=float("-inf"), dtype=ct.bfloat16)
    values = ct.where(keep, zero_bf, neginf_bf)
    ct.store(out_ptr, index=(pid,), tile=values)


@ct.kernel
def _write_scalar_kernel(neg_inf_ptr, zero_ptr):
    ct.store(neg_inf_ptr, index=(0,), tile=ct.full(shape=(1,), fill_value=float("-inf"), dtype=ct.bfloat16))
    ct.store(zero_ptr, index=(0,), tile=ct.full(shape=(1,), fill_value=0.0, dtype=ct.bfloat16))


@oracle_impl(hardware="B200", point="6c0a9ee3", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    mask, _shape0 = inputs
    neg_inf = torch.empty((), device=mask.device, dtype=torch.bfloat16)
    zero = torch.empty((), device=mask.device, dtype=torch.bfloat16)
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mask.device,
        dtype=torch.bfloat16,
    )
    mask_flat = mask.contiguous().view(-1)
    base_flat = base.view(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ELEMENTS, BLOCK), 1, 1),
        _mask_to_bias_kernel,
        (mask_flat, base_flat, BLOCK),
    )
    ct.launch(
        stream,
        (1, 1, 1),
        _write_scalar_kernel,
        (neg_inf.view(1), zero.view(1)),
    )
    return neg_inf, zero, base, base.expand(EXPAND_SHAPE)
