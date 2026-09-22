"""cuTile port of pointwise_bef60966ace3: bool mask -> bf16 (0 or -inf) sentinel."""

import math
import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _mask_to_bias_kernel(
    mask_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    mask = ct.load(mask_ptr, index=(pid,), shape=(BLOCK,))
    zeros = ct.zeros(shape=(BLOCK,), dtype=ct.bfloat16)
    neg_inf = ct.full(shape=(BLOCK,), fill_value=-math.inf, dtype=ct.bfloat16)
    out = ct.where(mask, zeros, neg_inf)
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="605f6819", BLOCK=1024)
@oracle_impl(hardware="B200", point="6fdaf67b", BLOCK=1024)
@oracle_impl(hardware="B200", point="08fb6d2d", BLOCK=1024)
@oracle_impl(hardware="B200", point="9ba6de11", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    (mask,) = inputs
    out = torch.empty_strided(
        tuple(mask.shape),
        tuple(mask.stride()),
        device=mask.device,
        dtype=torch.bfloat16,
    )
    numel = mask.numel()
    # Flatten to 1D
    mask_flat = mask.reshape(numel)
    out_flat = out.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _mask_to_bias_kernel,
        (mask_flat, out_flat, BLOCK),
    )
    return out
