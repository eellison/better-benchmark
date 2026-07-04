"""cuTile port of pointwise_c96952c07750 (BANDWIDTH_BOUND): Blenderbot bool
attention mask bias — returns (neg_inf, zero, where) where the [16,1,128,128]
where tensor is `bf16 = mask ? 0.0 : -inf`.
"""

import math

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


OUT_SHAPE = (16, 1, 128, 128)
OUT_STRIDE = (16384, 16384, 128, 1)
N_ELEMENTS = 16 * 128 * 128


@ct.kernel
def _attention_mask_bias_kernel(
    mask_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    mask_values = ct.load(mask_ptr, index=(pid,), shape=(BLOCK,))
    neg_inf = ct.full((BLOCK,), -math.inf, dtype=ct.float32)
    zero_f = ct.full((BLOCK,), 0.0, dtype=ct.float32)
    values = ct.astype(ct.where(mask_values, zero_f, neg_inf), ct.bfloat16)
    ct.store(out_ptr, index=(pid,), tile=values)


@oracle_impl(hardware="B200", point="61a8ad35", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK=1024):
    (mask,) = inputs
    neg_inf_scalar = torch.full(
        (), float("-inf"), device=mask.device, dtype=torch.bfloat16
    )
    zero_scalar = torch.full((), 0.0, device=mask.device, dtype=torch.bfloat16)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=mask.device,
        dtype=torch.bfloat16,
    )
    mask_flat = mask.reshape(N_ELEMENTS)
    output_flat = output.view(N_ELEMENTS)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ELEMENTS, BLOCK), 1, 1),
        _attention_mask_bias_kernel,
        (mask_flat, output_flat, BLOCK),
    )
    return neg_inf_scalar, zero_scalar, output
