"""cuTile port of pointwise_b6d762cd3d81: DeepRecommender SELU pointwise.

NEW_PATTERN: `bf16(where(x_f32>0, x*1.05070..., expm1(x)*1.75810...))` in one
pass, preserving the fp32 compute and bf16 output rounding.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _selu_bf16_kernel(
    x_ptr,       # bf16 [N]
    out_ptr,     # bf16 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf16 = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x = ct.astype(x_bf16, ct.float32)
    positive = x * 1.0507009873554805
    negative = (ct.exp(x) - 1.0) * 1.7580993408473766
    zero = ct.zeros((BLOCK,), dtype=ct.float32)
    y = ct.where(x > zero, positive, negative)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


@oracle_impl(hardware="B200", point="58f85f13", BLOCK=256)
@oracle_impl(hardware="B200", point="f1684e51", BLOCK=256)
@oracle_impl(hardware="B200", point="0f3e2fa1", BLOCK=256)
def oracle_forward(inputs, *, BLOCK: int):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = x.reshape(n_elements)
    out_flat = out.view(n_elements)
    grid = ((n_elements + BLOCK - 1) // BLOCK, 1, 1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _selu_bf16_kernel,
        (x_flat, out_flat, BLOCK),
    )
    return out
