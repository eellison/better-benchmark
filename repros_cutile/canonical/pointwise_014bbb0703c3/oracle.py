"""cuTile port of pointwise_014bbb0703c3: bf16 exact-erf GELU.

Computes `0.5 * x * (erf(x * 1/sqrt(2)) + 1)` fully in-kernel using an
Abramowitz-Stegun 7.1.26 polynomial approximation of erf (cuTile has no
built-in erf). Matches Triton kernel's in-kernel `libdevice.erf`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SQRT_HALF = 0.7071067811865476


def _erf_approx(x):
    """Abramowitz-Stegun 7.1.26 polynomial approximation for erf."""
    a1 = 0.278393
    a2 = 0.230389
    a3 = 0.000972
    a4 = 0.078108

    x_abs = ct.abs(x)
    x2 = x_abs * x_abs
    x3 = x2 * x_abs
    x4 = x3 * x_abs
    denom = 1.0 + a1 * x_abs + a2 * x2 + a3 * x3 + a4 * x4
    denom2 = denom * denom
    denom4 = denom2 * denom2
    erf_abs = 1.0 - 1.0 / denom4
    sign = ct.where(x >= 0.0, 1.0, -1.0)
    return sign * erf_abs


@ct.kernel
def _gelu_kernel(
    x_ptr,       # bf16 [N]
    out_ptr,     # bf16 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    x_f = ct.astype(x, ct.float32)
    erf_arg = x_f * SQRT_HALF
    erf_val = _erf_approx(erf_arg)
    y = x_f * 0.5 * (erf_val + 1.0)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


# ff8c3966: (T([128,3072,7,7], bf16, stride=(150528,1,21504,3072)))
@oracle_impl(hardware="B200", point="ff8c3966", BLOCK=1024)
# d3381898: (T([128,1536,14,14], bf16, stride=(301056,1,21504,1536)))
@oracle_impl(hardware="B200", point="d3381898", BLOCK=1024)
# e421f3e7: (T([128,384,28,28], bf16, stride=(301056,1,10752,384)))
@oracle_impl(hardware="B200", point="e421f3e7", BLOCK=1024)
# af87e8bd: (T([1,384,3000], bf16))
@oracle_impl(hardware="B200", point="af87e8bd", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (x,) = inputs
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    # Triton treats the (strided but dense) tensor as flat storage. Mirror
    # that with as_strided views — metadata-only, no copy.
    x_flat = torch.as_strided(x, (n_elements,), (1,))
    out_flat = torch.as_strided(out, (n_elements,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _gelu_kernel,
        (x_flat, out_flat, BLOCK),
    )
    return out
