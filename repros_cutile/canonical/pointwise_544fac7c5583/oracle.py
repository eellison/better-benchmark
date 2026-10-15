"""cuTile port of pointwise_544fac7c5583: VisFormer exact-GELU derivative pointwise.

cuTile has no `erf` primitive, so we precompute erf(x * sqrt(0.5)) with torch
(matching the Triton oracle's fp32 semantics) and pass it into a cuTile
elementwise kernel that does the rest of the derivative and the bf16 cast.

Inputs are channels-last strided; we operate on flattened contiguous views
(permute -> contiguous -> flatten) and rebuild the output with the same
channels-last strides.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _gelu_derivative_finish_kernel(
    grad_ptr,   # bf16 [N] (permuted-contiguous flattened)
    x_ptr,      # bf16 [N]
    erf_ptr,    # f32  [N]  (precomputed erf(x * 0.7071...))
    out_ptr,    # bf16 [N]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    grad = ct.load(grad_ptr, index=(pid,), shape=(BLOCK,))
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    erf_val = ct.load(erf_ptr, index=(pid,), shape=(BLOCK,))

    x_f = ct.astype(x, ct.float32)
    grad_f = ct.astype(grad, ct.float32)

    cdf = (erf_val + 1.0) * 0.5
    pdf = ct.exp(x_f * x_f * -0.5) * 0.3989422804014327
    derivative = cdf + x_f * pdf
    out = grad_f * derivative
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="f961de61", BLOCK=512)
@oracle_impl(hardware="B200", point="21c46fcd", BLOCK=512)
@oracle_impl(hardware="B200", point="cfcbfec7", BLOCK=512)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1 = inputs
    # Channels-last: permute to BHWC contiguous.
    grad_p = arg0_1.permute(0, 2, 3, 1).contiguous()
    x_p = arg1_1.permute(0, 2, 3, 1).contiguous()
    # Precompute erf(x * 0.7071...) in fp32.
    erf_val = torch.erf(x_p.to(torch.float32) * 0.7071067811865476)

    n_flat = grad_p.numel()
    grad_flat = grad_p.view(-1)
    x_flat = x_p.view(-1)
    erf_flat = erf_val.view(-1)

    # All target shapes here yield element counts divisible by BLOCK=512.
    out_flat = torch.empty(n_flat, device=arg0_1.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_flat, BLOCK), 1, 1),
        _gelu_derivative_finish_kernel,
        (grad_flat, x_flat, erf_flat, out_flat, BLOCK),
    )

    b, c, h, w = (int(d) for d in arg0_1.shape)
    out_bhwc = out_flat.view(b, h, w, c)
    out = out_bhwc.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
    return out
