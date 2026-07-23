"""cuTile port of pointwise_e8ae58ea3c65: bf16 sigmoid backward (grad * sig * (1-sig))."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _sigmoid_backward_kernel(
    grad_ptr,
    sigmoid_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    grad_bf16 = ct.load(grad_ptr, index=(pid,), shape=(BLOCK,))
    sig_bf16 = ct.load(sigmoid_ptr, index=(pid,), shape=(BLOCK,))
    grad = ct.astype(grad_bf16, ct.float32)
    sig = ct.astype(sig_bf16, ct.float32)
    partial = sig * (1.0 - sig)
    out = grad * partial
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out, ct.bfloat16))


@oracle_impl(hardware="B200", point="8ebdd403", BLOCK=64)
def oracle_forward(inputs, *, BLOCK: int):
    grad, sigmoid = inputs
    out = torch.empty_strided(
        tuple(grad.shape),
        tuple(grad.stride()),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    n = grad.numel()
    grad_flat = grad.view(-1)
    sig_flat = sigmoid.view(-1)
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n, BLOCK), 1, 1),
        _sigmoid_backward_kernel,
        (grad_flat, sig_flat, out_flat, BLOCK),
    )
    return out
