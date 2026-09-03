"""cuTile port of pointwise_562635abba38: bf16 tanh-GELU backward pointwise."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _albert_tanh_gelu_backward_kernel(
    grad_ptr,
    x_ptr,
    out_ptr,
    N_ELEMENTS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    grad_bf = ct.load(grad_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x_bf = ct.load(x_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    grad = ct.astype(grad_bf, ct.float32)
    x = ct.astype(x_bf, ct.float32)

    # half_x = (x * 0.5).to(tl.bfloat16).to(tl.float32)
    half_x = ct.astype(ct.astype(x_bf, ct.float32) * 0.5, ct.bfloat16)
    half_x_f = ct.astype(half_x, ct.float32)
    left = grad * half_x_f

    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    # emulate libdevice.tanh via exp/exp
    exp_pos = ct.exp(tanh_arg)
    exp_neg = ct.exp(-tanh_arg)
    tanh_val = (exp_pos - exp_neg) / (exp_pos + exp_neg)
    tanh_plus_one = tanh_val + 1.0

    right = grad * tanh_plus_one
    # right_half = (right.to(bf16).to(f32) * 0.5).to(bf16)
    right_bf = ct.astype(right, ct.bfloat16)
    right_half_f = ct.astype(right_bf, ct.float32) * 0.5
    right_half = ct.astype(right_half_f, ct.bfloat16)

    tanh_grad = 1.0 - tanh_val * tanh_val
    d_tanh = (left * tanh_grad) * 0.7978845608028654
    d_tanh_bf = ct.astype(d_tanh, ct.bfloat16)
    d_cubic = ct.astype((d_tanh * 0.044715) * (x2 * 3.0), ct.bfloat16)

    derivative_tail = ct.astype(
        ct.astype(d_tanh_bf, ct.float32) + ct.astype(d_cubic, ct.float32),
        ct.bfloat16,
    )
    out = ct.astype(
        ct.astype(derivative_tail, ct.float32) + ct.astype(right_half, ct.float32),
        ct.bfloat16,
    )
    ct.store(out_ptr, index=(pid,), tile=out)


@oracle_impl(hardware="B200", point="271eb370", BLOCK_SIZE=512)
def oracle_forward(inputs, *, BLOCK_SIZE: int):
    grad, x, _shape0, _shape1, shape2 = inputs
    out_shape = tuple(int(dim) for dim in shape2)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    numel = grad.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK_SIZE), 1, 1),
        _albert_tanh_gelu_backward_kernel,
        (grad.view(numel), x.view(numel), out.view(numel), numel, BLOCK_SIZE),
    )
    return out, out.permute(1, 0)
