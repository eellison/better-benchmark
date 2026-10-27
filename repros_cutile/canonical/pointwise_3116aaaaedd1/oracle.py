"""cuTile port of pointwise_3116aaaaedd1: MT5 gated tanh-GELU backward pointwise + aliases."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _mt5_gated_tanh_gelu_backward_kernel(
    grad_ptr,
    mask_ptr,
    x_ptr,
    gate_ptr,
    out0_ptr,
    out1_ptr,
    BLOCK_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)

    grad = ct.astype(ct.load(grad_ptr, index=(pid,), shape=(BLOCK_SIZE,)), ct.float32)
    keep = ct.astype(ct.load(mask_ptr, index=(pid,), shape=(BLOCK_SIZE,)), ct.float32)
    x = ct.astype(ct.load(x_ptr, index=(pid,), shape=(BLOCK_SIZE,)), ct.float32)
    gate = ct.astype(ct.load(gate_ptr, index=(pid,), shape=(BLOCK_SIZE,)), ct.float32)

    scaled_grad = grad * (keep * 1.1111111111111112)
    half = ct.astype(ct.astype(x * 0.5, ct.bfloat16), ct.float32)

    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    tanh_val = ct.tanh(tanh_arg)
    tanh_plus_one = tanh_val + 1.0

    gelu = half * tanh_plus_one
    ct.store(out0_ptr, index=(pid,), tile=ct.astype(scaled_grad * gelu, ct.bfloat16))

    gated_grad = scaled_grad * gate
    left = gated_grad * half
    right = gated_grad * tanh_plus_one
    right_half = ct.astype(ct.astype(ct.astype(right, ct.bfloat16), ct.float32) * 0.5, ct.bfloat16)

    tanh_grad = 1.0 - tanh_val * tanh_val
    d_tanh = ((left * tanh_grad) * 0.7978845608028654)
    d_tanh_bf16 = ct.astype(d_tanh, ct.bfloat16)
    d_cubic = ct.astype(((d_tanh * 0.044715) * (x2 * 3.0)), ct.bfloat16)

    derivative_tail = ct.astype(
        ct.astype(d_tanh_bf16, ct.float32) + ct.astype(d_cubic, ct.float32),
        ct.bfloat16,
    )
    out1 = ct.astype(
        ct.astype(derivative_tail, ct.float32) + ct.astype(right_half, ct.float32),
        ct.bfloat16,
    )
    ct.store(out1_ptr, index=(pid,), tile=out1)


@oracle_impl(hardware="B200", point="f749e533", BLOCK_SIZE=512)
def oracle_forward(inputs, *, BLOCK_SIZE):
    arg0, arg1, arg2, arg3, _shape0, _shape1, _shape2, out0_shape, out1_shape = inputs
    out0_shape = tuple(int(dim) for dim in out0_shape)
    out1_shape = tuple(int(dim) for dim in out1_shape)
    out0 = torch.empty_strided(
        out0_shape,
        (out0_shape[1], 1),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    out1 = torch.empty_strided(
        out1_shape,
        (out1_shape[1], 1),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    numel = arg0.numel()
    # Flatten inputs to 1D for cuTile.
    grad_flat = arg0.view(numel)
    mask_flat = arg1.view(numel)
    x_flat = arg2.view(numel)
    gate_flat = arg3.view(numel)
    out0_flat = out0.view(numel)
    out1_flat = out1.view(numel)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK_SIZE), 1, 1),
        _mt5_gated_tanh_gelu_backward_kernel,
        (grad_flat, mask_flat, x_flat, gate_flat, out0_flat, out1_flat, BLOCK_SIZE),
    )
    return out0, out0.permute(1, 0), out1, out1.permute(1, 0)
