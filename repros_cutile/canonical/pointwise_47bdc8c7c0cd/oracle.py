"""cuTile port of pointwise_47bdc8c7c0cd: bf16 tanh-approximate GELU.

Ports the Triton `_bf16_tanh_gelu_kernel` to cuda.tile. Applies the
tanh-approximate GELU with the bf16 half factor round-trip and returns
the bf16 output plus the transpose alias view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_tanh_gelu_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf16 = ct.load(input_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x = ct.astype(x_bf16, ct.float32)
    half = ct.astype(ct.astype(x * 0.5, ct.bfloat16), ct.float32)
    x2 = x * x
    x3 = x2 * x
    tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
    # tanh(z) = (exp(2z) - 1) / (exp(2z) + 1); use tanh via math identity
    # cuTile provides ct.tanh via math intrinsics? use ct.tanh if available.
    y = half * (ct.tanh(tanh_arg) + 1.0)
    ct.store(output_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


def _resolve_shape(shape, numel):
    out = [int(dim) for dim in shape]
    infer = -1
    known = 1
    for index, dim in enumerate(out):
        if dim == -1:
            infer = index
        else:
            known *= dim
    if infer >= 0:
        out[infer] = numel // known
    return tuple(out)


@oracle_impl(hardware="B200", point="92efc45e", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="d05618d1", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x, _shape0, shape1 = inputs
    out_shape = _resolve_shape(shape1, x.numel())
    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    # x is bf16[M, N] contiguous; flatten to 1D for tile-based loads
    x_flat = x.view(-1)
    out_flat = output.view(-1)
    ct.launch(
        stream,
        (ct.cdiv(x_flat.numel(), BLOCK_SIZE), 1, 1),
        _bf16_tanh_gelu_kernel,
        (x_flat, out_flat, BLOCK_SIZE),
    )
    return output, output.permute(1, 0)
