"""cuTile port of pointwise_49492eb7035e: bf16 tanh-approximate GELU."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_fp32_tanh_gelu_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: ct.Constant[int],
):
    pid = ct.bid(0)
    x_bf = ct.load(input_ptr, index=(pid,), shape=(BLOCK_SIZE,))
    x = ct.astype(x_bf, ct.float32)
    half = ct.astype(ct.astype(x * 0.5, ct.bfloat16), ct.float32)
    x_cubed = x * x * x
    tanh_arg = (x + x_cubed * 0.044715) * 0.7978845608028654
    # tanh via ct: use math library available - use tanh through built-in.
    # cuTile may expose ct.tanh; if not, use identity: tanh(x) = 2 * sigmoid(2x) - 1.
    y = half * (ct.tanh(tanh_arg) + 1.0)
    ct.store(output_ptr, index=(pid,), tile=ct.astype(y, ct.bfloat16))


def _resolve_shape(shape, numel):
    out = [int(dim) for dim in shape]
    infer_index = -1
    known = 1
    for index, dim in enumerate(out):
        if dim == -1:
            infer_index = index
        else:
            known *= dim
    if infer_index >= 0:
        out[infer_index] = numel // known
    return tuple(out)


@oracle_impl(hardware="B200", point="7b14189f", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="cd47785e", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="5cd82a46", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x, _shape0, shape1 = inputs
    out_shape = _resolve_shape(shape1, x.numel())
    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    n_elements = x.numel()
    x_flat = x.reshape(n_elements)
    out_flat = output.reshape(n_elements)
    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(n_elements, BLOCK_SIZE), 1, 1)
    ct.launch(
        stream,
        grid,
        _bf16_fp32_tanh_gelu_kernel,
        (x_flat, out_flat, BLOCK_SIZE),
    )
    return output
