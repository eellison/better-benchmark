"""cuTile port of pointwise_2d969ac9ad59 (NEW_PATTERN): tanh-approximate GELU
on flat input, dtype-preserving."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _tanh_gelu_kernel(
    x_ptr,
    out_ptr,
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.load(x_ptr, index=(pid,), shape=(BLOCK,))
    xf = ct.astype(x, ct.float32)
    x2 = xf * xf
    x3 = x2 * xf
    arg = (xf + x3 * 0.044715) * 0.7978845608028654
    tanh_val = ct.tanh(arg)
    y = (xf * 0.5) * (tanh_val + 1.0)
    ct.store(out_ptr, index=(pid,), tile=ct.astype(y, x.dtype))


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


@oracle_impl(hardware="B200", point="cd47785e", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="92efc45e", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="d05618d1", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="7b14189f", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="5cd82a46", BLOCK_SIZE=1024)
@oracle_impl(hardware="B200", point="8f687505", BLOCK_SIZE=1024)
def oracle_forward(inputs, *, BLOCK_SIZE):
    x, _shape0, shape1 = inputs
    out_shape = _resolve_shape(shape1, x.numel())
    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=x.dtype,
    )
    numel = x.numel()
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK_SIZE), 1, 1),
        _tanh_gelu_kernel,
        (x.view(-1), output.view(-1), BLOCK_SIZE),
    )
    return output
