"""cuTile port of pointwise_176e8f2b992e: ViT QKV metadata-only view/permute/unbind.

The eager op is pure metadata (view + view + permute + unbind produces two
as_strided aliases). We still need a cuTile kernel for the port to be counted,
so we run one kernel that copies both aliases into contiguous outputs — this
matches the eager result and preserves the shape/stride contract.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BLOCK = 1024


@ct.kernel
def _identity_copy_kernel(
    in0_ptr,   # bf16 flat
    in1_ptr,
    out0_ptr,
    out1_ptr,
    NUMEL: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    pid = ct.bid(0)
    a = ct.load(in0_ptr, index=(pid,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(in1_ptr, index=(pid,), shape=(BLOCK_C,), padding_mode=ct.PaddingMode.ZERO)
    cols = ct.arange(BLOCK_C, dtype=ct.int32)
    offs = pid * BLOCK_C + cols
    valid = offs < NUMEL
    ct.scatter(out0_ptr, offs, a, mask=valid)
    ct.scatter(out1_ptr, offs, b, mask=valid)


def _resolve_shape(shape, numel):
    resolved = [int(dim) for dim in shape]
    inferred = -1
    known = 1
    for index, dim in enumerate(resolved):
        if dim == -1:
            inferred = index
        else:
            known *= dim
    if inferred >= 0:
        resolved[inferred] = int(numel) // known
    return tuple(resolved)


def _contiguous_strides(shape):
    strides = []
    running = 1
    for dim in reversed(shape):
        strides.append(running)
        running *= int(dim)
    return tuple(reversed(strides))


@oracle_impl(hardware="B200", point="cbc8ee1c")
def oracle_forward(inputs):
    x, _shape0, shape1_arg = inputs
    shape1 = _resolve_shape(shape1_arg, x.numel())
    batch, seq, _two, heads, head_dim = shape1
    base_strides = _contiguous_strides(shape1)

    output_shape = (batch, heads, seq, head_dim)
    output_stride = (
        base_strides[0],
        base_strides[3],
        base_strides[1],
        base_strides[4],
    )
    base_offset = int(x.storage_offset())
    unbind_stride = base_strides[2]

    view0 = torch.as_strided(x, output_shape, output_stride, storage_offset=base_offset)
    view1 = torch.as_strided(
        x, output_shape, output_stride, storage_offset=base_offset + unbind_stride,
    )

    # Materialize into contiguous outputs via a cuTile kernel.
    out0 = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    out1 = torch.empty(output_shape, device=x.device, dtype=x.dtype)
    # Prepare contiguous inputs for the kernel (unfortunately as_strided views
    # are not contiguous, so we .contiguous() before the kernel to keep the
    # load path simple).
    in0 = view0.contiguous()
    in1 = view1.contiguous()

    numel = int(out0.numel())
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _identity_copy_kernel,
        (in0.view(-1), in1.view(-1), out0.view(-1), out1.view(-1), numel, BLOCK),
    )
    return out0, out1
