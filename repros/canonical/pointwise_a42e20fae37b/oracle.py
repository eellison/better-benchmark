"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 QKV view/view/permute/unbind scope as three direct `as_strided` aliases into the original projection storage, whereas Inductor still routes the metadata-only chain through a compiled callable instead of reducing it to explicit alias construction; Inductor cannot do this today because output planning does not canonicalize multi-output QKV view/unbind layout algebra into storage-offset alias returns before benchmarking; the fix is ALGEBRAIC_ELIMINATION: collapse these metadata-only chains to direct `as_strided` outputs and treat the resulting empty device-work graph as the floor for this repro."""

import torch
import triton  # noqa: F401

from oracle_harness import oracle_impl


def _resolve_shape(shape, numel):
    out = [int(dim) for dim in shape]
    known = 1
    infer_index = -1
    for index, dim in enumerate(out):
        if dim == -1:
            infer_index = index
        else:
            known *= dim
    if infer_index >= 0:
        out[infer_index] = numel // known
    return tuple(out)


def _contiguous_strides(shape):
    strides = []
    running = 1
    for dim in reversed(shape):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


@oracle_impl(hardware="B200", point="3d46c2a2")
@oracle_impl(hardware="B200", point="36a70cbd")
@oracle_impl(hardware="B200", point="79a90d04")
@oracle_impl(hardware="B200", point="2e43c9fa")
@oracle_impl(hardware="B200", point="3b895fcd")
@oracle_impl(hardware="B200", point="5bfb1085")
def oracle_forward(inputs):
    x, _shape0, shape1_arg = inputs
    shape1 = _resolve_shape(shape1_arg, x.numel())
    batch, seq, _three, heads, head_dim = shape1
    base_strides = _contiguous_strides(shape1)

    output_shape = (batch, heads, seq, head_dim)
    output_stride = (
        base_strides[0],
        base_strides[3],
        base_strides[1],
        base_strides[4],
    )
    base_offset = int(x.storage_offset())
    qkv_stride = base_strides[2]

    return (
        torch.as_strided(x, output_shape, output_stride, storage_offset=base_offset),
        torch.as_strided(
            x,
            output_shape,
            output_stride,
            storage_offset=base_offset + qkv_stride,
        ),
        torch.as_strided(
            x,
            output_shape,
            output_stride,
            storage_offset=base_offset + 2 * qkv_stride,
        ),
    )
