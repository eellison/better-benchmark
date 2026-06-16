"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete ViT QKV view/view/permute/unbind scope as two direct `as_strided` aliases into the original bf16 projection storage, preserving the returned shapes, strides, and storage offsets with no data movement, whereas Inductor still routes the metadata-only layout algebra through a compiled callable before producing the same aliases; Inductor cannot materially improve this through Triton scheduling because the full repro has no arithmetic, casts, RNG, reductions, or materialization once the view chain is algebraically eliminated; the fix is ALGEBRAIC_ELIMINATION: canonicalize QKV metadata-only reshape/permute/unbind chains into direct storage-offset alias returns and treat the resulting empty device-work graph as the floor."""

import torch
import triton  # noqa: F401

from oracle_harness import oracle_impl


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


# cbc8ee1c: (T([32768,1536], bf16), S([128,256,1536]), S([128,256,2,12,64]))
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

    return (
        torch.as_strided(x, output_shape, output_stride, storage_offset=base_offset),
        torch.as_strided(
            x,
            output_shape,
            output_stride,
            storage_offset=base_offset + unbind_stride,
        ),
    )
