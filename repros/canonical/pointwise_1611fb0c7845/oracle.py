"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 QKV split/view/permute scope as three direct `as_strided` aliases of the original projection storage, whereas Inductor routes the metadata-only view/split/getitem/view/permute chain through compiled-callable benchmarking and the locked CUDAGraph bench sees no executable GPU graph; Inductor cannot do this today because output planning does not canonicalize multi-output split-plus-layout metadata into direct alias returns before measurement; the fix is ALGEBRAIC_ELIMINATION: collapse this QKV metadata chain to storage-offset algebra and treat empty-graph timing as an alias-only graph artifact, not a device-kernel floor."""

import torch
import triton  # noqa: F401

from oracle_harness import oracle_impl


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    known = 1
    infer_index = -1
    for index, dim in enumerate(dims):
        if dim == -1:
            infer_index = index
        else:
            known *= dim
    if infer_index >= 0:
        dims[infer_index] = numel // known
    return tuple(dims)


def _qkv_aliases(inputs):
    x, shape0, shape1, _shape2, _shape3 = inputs
    batch, seq, hidden3 = _resolve_shape(shape0, x.numel())
    _batch1, _seq1, heads, head_dim = _resolve_shape(shape1, x.numel() // 3)
    chunk = heads * head_dim

    output_shape = (batch, heads, seq, head_dim)
    output_stride = (seq * hidden3, head_dim, hidden3, 1)
    base_offset = int(x.storage_offset())

    return (
        torch.as_strided(x, output_shape, output_stride, storage_offset=base_offset),
        torch.as_strided(x, output_shape, output_stride, storage_offset=base_offset + chunk),
        torch.as_strided(
            x,
            output_shape,
            output_stride,
            storage_offset=base_offset + 2 * chunk,
        ),
    )


# 74f1c5d9: (T([16384,2304], bf16), S([32,512,2304]), 3x S([32,512,12,64]))
@oracle_impl(hardware="B200", point="74f1c5d9")
# d8de82d9: (T([8192,2304], bf16), S([8,1024,2304]), 3x S([8,1024,12,64]))
@oracle_impl(hardware="B200", point="d8de82d9")
def oracle_forward(inputs):
    return _qkv_aliases(inputs)
