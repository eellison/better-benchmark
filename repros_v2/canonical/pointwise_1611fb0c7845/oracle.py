"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 GPT-style Q/K/V view/split/view/permute scope by returning the three final `[B, 12, S, 64]` aliases directly from the original contiguous projection storage with the exact eager storage offsets and strides, whereas Inductor still routes the metadata-only layout graph through a compiled callable before producing equivalent aliases; Inductor cannot turn this into useful Triton work because the full repro has no arithmetic, casts, RNG, reductions, or materialization once the view algebra is simplified; the fix is ALGEBRAIC_ELIMINATION: canonicalize split-layout metadata chains into direct storage-offset alias returns and classify empty-CUDAGraph timing as invalid rather than a device-kernel floor."""

import torch
import triton  # noqa: F401

from oracle_harness import oracle_impl


@oracle_impl(hardware="B200", point="74f1c5d9")
@oracle_impl(hardware="B200", point="d8de82d9")
def oracle_forward(inputs):
    x, shape0, shape1, _shape2, _shape3 = inputs
    _batch0, seq, hidden3 = (int(dim) for dim in shape0)
    batch, _seq1, heads, head_dim = (int(dim) for dim in shape1)
    chunk = heads * head_dim
    output_shape = (batch, heads, seq, head_dim)
    output_stride = (seq * hidden3, head_dim, hidden3, 1)
    base = int(x.storage_offset())
    return (
        torch.as_strided(x, output_shape, output_stride, storage_offset=base),
        torch.as_strided(x, output_shape, output_stride, storage_offset=base + chunk),
        torch.as_strided(x, output_shape, output_stride, storage_offset=base + 2 * chunk),
    )
