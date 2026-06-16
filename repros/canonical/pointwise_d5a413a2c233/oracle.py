"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GPT-style Q/K/V split-view-permute scope by returning the three final `[B, H, S, D]` aliases directly from the original bf16 projection storage with the exact eager storage offsets and strides, whereas Inductor also reduces the captured view/split/getitem/view/permute graph to metadata-only alias returns and the benchmark harness captures an empty CUDA graph; Inductor cannot produce a meaningful device-kernel floor for this today because the whole computation is host-side tensor metadata and alias construction rather than GPU work; the fix is ALGEBRAIC_ELIMINATION: canonicalize this multi-output QKV metadata chain to direct alias-return algebra and classify empty-graph measurements as invalid instead of a kernel floor."""

import torch

from oracle_harness import oracle_impl


def _split_permute_view(x, shape0, head_shape, chunk_index):
    batch = int(shape0[0])
    seq = int(shape0[1])
    hidden = int(shape0[2])
    heads = int(head_shape[2])
    head_dim = int(head_shape[3])
    chunk = heads * head_dim
    return torch.as_strided(
        x,
        (batch, heads, seq, head_dim),
        (seq * hidden, head_dim, hidden, 1),
        storage_offset=int(x.storage_offset()) + chunk_index * chunk,
    )


@oracle_impl(hardware="B200", point="74f1c5d9")
@oracle_impl(hardware="B200", point="d8de82d9")
def oracle_forward(inputs):
    x, shape0, shape1, shape2, shape3 = inputs
    return (
        _split_permute_view(x, shape0, shape1, 1),
        _split_permute_view(x, shape0, shape2, 2),
        _split_permute_view(x, shape0, shape3, 0),
    )
