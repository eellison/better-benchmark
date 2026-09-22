"""cuTile port of pointwise_b3f01ad9586c: grouped-KV expand-clone.

Duplicates KV heads across GROUPS=2 output heads. The `permute` output is a
metadata-only alias view of the input.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


S_TILE = 8  # divides 1000 exactly


@ct.kernel
def _grouped_kv_kernel(
    src,  # [SEQ, KV_HEADS, HEAD_DIM] bf16 contig
    out,  # [OUT_HEADS, SEQ, HEAD_DIM] bf16 contig
    S_TILE_c: ct.Constant[int],
    HEAD_DIM_c: ct.Constant[int],
):
    kv = ct.bid(0)
    st = ct.bid(1)
    values = ct.load(src, index=(st, kv, 0), shape=(S_TILE_c, 1, HEAD_DIM_c))
    out_tile = ct.reshape(values, (1, S_TILE_c, HEAD_DIM_c))
    ct.store(out, index=(2 * kv, st, 0), tile=out_tile)
    ct.store(out, index=(2 * kv + 1, st, 0), tile=out_tile)


@oracle_impl(hardware="B200", point="ed385436")
def oracle_forward(inputs):
    arg0_1, _shape_param_0, _shape_param_1, expand_shape, out_shape = inputs

    batch = int(expand_shape[0])
    kv_heads = int(expand_shape[1])
    groups = int(expand_shape[2])
    seq = int(expand_shape[3])
    head_dim = int(expand_shape[4])

    permute = arg0_1.as_strided(
        (batch, kv_heads, seq, head_dim),
        (seq * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1),
    )

    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * seq * head_dim, seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    src_3d = arg0_1.view(seq, kv_heads, head_dim)
    out_3d = out.view(kv_heads * groups, seq, head_dim)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (kv_heads, seq // S_TILE, 1),
        _grouped_kv_kernel,
        (src_3d, out_3d, S_TILE, head_dim),
    )
    return permute, out
