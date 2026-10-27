"""cuTile port of pointwise_9640aefa2584: grouped KV expand-clone.

Materializes the contiguous repeated-head clone: for each (batch, kv_head)
loads `src[:, seq, kv_head, :]` and writes into `dst[batch, kv_head*groups+g, :, :]`
for g in 0..groups-1. Also returns a strided alias of the source as `slice_1`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _grouped_kv_kernel(
    src,  # (B, S, KV, D) bf16
    dst,  # (B, KV*G, S, D) bf16
    S: ct.Constant[int],
    D: ct.Constant[int],
    GROUPS: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    b = ct.bid(0)
    kv = ct.bid(1)
    tile = ct.bid(2)
    # Load [1, BLOCK/D, 1, D] tile from src at (b, tile*BLOCK/D, kv, 0)
    # But we want to tile 1D flat over S*D
    # Simpler: load tile as [1, seq_block, 1, D] with seq_block = BLOCK / D
    # Actually since D is the innermost of both src and dst, we can tile as [BLOCK] on flat.
    # Load flat: seq_stride*S*D elements starting at (b, 0, kv, 0)?
    # Use 4D indexing: (b, seq_tile, kv, 0) with shape (1, seq_block, 1, D)
    seq_block: ct.Constant[int] = BLOCK // D
    values = ct.load(src, index=(b, tile, kv, 0), shape=(1, seq_block, 1, D))
    values_reshaped = ct.reshape(values, (1, 1, seq_block, D))
    for g in ct.static_iter(range(GROUPS)):
        out_head = kv * GROUPS + g
        ct.store(dst, index=(b, out_head, tile, 0), tile=values_reshaped)


@oracle_impl(hardware="B200", point="ed385436", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, _shape_param_0, _shape_param_1, expand_shape, out_shape = inputs

    batch = int(expand_shape[0])
    kv_heads = int(expand_shape[1])
    groups = int(expand_shape[2])
    seq = int(expand_shape[3])
    head_dim = int(expand_shape[4])

    slice_1 = arg0_1.as_strided(
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
    # View src as [B, S, KV, D]
    src_4d = arg0_1.view(batch, seq, kv_heads, head_dim)

    # Tile: (B, KV, seq_tiles) where seq_tiles = seq / (BLOCK/D)
    seq_block = BLOCK // head_dim
    if seq % seq_block != 0:
        raise NotImplementedError(
            f"cuTile port unsupported: seq={seq} not divisible by seq_block={seq_block}."
        )
    grid = (batch, kv_heads, seq // seq_block)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _grouped_kv_kernel,
        (src_4d, out, seq, head_dim, groups, BLOCK),
    )
    return out, slice_1
