"""cuTile port of pointwise_65717f3841e5: GPT-OSS grouped-KV expand.

Reads a bf16 `[seq, kv_heads*head_dim]` source and materializes a repeated
`[kv_heads * groups, seq, head_dim]` bf16 output plus a stridden slice view.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _grouped_kv_repeat_kernel(
    src,       # (seq, kv_heads, head_dim) bf16
    out,       # (kv_heads, groups, seq, head_dim) bf16
    BLOCK_SEQ: ct.Constant[int],
    D: ct.Constant[int],
    GROUPS: ct.Constant[int],
):
    kv_head = ct.bid(0)
    seq_block = ct.bid(1)

    # Load a tile (BLOCK_SEQ, 1, D) from src at (seq_block, kv_head, 0)
    values = ct.load(src, index=(seq_block, kv_head, 0), shape=(BLOCK_SEQ, 1, D))
    # Store to each of the GROUPS output heads at (kv_head, g, seq_block, 0)
    # in (1, 1, BLOCK_SEQ, D) tiles.
    values_out = ct.reshape(values, (1, 1, BLOCK_SEQ, D))
    for g in range(GROUPS):
        ct.store(out, index=(kv_head, g, seq_block, 0), tile=values_out)


@oracle_impl(hardware="B200", point="94ef836f", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0_1, _s0, _s1, expand_shape, _s3, _s4, out_shape = inputs
    _, kv_heads, groups, seq, head_dim = (int(dim) for dim in expand_shape)

    out = torch.empty_strided(
        tuple(int(dim) for dim in out_shape),
        (seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # arg0_1 is (seq, kv_heads * head_dim); view as (seq, kv_heads, head_dim)
    src_3d = arg0_1.view(seq, kv_heads, head_dim)
    # out is (kv_heads * groups, seq, head_dim); view as (kv_heads, groups, seq, head_dim)
    out_4d = out.view(kv_heads, groups, seq, head_dim)

    BLOCK_SEQ = 8  # seq=1000 not power of 2; pad by using empty extra rows.
    # Since seq=1000 is not power of 2, we need BLOCK_SEQ that divides 1000
    # or use a padded approach. 1000 = 8 * 125. cuTile requires pow2 tile
    # dims — pick 8, launch cdiv(1000, 8) = 125 tiles, but the last tile
    # extends to 1000 exactly (125*8 = 1000). Wait that's 1000 exactly if
    # 8*125=1000, and 125 is int. So we launch exactly 125 tiles of size 8.
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (kv_heads, (seq + BLOCK_SEQ - 1) // BLOCK_SEQ, 1),
        _grouped_kv_repeat_kernel,
        (src_3d, out_4d, BLOCK_SEQ, head_dim, groups),
    )

    slice_view = torch.as_strided(
        arg0_1,
        (1, kv_heads, 127, head_dim),
        (seq * kv_heads * head_dim, head_dim, kv_heads * head_dim, 1),
        storage_offset=(seq - 127) * kv_heads * head_dim,
    )
    return out, slice_view
