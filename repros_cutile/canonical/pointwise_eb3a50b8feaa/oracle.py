"""cuTile port of pointwise_eb3a50b8feaa (NEW_PATTERN): grouped-KV clone.
Materializes the [1,8,1000,256] output by copying each of 4 KV heads into 2 group slots.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


KV = 4
GROUPS = 2
S = 1000
D = 256


@ct.kernel
def _grouped_kv_clone_kernel(
    src_ptr,   # bf16 [S, KV, D]  (physical [1000, 4, 256], contig)
    out_ptr,   # bf16 [KV, GROUPS, S, D]
    BLOCK_S: ct.Constant[int],
    S_C: ct.Constant[int],
    D_C: ct.Constant[int],
    KV_C: ct.Constant[int],
):
    kv_head = ct.bid(0)
    s_tile = ct.bid(1)

    # Load [BLOCK_S, D] from src with tile-space index (s_tile, kv_head, 0)
    values = ct.load(src_ptr, index=(s_tile, kv_head, 0), shape=(BLOCK_S, 1, D_C))
    # values: [BLOCK_S, 1, D_C]
    # Store into out at (kv_head, group, s_tile, 0) for each group
    values_out = ct.reshape(values, (1, 1, BLOCK_S, D_C))
    ct.store(out_ptr, index=(kv_head, 0, s_tile, 0), tile=values_out)
    ct.store(out_ptr, index=(kv_head, 1, s_tile, 0), tile=values_out)


@oracle_impl(hardware="B200", point="ed385436", BLOCK_N=1024)
def oracle_forward(inputs, *, BLOCK_N: int):
    arg0, _shape0, _shape1, expand_shape, out_shape = inputs
    # arg0 shape [1000, 1024], view as [S=1000, KV=4, D=256]
    src = arg0.view(S, KV, D)

    out = torch.empty_strided(
        tuple(int(dim) for dim in out_shape),
        (KV * GROUPS * S * D, S * D, D, 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )
    # Reshape out as [KV, GROUPS, S, D]
    out_view = out.view(KV, GROUPS, S, D)

    # For BLOCK_S: pick 8 (1000/8=125, 8 divides 1000 evenly)
    # Actually let's pick BLOCK_S=1 so the tile is (1, 1, 256), simple
    # But that's 1000*4 = 4000 launches. Let's pick BLOCK_S=8: 125 * 4 = 500 launches
    BLOCK_S = 8

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (KV, S // BLOCK_S, 1),
        _grouped_kv_clone_kernel,
        (src, out_view, BLOCK_S, S, D, KV),
    )
    slice_view = torch.as_strided(
        arg0,
        (1, KV, S, D),
        (S * KV * D, D, KV * D, 1),
    )
    return out, slice_view
