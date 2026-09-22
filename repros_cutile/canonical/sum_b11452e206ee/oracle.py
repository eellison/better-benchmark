"""cuTile port of sum_b11452e206ee: SigLIP QK cat/view/permute/clone + column sum roundtrip.

Uses torch to build the [32768, 1536] contiguous layout,
then a cuTile kernel to do the column sum + bf16 roundtrip.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 128
HEADS = 12
TOKENS = 256
HEAD_DIM = 64
SOURCES = 2


@ct.kernel
def _col_sum_bf16_roundtrip_kernel(
    input_ptr,     # bf16 [B, C]
    out_ptr,       # f32 [C]
    B: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_tile = ct.bid(0)
    tile = ct.load(input_ptr, index=(0, col_tile), shape=(B, BLOCK_C))
    tile_f = ct.astype(tile, ct.float32)
    col_sum = ct.sum(tile_f, axis=0)
    rounded_bf16 = ct.astype(col_sum, ct.bfloat16)
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    ct.store(out_ptr, index=(col_tile,), tile=rounded_f32)


@oracle_impl(hardware="B200", point="4ad1eedb")
def oracle_forward(inputs):
    arg0_1, arg1_1, view0_shape, view1_shape, view2_shape, sum_shape = inputs
    view0_shape = tuple(int(d) for d in view0_shape)
    view1_shape = tuple(int(d) for d in view1_shape)
    view2_shape = tuple(int(d) for d in view2_shape)

    cat = torch.cat([arg0_1, arg1_1])
    view = cat.view(view0_shape)
    permute = view.permute(1, 3, 0, 2, 4).contiguous()
    view_1 = permute.view(view1_shape)
    view_2 = view_1.view(view2_shape)

    B = int(view_2.shape[0])
    C = int(view_2.shape[1])
    BLOCK_C = 64

    out_flat = torch.empty(C, device=view_2.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C // BLOCK_C, 1, 1),
        _col_sum_bf16_roundtrip_kernel,
        (view_2, out_flat, B, BLOCK_C),
    )
    return (view_2, view_2.permute(1, 0), out_flat)
