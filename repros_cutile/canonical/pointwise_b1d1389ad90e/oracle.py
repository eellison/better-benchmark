"""cuTile port of pointwise_b1d1389ad90e: DeBERTa attention layout-divide.

view(B,S,H,D).permute(0,2,1,3).clone.contiguous.view(B*H,S,D).div(8) plus aliases.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


_FULL = torch.full((), 8.0, dtype=torch.bfloat16)


@ct.kernel
def _layout_div_kernel(
    src,  # (B, S, H, D)
    dst,  # (B, H, S, D)
    D_: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s = ct.bid(2)
    tile = ct.load(src, index=(b, s, h, 0), shape=(1, 1, 1, D_))
    tile_f = ct.astype(tile, ct.float32) * 0.125
    tile_bf = ct.astype(tile_f, ct.bfloat16)
    ct.store(dst, index=(b, h, s, 0), tile=tile_bf)


@oracle_impl(hardware="B200", point="981155f5", BLOCK=4096)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, _shape_param_0, _shape_param_1, shape_param_2 = inputs

    seq = int(shape_param_2[1])
    head_dim = int(shape_param_2[2])
    heads = 24
    batch = int(arg0_1.numel() // (seq * heads * head_dim))
    output = torch.empty_strided(
        (batch * heads, seq, head_dim),
        (seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    src4 = arg0_1.view(batch, seq, heads, head_dim)
    dst4 = output.view(batch, heads, seq, head_dim)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (batch, heads, seq),
        _layout_div_kernel,
        (src4, dst4, head_dim),
    )
    return _FULL, output.permute(0, 2, 1), output
