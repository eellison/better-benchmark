"""cuTile port of pointwise_783b3f21ba50: scaled attention head layout materialization.

Graph: view(x, shape0) -> mul(0.125, bf16) -> view(shape1=(B,S,H,D)) -> permute(0,2,1,3)
       -> clone contig -> view(shape2=(B*H, S, D)) -> return (view, view.permute(0,2,1)).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _scale_permute_clone_kernel(
    src_ptr,          # bf16 (B, S, H, D) contig  (source of the permute)
    dst_ptr,          # bf16 (B, H, S, D) contig
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    hs = ct.bid(1)
    # Decompose hs into (H, S) using ct.cdiv-style loops in Python.
    # Better: use 3 grid dims.
    pass


@ct.kernel
def _scale_perm_kernel(
    src_ptr,         # bf16 (B, S, H, D)
    dst_ptr,         # bf16 (B, H, S, D)
    BLOCK_D: ct.Constant[int],
):
    b = ct.bid(0)
    s = ct.bid(1)
    h = ct.bid(2)
    # Load one row of D elements from (b, s, h, :)
    tile = ct.load(src_ptr, index=(b, s, h, 0), shape=(1, 1, 1, BLOCK_D))
    # Scale by 0.125 in fp32 with bf16 output.
    tile_f = ct.astype(tile, ct.float32)
    scaled = tile_f * 0.125
    scaled_bf = ct.astype(scaled, ct.bfloat16)
    # Store at (b, h, s, :)
    ct.store(dst_ptr, index=(b, h, s, 0), tile=scaled_bf)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_D=64)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_D=64)
def oracle_forward(inputs, *, BLOCK_D: int):
    arg0_1, shape0, shape1, shape2 = inputs
    shape0_t = tuple(int(d) for d in shape0)  # (B, S, H*D)
    shape1_t = tuple(int(d) for d in shape1)  # (B, S, H, D)
    shape2_t = tuple(int(d) for d in shape2)  # (B*H, S, D)

    b, s, h, d = shape1_t
    # Source: reshape x to (B, S, H, D). x may be (B*S, H*D).
    src = arg0_1.view(*shape1_t)

    # Destination is contig (B, H, S, D). Then viewed as (B*H, S, D).
    clone = torch.empty_strided(
        (b, h, s, d),
        _contiguous_stride((b, h, s, d)),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(stream, (b, s, h), _scale_perm_kernel, (src, clone, BLOCK_D))

    view_2 = clone.view(shape2_t)
    return view_2, view_2.permute(0, 2, 1)
