"""cuTile port of pointwise_7939829e1a9f: Longformer head layout materialization.

Rewrites the view/permute chain as (B,S,H,D) -> (B,H,S,D) then trivial reshape
to (B*H*S/CHUNK, CHUNK, D). Only the (B,S,H,D)->(B,H,S,D) permutation is
actual work; the rest is metadata."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


B = 8
S = 1024
H = 12
D = 64


@ct.kernel
def _permute_kernel(
    src,  # (B, S, H, D) bf16
    dst,  # (B, H, S, D) bf16
    BLOCK_S: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    s_block = ct.bid(2)
    # Load (1, BLOCK_S, 1, D) tile from src at (b, s_block, h, 0).
    tile = ct.load(src, index=(b, s_block, h, 0), shape=(1, BLOCK_S, 1, D))
    # Reshape to (1, 1, BLOCK_S, D) for the destination.
    tile2 = ct.reshape(tile, (1, 1, BLOCK_S, D))
    ct.store(dst, index=(b, h, s_block, 0), tile=tile2)


@oracle_impl(hardware="B200", point="07bfd41e")
def oracle_forward(inputs):
    arg0_1, *_ = inputs
    src4 = arg0_1.view(B, S, H, D)
    out = torch.empty_strided(
        (384, 256, 64),
        (256 * 64, 64, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    # (B*H*S/256, 256, D) = (96, 4, 256, 64) then flat to (384, 256, 64).
    # Since B*H = 96 and S = 1024 = 4 * 256, treat contiguously as (B, H, S, D).
    dst4 = out.view(B, H, S, D)

    BLOCK_S = 128
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (B, H, ct.cdiv(S, BLOCK_S)),
        _permute_kernel,
        (src4, dst4, BLOCK_S),
    )
    return out
