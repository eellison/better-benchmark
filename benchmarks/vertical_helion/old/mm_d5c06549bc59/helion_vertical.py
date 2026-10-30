"""Linear (K = 384) -> softmax over groups of 9 logits (ConvBERT, 12 occurrences).

Graph: logits = x @ W + b ([16384, 54]), viewed as [98304, 9]: softmax over
each group of 9 columns, bf16 output.

New spec: rejected (first MM R = 384 > 192). Old partitioner: kept.
One program per row tile loads its x rows once (K = 384 is streamed from
the input, not an intermediate) and computes each of the 6 groups' 9 logits
and their softmax. Intermediate per row: F = 9 per group. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per row tile; first-MM K = 384 streamed, F = 9."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, w3, b2):
    M, K = x.shape
    G = w3.size(1)
    E = w3.size(2)
    out = torch.empty([M, G, E], dtype=torch.bfloat16, device=x.device)
    # E = 9 is one masked 16-wide tile, so the softmax sees whole groups.
    for tile_m, tile_g, tile_e in hl.tile([M, G, E], block_size=[None, 1, 16]):
        g = tile_g.begin
        acc = hl.zeros([tile_m, tile_e], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[tile_m, tile_k], w3[tile_k, g, tile_e])
        z = (acc.to(torch.bfloat16) + b2[g, tile_e]).to(torch.bfloat16).to(torch.float32)
        e = torch.exp(z - torch.amax(z, dim=-1, keepdim=True))
        out[tile_m, g, tile_e] = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def addmm_kernel(x, w, b):
    M, K = x.shape
    N = w.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = (acc.to(torch.bfloat16) + b[tile_n]).to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def group_softmax_kernel(z):
    R, E = z.shape
    E = hl.specialize(E)
    out = torch.empty([R, E], dtype=torch.bfloat16, device=z.device)
    for tile_r in hl.tile(R):
        v = z[tile_r, :].to(torch.float32)
        e = torch.exp(v - torch.amax(v, dim=-1, keepdim=True))
        out[tile_r, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
    return out


def fused(w, x, b, *shape_params):
    E = 9
    G = w.size(1) // E
    out = fused_kernel(x, w.view(w.size(0), G, E), b.view(G, E))
    return out.view(-1, E, 1)


def unfused(w, x, b, *shape_params):
    z = addmm_kernel(x, w, b)
    return group_softmax_kernel(z.view(-1, 9)).view(-1, 9, 1)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
