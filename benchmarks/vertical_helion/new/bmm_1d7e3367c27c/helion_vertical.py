"""Softmax over 9 taps -> per-row [64 x 9] @ [9] matvec (ConvBERT dynamic conv, 24 occ).

Graph: softmax over 9 logits per (token, group), bf16 probs, bmm with the
unfolded inputs [98304, 64, 9] @ probs [98304, 9, 1] -> [32, 512, 6, 64].

One program per row tile: probs (F = 9) and the matvec stay on chip. The
matvec has N = 1 (no tensor cores); reading the 113 MB unfolded input
dominates, so fusion only saves the small probs round trip.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per row tile; memory-bound N=1 matvec."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(z, a):
    R, E = z.shape
    E = hl.specialize(E)
    D = hl.specialize(a.size(1))
    out = torch.empty([R, D], dtype=torch.bfloat16, device=z.device)
    for tile_r in hl.tile(R):
        v = z[tile_r, :]
        e = torch.exp(v - torch.amax(v, dim=-1, keepdim=True))
        p = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
        prod = a[tile_r, :, :].to(torch.float32) * p.to(torch.float32)[:, None, :]
        out[tile_r, :] = torch.sum(prod, dim=-1).to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def softmax_kernel(z):
    R, E = z.shape
    E = hl.specialize(E)
    p_out = torch.empty([R, E], dtype=torch.bfloat16, device=z.device)
    for tile_r in hl.tile(R):
        v = z[tile_r, :]
        e = torch.exp(v - torch.amax(v, dim=-1, keepdim=True))
        p_out[tile_r, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
    return p_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def matvec_kernel(a, p):
    R, D, E = a.shape
    E = hl.specialize(E)
    D = hl.specialize(D)
    out = torch.empty([R, D], dtype=torch.bfloat16, device=a.device)
    for tile_r in hl.tile(R):
        prod = a[tile_r, :, :].to(torch.float32) * p[tile_r, :].to(torch.float32)[:, None, :]
        out[tile_r, :] = torch.sum(prod, dim=-1).to(torch.bfloat16)
    return out


def fused(z, a, shape0):
    return fused_kernel(z.view(z.size(0), -1), a).view(shape0)


def unfused(z, a, shape0):
    p = softmax_kernel(z.view(z.size(0), -1))
    return matvec_kernel(a, p).view(shape0)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
