"""Three-layer tanh MLP, widths 16 -> 16 -> 16 -> 1 (lennard_jones, 2 occurrences).

Graph: tanh(x @ W1 + b1) -> tanh(. @ W2 + b2) -> . @ W3 + b3, x [128, 16].

One program per row tile keeps the 16-wide hidden rows (F = 16). The last
layer has N = 1, below tl.dot's minimum, so it is a row sum.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per row tile, 16-wide hidden rows on chip."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, w1, b1, w2, b2, w3, b3):
    M, K = x.shape
    K = hl.specialize(K)
    N = hl.specialize(w1.size(1))
    out = torch.empty([M, 1], dtype=torch.bfloat16, device=x.device)
    for tile_m in hl.tile(M):
        h = hl.dot(x[tile_m, :], w1[:, :], out_dtype=torch.float32)
        h = torch.tanh((h + b1[:].to(torch.float32)).to(torch.bfloat16))
        h = hl.dot(h, w2[:, :], out_dtype=torch.float32)
        h = torch.tanh((h + b2[:].to(torch.float32)).to(torch.bfloat16))
        y = torch.sum(h.to(torch.float32) * w3[:, 0].to(torch.float32), dim=-1, keepdim=True)
        out[tile_m, :] = (y + b3[:].to(torch.float32)).to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def linear_tanh_kernel(x, w, b):
    M, K = x.shape
    K = hl.specialize(K)
    N = hl.specialize(w.size(1))
    out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
    for tile_m in hl.tile(M):
        h = hl.dot(x[tile_m, :], w[:, :], out_dtype=torch.float32)
        out[tile_m, :] = torch.tanh((h + b[:].to(torch.float32)).to(torch.bfloat16))
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def linear_n1_kernel(x, w, b):
    M, K = x.shape
    K = hl.specialize(K)
    out = torch.empty([M, 1], dtype=torch.bfloat16, device=x.device)
    for tile_m in hl.tile(M):
        y = torch.sum(x[tile_m, :].to(torch.float32) * w[:, 0].to(torch.float32), dim=-1, keepdim=True)
        out[tile_m, :] = (y + b[:].to(torch.float32)).to(torch.bfloat16)
    return out


def fused(w1, b1, x, w2, b2, w3, b3):
    return fused_kernel(x, w1, b1, w2, b2, w3, b3)


def unfused(w1, b1, x, w2, b2, w3, b3):
    h = linear_tanh_kernel(linear_tanh_kernel(x, w1, b1), w2, b2)
    return linear_n1_kernel(h, w3, b3)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
