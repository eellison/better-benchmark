"""Projection (K = 768) -> sum over the 128 rows (ViT-SigLIP head, 1 occurrence).

Graph: mm [128, 768] @ [768, 768] -> f32 -> sum over the rows -> [1, 1, 768].
Only the column sums are returned.

New spec: rejected (first MM R = 768 > 192). Old partitioner: kept.
One program per column tile streams K = 768 and keeps all 128 rows of its
columns (F = 128), then sums them. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per column tile, K = 768 streamed, F = 128 rows."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, w):
    M, K = x.shape
    M = hl.specialize(M)
    N = w.size(1)
    out = torch.empty([N], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(N):
        acc = hl.zeros([M, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[:, tile_k], w[tile_k, tile_n])
        out[tile_n] = torch.sum(acc.to(torch.bfloat16).to(torch.float32), dim=0)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def mm_kernel(x, w):
    M, K = x.shape
    N = w.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def colsum_kernel(y):
    M, N = y.shape
    M = hl.specialize(M)
    out = torch.empty([N], dtype=torch.float32, device=y.device)
    for tile_n in hl.tile(N):
        out[tile_n] = torch.sum(y[:, tile_n].to(torch.float32), dim=0)
    return out


def fused(x, w, *shape_params):
    return fused_kernel(x.reshape(x.size(0), -1), w).view(1, 1, -1)


def unfused(x, w, *shape_params):
    return colsum_kernel(mm_kernel(x.reshape(x.size(0), -1), w)).view(1, 1, -1)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
