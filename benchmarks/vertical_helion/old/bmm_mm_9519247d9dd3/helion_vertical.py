"""Q projection -> per-head attention scores -> mask scaling, as Helion kernels.

Region (BERT/Electra, 84 occurrences): q = x @ Wq (K=768), split into heads,
s = q_h @ k_h^T (K=64), then bf16 * m1 -> fp32 * m2. Output [B, H, S, S] fp32.

Slice analysis: an output row (b, h, i) needs q[b, i, h, :] (64 values) and
the external k_h^T. Different (b, h, i) need disjoint q rows, so one program
per (b, h, i-tile) computes its q tile once (K-loop over 768) and streams the
scores over j. No recomputation; on-chip intermediate F = 64 per row.
New-spec verdict: rejected (first MM R = 768 > 192). Old partitioner: kept.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = (
    "Fusable without recomputation: program per (b, h, i-tile) computes q "
    "tile with a K=768 loop, keeps it (F=64) and streams scores over j."
)


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, wq, kt, m1, m2):
    B, H, S, _ = m1.shape
    D = hl.specialize(kt.size(1))
    K = x.size(1)
    x3 = x.view(B, S, K)
    w3 = wq.view(K, H, D)
    kt4 = kt.view(B, H, D, S)
    out = torch.empty([B, H, S, S], dtype=torch.float32, device=x.device)
    for tile_b, tile_h, tile_i in hl.tile([B, H, S], block_size=[1, 1, None]):
        b, h = tile_b.begin, tile_h.begin
        q = hl.zeros([tile_i, D], dtype=torch.float32)
        for tile_k in hl.tile(K):
            q = torch.addmm(q, x3[b, tile_i, tile_k], w3[tile_k, h, :])
        qb = q.to(x.dtype)
        for tile_j in hl.tile(S):
            s = hl.dot(qb, kt4[b, h, :, tile_j], out_dtype=torch.float32).to(x.dtype)
            s = s * m1[b, h, tile_i, tile_j]
            out[b, h, tile_i, tile_j] = s.to(torch.float32) * m2[b, h, tile_i, tile_j]
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def q_proj_kernel(x, wq):
    """Stage 1: q = x @ Wq; stage 2 reads it per head (no permute copy)."""
    M, K = x.size()
    N = wq.size(1)
    out = torch.empty([M, N], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[tile_m, tile_k], wq[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def scores_kernel(q4, kt, m1, m2):
    """Stage 2: s = q_h @ k_h^T, then the mask scaling epilogue."""
    B, H, S, _ = m1.shape
    D = hl.specialize(kt.size(1))
    kt4 = kt.view(B, H, D, S)
    out = torch.empty([B, H, S, S], dtype=torch.float32, device=m1.device)
    for tile_b, tile_h, tile_i, tile_j in hl.tile([B, H, S, S], block_size=[1, 1, None, None]):
        b, h = tile_b.begin, tile_h.begin
        s = hl.dot(q4[b, tile_i, h, :], kt4[b, h, :, tile_j], out_dtype=torch.float32)
        s = s.to(q4.dtype) * m1[b, h, tile_i, tile_j]
        out[b, h, tile_i, tile_j] = s.to(torch.float32) * m2[b, h, tile_i, tile_j]
    return out


def fused(x, wq, kt, m1, m2, *shape_params):
    return fused_kernel(x, wq, kt, m1, m2)


def unfused(x, wq, kt, m1, m2, *shape_params):
    B, H = m1.shape[:2]
    q = q_proj_kernel(x, wq)  # [B*S, H*D]
    q4 = q.view(B, -1, H, q.size(1) // H)  # [B, S, H, D] view, read per head
    return scores_kernel(q4, kt, m1, m2)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
