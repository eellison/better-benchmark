"""Q and K projections -> per-head scores (Albert/BERT, 96 occurrences).

Graph: q = x @ Wq + b, k = x' @ Wk + b' (both K = 4096), split into 64 heads
of 64, scaled, scores = q_h @ k_h^T over 512 x 512 per head; outputs
scores != -inf (bool) and scores as f32.

Slice analysis: scores[b, h, i, j] needs q[b, i, h, :] AND k[b, j, h, :],
both internal MM results. A query slice needs ALL keys of the head, a key
slice needs ALL queries, so the only slice is (b, h) with
F(k) = 512 * 64 = 32768 values. Old partitioner: kept (first MM any K,
chained bmm K = 64 <= 128). New spec: rejected (R = 4096, F = 32768).

The fused kernel below is the only one-kernel form without recomputation:
one program per (b, h) projects all 512 keys into a [512, 64] accumulator
and keeps it while it walks the query tiles.
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
    "Only one-kernel form without recomputation keeps all 512 projected keys "
    "of a head ([512, 64] fp32) on chip per program."
)
SCALE = 0.3535533905932738


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, wq, bq, xk, wk, bk):
    B, S, C = x.shape
    _, H, D = wq.shape
    S = hl.specialize(S)
    D = hl.specialize(D)
    mask = torch.empty([B, H, S, S], dtype=torch.bool, device=x.device)
    scores = torch.empty([B, H, S, S], dtype=torch.float32, device=x.device)
    for tile_b, tile_h in hl.tile([B, H], block_size=[1, 1]):
        b, h = tile_b.begin, tile_h.begin
        k = hl.zeros([S, D], dtype=torch.float32)
        for tile_c in hl.tile(C):
            k = torch.addmm(k, xk[b, :, tile_c], wk[tile_c, h, :])
        kb = (((k + bk[h, :].to(torch.float32)).to(torch.bfloat16)) * SCALE).to(torch.bfloat16)
        for tile_i in hl.tile(S):
            q = hl.zeros([tile_i, D], dtype=torch.float32)
            for tile_c in hl.tile(C):
                q = torch.addmm(q, x[b, tile_i, tile_c], wq[tile_c, h, :])
            qb = (((q + bq[h, :].to(torch.float32)).to(torch.bfloat16)) * SCALE).to(torch.bfloat16)
            s = hl.dot(qb, kb.permute(1, 0), out_dtype=torch.float32).to(torch.bfloat16)
            mask[b, h, tile_i, :] = torch.logical_not(s == float("-inf"))
            scores[b, h, tile_i, :] = s.to(torch.float32)
    return mask, scores


@helion.kernel(static_shapes=True, autotune_effort="quick")
def proj_kernel(x, w, bias):
    """q or k projection, scaled, head-major [B, H, S, D]."""
    B, S, C = x.shape
    _, H, D = w.shape
    D = hl.specialize(D)
    out = torch.empty([B, H, S, D], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_s, tile_h in hl.tile([B, S, H], block_size=[1, None, 1]):
        b, h = tile_b.begin, tile_h.begin
        acc = hl.zeros([tile_s, D], dtype=torch.float32)
        for tile_c in hl.tile(C):
            acc = torch.addmm(acc, x[b, tile_s, tile_c], w[tile_c, h, :])
        out[b, h, tile_s, :] = (
            ((acc + bias[h, :].to(torch.float32)).to(torch.bfloat16)) * SCALE
        ).to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def scores_kernel(q, k):
    B, H, S, D = q.shape
    D = hl.specialize(D)
    mask = torch.empty([B, H, S, S], dtype=torch.bool, device=q.device)
    scores = torch.empty([B, H, S, S], dtype=torch.float32, device=q.device)
    for tile_b, tile_h, tile_i, tile_j in hl.tile([B, H, S, S], block_size=[1, 1, None, None]):
        b, h = tile_b.begin, tile_h.begin
        s = hl.dot(q[b, h, tile_i, :], k[b, h, tile_j, :].permute(1, 0), out_dtype=torch.float32)
        s = s.to(torch.bfloat16)
        mask[b, h, tile_i, tile_j] = torch.logical_not(s == float("-inf"))
        scores[b, h, tile_i, tile_j] = s.to(torch.float32)
    return mask, scores


def _inputs(wq, bq, x, bk, xk, wk, shape0):
    B, S, H, D = shape0
    return (
        x.view(B, S, -1), wq.view(wq.size(0), H, D), bq.view(H, D),
        xk.view(B, S, -1), wk.view(wk.size(0), H, D), bk.view(H, D),
    )


def fused(wq, bq, x, bk, xk, wk, shape0, *shape_params):
    return fused_kernel(*_inputs(wq, bq, x, bk, xk, wk, shape0))


def unfused(wq, bq, x, bk, xk, wk, shape0, *shape_params):
    x3, wq3, bq2, xk3, wk3, bk2 = _inputs(wq, bq, x, bk, xk, wk, shape0)
    return scores_kernel(proj_kernel(x3, wq3, bq2), proj_kernel(xk3, wk3, bk2))


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
