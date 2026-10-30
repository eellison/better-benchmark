"""Key projection -> per-head q @ k^T scores (MobileBERT, 48 occurrences).

Graph: k = x @ Wk + b ([32768, 128], K = 128), split into 4 heads of 32,
scaled and transposed, scores = q[bh] @ k^T[bh] (K = 32) -> [256, 4, 128, 128];
outputs: scores != -inf (bool) and scores as f32.

Slice analysis: an output column (b, h, t) needs k[b, t, h, :] (32 values);
q is an external input. One program per (b, h, key tile) projects its keys
once and computes the scores for all 128 queries. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (b, h, key tile) projects keys once (F = 32)."
SCALE = 0.4204482076268573


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, w, bias, q):
    B, T, C = x.shape
    BH, S, D = q.shape
    D = hl.specialize(D)
    S = hl.specialize(S)
    H = BH // B
    mask = torch.empty([BH, S, T], dtype=torch.bool, device=x.device)
    scores = torch.empty([BH, S, T], dtype=torch.float32, device=x.device)
    for tile_b, tile_h, tile_t in hl.tile([B, H, T], block_size=[1, 1, None]):
        b, h = tile_b.begin, tile_h.begin
        k = hl.zeros([tile_t, D], dtype=torch.float32)
        for tile_c in hl.tile(C):
            k = torch.addmm(k, x[b, tile_t, tile_c], w[tile_c, h, :])
        k = (k + bias[h, :].to(torch.float32)).to(torch.bfloat16)
        k = (k * SCALE).to(torch.bfloat16)
        s = hl.dot(q[b * H + h, :, :], k.permute(1, 0), out_dtype=torch.float32)
        s = s.to(torch.bfloat16)
        mask[b * H + h, :, tile_t] = torch.logical_not(s == float("-inf"))
        scores[b * H + h, :, tile_t] = s.to(torch.float32)
    return mask, scores


@helion.kernel(static_shapes=True, autotune_effort="quick")
def k_proj_kernel(x, w, bias):
    """Stage 1: k = (x @ Wk + b) * scale, stored [B, H, D, T] (k^T per head)."""
    B, T, C = x.shape
    _, H, D = w.shape
    kt = torch.empty([B, H, D, T], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_t, tile_h in hl.tile([B, T, H], block_size=[1, None, 1]):
        b, h = tile_b.begin, tile_h.begin
        acc = hl.zeros([tile_t, D], dtype=torch.float32)
        for tile_c in hl.tile(C):
            acc = torch.addmm(acc, x[b, tile_t, tile_c], w[tile_c, h, :])
        k = (acc + bias[h, :].to(torch.float32)).to(torch.bfloat16)
        kt[b, h, :, tile_t] = (k * SCALE).to(torch.bfloat16).permute(1, 0)
    return kt


@helion.kernel(static_shapes=True, autotune_effort="quick")
def scores_kernel(q, kt):
    """Stage 2: scores = q @ k^T with the two outputs."""
    BH, S, D = q.shape
    T = kt.size(2)
    mask = torch.empty([BH, S, T], dtype=torch.bool, device=q.device)
    scores = torch.empty([BH, S, T], dtype=torch.float32, device=q.device)
    for tile_bh, tile_s, tile_t in hl.tile([BH, S, T], block_size=[1, None, None]):
        bh = tile_bh.begin
        acc = hl.zeros([tile_s, tile_t], dtype=torch.float32)
        for tile_d in hl.tile(D):
            acc = torch.addmm(acc, q[bh, tile_s, tile_d], kt[bh, tile_d, tile_t])
        s = acc.to(torch.bfloat16)
        mask[bh, tile_s, tile_t] = torch.logical_not(s == float("-inf"))
        scores[bh, tile_s, tile_t] = s.to(torch.float32)
    return mask, scores


def _inputs(w, bias, x, q):
    B = q.size(0) // 4
    H, D = 4, q.size(2)
    return x.view(B, -1, x.size(1)), w.view(w.size(0), H, D), bias.view(H, D), q


def _outputs(mask, scores, q):
    B = q.size(0) // 4
    shape = (B, 4, q.size(1), -1)
    return mask.view(shape), scores.view(shape)


def fused(w, bias, x, q, *shape_params):
    return _outputs(*fused_kernel(*_inputs(w, bias, x, q)), q)


def unfused(w, bias, x, q, *shape_params):
    xb, w3, b2, q = _inputs(w, bias, x, q)
    kt = k_proj_kernel(xb, w3, b2)
    mask, scores = scores_kernel(q, kt.view(-1, kt.size(2), kt.size(3)))
    return _outputs(mask, scores, q)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
