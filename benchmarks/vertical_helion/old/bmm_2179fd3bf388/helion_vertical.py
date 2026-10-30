"""Attention scores (fp32, K = 256) -> mask -> softmax over 128 keys (GPT-J, 54 occ).

Graph: s = q @ k^T / 16 + mask, softmax over the last axis, bf16 probs.

New spec: rejected (first bmm R = 256 > 192); the new corpus keeps only the
softmax -> P @ V half (bmm_01f3eb169a50). Old partitioner: kept.
One program per (head, query tile) streams K = 256 from q/k and keeps the
128-wide score rows (F = 128). No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (head, row tile), K = 256 streamed, F = 128."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(q, kt, mask):
    H, S, D = q.shape
    J = hl.specialize(kt.size(2))
    probs = torch.empty([H, S, J], dtype=torch.bfloat16, device=q.device)
    for tile_h, tile_i in hl.tile([H, S], block_size=[1, None]):
        h = tile_h.begin
        s = hl.zeros([tile_i, J], dtype=torch.float32)
        for tile_d in hl.tile(D):
            s = torch.addmm(s, q[h, tile_i, tile_d], kt[h, tile_d, :])
        s = s / 16.0 + mask[tile_i, :].to(torch.float32)
        e = torch.exp(s - torch.amax(s, dim=-1, keepdim=True))
        probs[h, tile_i, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
    return probs


@helion.kernel(static_shapes=True, autotune_effort="quick")
def scores_kernel(q, kt, mask):
    H, S, D = q.shape
    J = kt.size(2)
    s_out = torch.empty([H, S, J], dtype=torch.float32, device=q.device)
    for tile_h, tile_i, tile_j in hl.tile([H, S, J], block_size=[1, None, None]):
        h = tile_h.begin
        s = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        for tile_d in hl.tile(D):
            s = torch.addmm(s, q[h, tile_i, tile_d], kt[h, tile_d, tile_j])
        s_out[h, tile_i, tile_j] = s / 16.0 + mask[tile_i, tile_j].to(torch.float32)
    return s_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def softmax_kernel(s_in):
    H, S, J = s_in.shape
    J = hl.specialize(J)
    probs = torch.empty([H, S, J], dtype=torch.bfloat16, device=s_in.device)
    for tile_h, tile_i in hl.tile([H, S]):
        s = s_in[tile_h, tile_i, :]
        e = torch.exp(s - torch.amax(s, dim=-1, keepdim=True))
        probs[tile_h, tile_i, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
    return probs


def fused(q, kt, mask, *shape_params):
    return fused_kernel(q, kt, mask[0, 0])


def unfused(q, kt, mask, *shape_params):
    return softmax_kernel(scores_kernel(q, kt, mask[0, 0]))


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
