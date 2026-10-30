"""Softmax over 128 keys -> P @ V (GPT-J attention, 112 occurrences).

Graph: amax/exp/sum/div over the last axis of f32 scores [1, 16, 128, 128],
bf16 probs @ V[h] (128 x 256), output re-laid out token-major [128, 4096].
Outputs: f32 probs, the attention output, and a transposed view of the bf16
probs (a view: no kernel).

One program per (head, query tile) holds its 128-wide score rows, so
amax -> sum -> bmm need nothing from other programs: F(amax) = F(sum) = 1,
F(probs) = 128 values per row feeding the bmm, accumulator [rows, 256].
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (head, row tile) keeps full 128-wide rows."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, v):
    _, H, S, J = x.shape
    J = hl.specialize(J)
    D = hl.specialize(v.size(2))
    x3 = x.view(H, S, J)
    probs = torch.empty([H, S, J], dtype=torch.float32, device=x.device)
    probs_bf = torch.empty([H, S, J], dtype=torch.bfloat16, device=x.device)
    out = torch.empty([S, H, D], dtype=torch.bfloat16, device=x.device)
    for tile_h, tile_i in hl.tile([H, S], block_size=[1, None]):
        h = tile_h.begin
        row = x3[h, tile_i, :]
        e = torch.exp(row - torch.amax(row, dim=-1, keepdim=True))
        p = e / torch.sum(e, dim=-1, keepdim=True)
        probs[h, tile_i, :] = p
        pb = p.to(torch.bfloat16)
        probs_bf[h, tile_i, :] = pb
        acc = hl.dot(pb, v[h, :, :], out_dtype=torch.float32)
        out[tile_i, h, :] = acc.to(torch.bfloat16)
    return probs.view(1, H, S, J), out.view(S, H * D), probs_bf.permute(0, 2, 1)


@helion.kernel(static_shapes=True, autotune_effort="quick")
def softmax_kernel(x):
    _, H, S, J = x.shape
    J = hl.specialize(J)
    x3 = x.view(H, S, J)
    probs = torch.empty([H, S, J], dtype=torch.float32, device=x.device)
    probs_bf = torch.empty([H, S, J], dtype=torch.bfloat16, device=x.device)
    for tile_h, tile_i in hl.tile([H, S], block_size=[1, None]):
        h = tile_h.begin
        row = x3[h, tile_i, :]
        e = torch.exp(row - torch.amax(row, dim=-1, keepdim=True))
        p = e / torch.sum(e, dim=-1, keepdim=True)
        probs[h, tile_i, :] = p
        probs_bf[h, tile_i, :] = p.to(torch.bfloat16)
    return probs, probs_bf


@helion.kernel(static_shapes=True, autotune_effort="quick")
def pv_kernel(pb, v):
    H, S, J = pb.shape
    D = v.size(2)
    out = torch.empty([S, H, D], dtype=torch.bfloat16, device=pb.device)
    for tile_h, tile_i, tile_d in hl.tile([H, S, D], block_size=[1, None, None]):
        h = tile_h.begin
        acc = hl.zeros([tile_i, tile_d], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc = torch.addmm(acc, pb[h, tile_i, tile_j], v[h, tile_j, tile_d])
        out[tile_i, h, tile_d] = acc.to(torch.bfloat16)
    return out


def fused(x, v, *shape_params):
    return fused_kernel(x, v)


def unfused(x, v, *shape_params):
    probs, probs_bf = softmax_kernel(x)
    out = pv_kernel(probs_bf, v)
    H, S, J = probs.shape
    return probs.view(1, H, S, J), out.view(S, -1), probs_bf.permute(0, 2, 1)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
