"""Attention with fully-masked-row guard (Blenderbot/DistilBERT, 56 occurrences).

Graph: scores = q @ k^T (bf16, K = head_dim 80 or 64), rows that are all -inf
get probs 0 (any/logical_not), softmax over 128 keys in f32, bf16 probs @ V,
token-major output.

One program per (batch*head, query tile) keeps its 128-wide score rows: the
any, amax and sum reductions and P @ V need nothing else. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (bh, row tile), 128-wide score rows on chip."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(q, kt, v, B: hl.constexpr):
    BH, S, D = q.shape
    J = hl.specialize(kt.size(2))
    Dv = hl.specialize(v.size(2))
    H = BH // B
    out = torch.empty([B, S, H, Dv], dtype=torch.bfloat16, device=q.device)
    for tile_bh, tile_i in hl.tile([BH, S], block_size=[1, None]):
        bh = tile_bh.begin
        s = hl.zeros([tile_i, J], dtype=torch.float32)
        for tile_d in hl.tile(D):
            s = torch.addmm(s, q[bh, tile_i, tile_d], kt[bh, tile_d, :])
        s = s.to(torch.bfloat16).to(torch.float32)
        live = torch.amax((s != float("-inf")).to(torch.float32), dim=-1, keepdim=True)
        e = torch.exp(s - torch.amax(s, dim=-1, keepdim=True))
        p = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
        p = torch.where(live > 0, p, 0.0).to(torch.bfloat16)
        o = hl.dot(p, v[bh, :, :], out_dtype=torch.float32)
        out[bh // H, tile_i, bh % H, :] = o.to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def scores_kernel(q, kt):
    BH, S, D = q.shape
    J = kt.size(2)
    s_out = torch.empty([BH, S, J], dtype=torch.bfloat16, device=q.device)
    for tile_bh, tile_i, tile_j in hl.tile([BH, S, J], block_size=[1, None, None]):
        bh = tile_bh.begin
        acc = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        for tile_d in hl.tile(D):
            acc = torch.addmm(acc, q[bh, tile_i, tile_d], kt[bh, tile_d, tile_j])
        s_out[bh, tile_i, tile_j] = acc.to(torch.bfloat16)
    return s_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def softmax_kernel(s_in):
    BH, S, J = s_in.shape
    J = hl.specialize(J)
    p_out = torch.empty([BH, S, J], dtype=torch.bfloat16, device=s_in.device)
    for tile_bh, tile_i in hl.tile([BH, S]):
        s = s_in[tile_bh, tile_i, :].to(torch.float32)
        live = torch.amax((s != float("-inf")).to(torch.float32), dim=-1, keepdim=True)
        e = torch.exp(s - torch.amax(s, dim=-1, keepdim=True))
        p = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
        p_out[tile_bh, tile_i, :] = torch.where(live > 0, p, 0.0).to(torch.bfloat16)
    return p_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def pv_kernel(p, v, B: hl.constexpr):
    BH, S, J = p.shape
    Dv = v.size(2)
    H = BH // B
    out = torch.empty([B, S, H, Dv], dtype=torch.bfloat16, device=p.device)
    for tile_bh, tile_i, tile_d in hl.tile([BH, S, Dv], block_size=[1, None, None]):
        bh = tile_bh.begin
        acc = hl.zeros([tile_i, tile_d], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc = torch.addmm(acc, p[bh, tile_i, tile_j], v[bh, tile_j, tile_d])
        out[bh // H, tile_i, bh % H, tile_d] = acc.to(torch.bfloat16)
    return out


def fused(q, kt, v, shape0, *shape_params):
    B = shape0[0]
    out = fused_kernel(q, kt, v, B)
    return out.view(out.size(0) * out.size(1), -1)


def unfused(q, kt, v, shape0, *shape_params):
    B = shape0[0]
    out = pv_kernel(softmax_kernel(scores_kernel(q, kt)), v, B)
    return out.view(out.size(0) * out.size(1), -1)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
