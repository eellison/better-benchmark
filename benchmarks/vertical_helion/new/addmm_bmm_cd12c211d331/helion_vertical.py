"""Query projection -> full self-attention over 128 keys (MobileBERT, 24 occ).

Graph: q = (x @ Wq + b) * scale (K = 128, 4 heads x 32), scores = q @ k^T
(K = 32), all -inf rows -> probs 0, softmax over 128 keys in f32, bf16 probs
@ V (K = 128), token-major [32768, 128].

One program per (batch, head, query tile) projects its own queries
(F = 32 per row), keeps the 128-wide score rows and does P @ V; k and V are
inputs. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (b, h, query tile) projects q (F = 32) and attends."
SCALE = 0.4204482076268573


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, w, bias, kt, v):
    B, S, C = x.shape
    _, H, D = w.shape
    D = hl.specialize(D)
    J = hl.specialize(kt.size(2))
    out = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_h, tile_i in hl.tile([B, H, S], block_size=[1, 1, None]):
        b, h = tile_b.begin, tile_h.begin
        q = hl.zeros([tile_i, D], dtype=torch.float32)
        for tile_c in hl.tile(C):
            q = torch.addmm(q, x[b, tile_i, tile_c], w[tile_c, h, :])
        qb = ((q + bias[h, :].to(torch.float32)).to(torch.bfloat16) * SCALE).to(torch.bfloat16)
        s = hl.dot(qb, kt[b * H + h, :, :], out_dtype=torch.float32)
        s = s.to(torch.bfloat16).to(torch.float32)
        live = torch.amax((s != float("-inf")).to(torch.float32), dim=-1, keepdim=True)
        e = torch.exp(s - torch.amax(s, dim=-1, keepdim=True))
        p = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
        p = torch.where(live > 0, p, 0.0).to(torch.bfloat16)
        o = hl.dot(p, v[b * H + h, :, :], out_dtype=torch.float32)
        out[b, tile_i, h, :] = o.to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def q_proj_kernel(x, w, bias):
    B, S, C = x.shape
    _, H, D = w.shape
    D = hl.specialize(D)
    q = torch.empty([B, H, S, D], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_h, tile_i in hl.tile([B, H, S], block_size=[1, 1, None]):
        b, h = tile_b.begin, tile_h.begin
        acc = hl.zeros([tile_i, D], dtype=torch.float32)
        for tile_c in hl.tile(C):
            acc = torch.addmm(acc, x[b, tile_i, tile_c], w[tile_c, h, :])
        q[b, h, tile_i, :] = (
            (acc + bias[h, :].to(torch.float32)).to(torch.bfloat16) * SCALE
        ).to(torch.bfloat16)
    return q


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
    D = v.size(2)
    H = BH // B
    out = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=p.device)
    for tile_bh, tile_i, tile_d in hl.tile([BH, S, D], block_size=[1, None, None]):
        bh = tile_bh.begin
        acc = hl.zeros([tile_i, tile_d], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc = torch.addmm(acc, p[bh, tile_i, tile_j], v[bh, tile_j, tile_d])
        out[bh // H, tile_i, bh % H, tile_d] = acc.to(torch.bfloat16)
    return out


def _inputs(w, bias, x, kt):
    BH, D, J = kt.shape
    H = w.size(1) // D
    B = BH // H
    return x.view(B, -1, x.size(1)), w.view(w.size(0), H, D), bias.view(H, D)


def fused(w, bias, x, kt, v, *shape_params):
    xb, w3, b2 = _inputs(w, bias, x, kt)
    out = fused_kernel(xb, w3, b2, kt, v)
    return out.view(out.size(0) * out.size(1), -1)


def unfused(w, bias, x, kt, v, *shape_params):
    xb, w3, b2 = _inputs(w, bias, x, kt)
    q = q_proj_kernel(xb, w3, b2)
    q = q.view(-1, q.size(2), q.size(3))
    out = pv_kernel(softmax_kernel(scores_kernel(q, kt)), v, xb.size(0))
    return out.view(out.size(0) * out.size(1), -1)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
