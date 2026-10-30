"""Causal-masked attention forward with 128 keys (GPT-Neo, 48 occurrences).

Graph: scores = q @ k^T (fp32, K=128) -> where(causal mask) -> + bias ->
softmax over 128 keys -> bf16 probs @ V (K=128) -> token-major [4096, 2048].

One program per (batch*head, query tile): scores [rows, 128] (F = 128 per
row), softmax and P @ V stay on chip. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (bh, row tile), scores row of 128 kept on chip."
NEG = -3.4028234663852886e38


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(mask, q, kt, bias, v):
    BH, S, D = q.shape
    J = hl.specialize(kt.size(2))
    Dv = hl.specialize(v.size(2))
    B = bias.size(0)
    H = BH // B
    out = torch.empty([B, S, H, Dv], dtype=torch.bfloat16, device=q.device)
    for tile_bh, tile_i in hl.tile([BH, S], block_size=[1, None]):
        bh = tile_bh.begin
        b = bh // H
        s = hl.zeros([tile_i, J], dtype=torch.float32)
        for tile_d in hl.tile(D):
            s = torch.addmm(s, q[bh, tile_i, tile_d], kt[bh, tile_d, :])
        s = torch.where(mask[tile_i, :], s, NEG) + bias[b, tile_i, :].to(torch.float32)
        e = torch.exp(s - torch.amax(s, dim=-1, keepdim=True))
        p = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
        o = hl.dot(p, v[bh, :, :], out_dtype=torch.float32)
        out[b, tile_i, bh % H, :] = o.to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def scores_kernel(mask, q, kt, bias):
    """Stage 1: q @ k^T with the mask/bias epilogue (fp32 scores)."""
    BH, S, D = q.shape
    J = kt.size(2)
    B = bias.size(0)
    H = BH // B
    s_out = torch.empty([BH, S, J], dtype=torch.float32, device=q.device)
    for tile_bh, tile_i, tile_j in hl.tile([BH, S, J], block_size=[1, None, None]):
        bh = tile_bh.begin
        s = hl.zeros([tile_i, tile_j], dtype=torch.float32)
        for tile_d in hl.tile(D):
            s = torch.addmm(s, q[bh, tile_i, tile_d], kt[bh, tile_d, tile_j])
        s = torch.where(mask[tile_i, tile_j], s, NEG)
        s_out[bh, tile_i, tile_j] = s + bias[bh // H, tile_i, tile_j].to(torch.float32)
    return s_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def softmax_kernel(s):
    """Stage 2: amax and sum over the 128 keys (one row pass)."""
    BH, S, J = s.shape
    J = hl.specialize(J)
    p_out = torch.empty([BH, S, J], dtype=torch.bfloat16, device=s.device)
    for tile_bh, tile_i in hl.tile([BH, S]):
        row = s[tile_bh, tile_i, :]
        e = torch.exp(row - torch.amax(row, dim=-1, keepdim=True))
        p_out[tile_bh, tile_i, :] = (e / torch.sum(e, dim=-1, keepdim=True)).to(
            torch.bfloat16
        )
    return p_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def pv_kernel(p, v, B: hl.constexpr):
    """Stage 3: P @ V written token-major."""
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


def _views(mask, bias, kt):
    S, J = bias.shape[-2], kt.shape[-1]
    return mask[0, 0, :S, :J], bias[:, 0]


def fused(mask, q, kt, bias, v, *shape_params):
    mask2, bias3 = _views(mask, bias, kt)
    out = fused_kernel(mask2, q, kt, bias3, v)
    return out.view(out.size(0) * out.size(1), -1)


def unfused(mask, q, kt, bias, v, *shape_params):
    mask2, bias3 = _views(mask, bias, kt)
    p = softmax_kernel(scores_kernel(mask2, q, kt, bias3))
    out = pv_kernel(p, v, bias.size(0))
    return out.view(out.size(0) * out.size(1), -1)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
