"""P @ V -> token-major attention output -> weight gradient (GPT-J, 56 occurrences).

Graph: attn = bmm(P, V) ([16, 128, 256] -> token-major [128, 4096]),
dW = attn^T @ g (K = 128 tokens) in bf16, then converted to f32 [4096, 4096].
Outputs: attn and dW.

Slice analysis: dW[n, m] needs attn[:, n] for all 128 tokens; different m
reuse it. One program per (head, d tile) computes its attn columns once
(K = 128 keys) and writes dW rows for every m. No recomputation; the number
of programs is 16 * 256 / BLOCK_D.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (head, d tile) keeps attn[:, d tile] (128 tokens)."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(p, v, g):
    H, S, J = p.shape
    S = hl.specialize(S)
    Dh = v.size(2)
    M = g.size(1)
    attn = torch.empty([S, H, Dh], dtype=torch.bfloat16, device=p.device)
    dw = torch.empty([H, Dh, M], dtype=torch.float32, device=p.device)
    for tile_h, tile_d in hl.tile([H, Dh], block_size=[1, None]):
        h = tile_h.begin
        a = hl.zeros([S, tile_d], dtype=torch.float32)
        for tile_j in hl.tile(J):
            a = torch.addmm(a, p[h, :, tile_j], v[h, tile_j, tile_d])
        ab = a.to(torch.bfloat16)
        attn[:, h, tile_d] = ab
        abt = ab.permute(1, 0)
        for tile_m in hl.tile(M):
            w = hl.dot(abt, g[:, tile_m], out_dtype=torch.float32)
            dw[h, tile_d, tile_m] = w.to(torch.bfloat16).to(torch.float32)
    return attn, dw


@helion.kernel(static_shapes=True, autotune_effort="quick")
def pv_kernel(p, v):
    H, S, J = p.shape
    Dh = v.size(2)
    attn = torch.empty([S, H, Dh], dtype=torch.bfloat16, device=p.device)
    for tile_h, tile_i, tile_d in hl.tile([H, S, Dh], block_size=[1, None, None]):
        h = tile_h.begin
        acc = hl.zeros([tile_i, tile_d], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc = torch.addmm(acc, p[h, tile_i, tile_j], v[h, tile_j, tile_d])
        attn[tile_i, h, tile_d] = acc.to(torch.bfloat16)
    return attn


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dw_kernel(at, g):
    N, T = at.size()
    M = g.size(1)
    dw = torch.empty([N, M], dtype=torch.float32, device=g.device)
    for tile_n, tile_m in hl.tile([N, M]):
        acc = hl.zeros([tile_n, tile_m], dtype=torch.float32)
        for tile_t in hl.tile(T):
            acc = torch.addmm(acc, at[tile_n, tile_t], g[tile_t, tile_m])
        dw[tile_n, tile_m] = acc.to(torch.bfloat16).to(torch.float32)
    return dw


def fused(p, v, g, *shape_params):
    attn, dw = fused_kernel(p, v, g)
    S = attn.size(0)
    return attn.view(S, -1), dw.view(-1, dw.size(2))


def unfused(p, v, g, *shape_params):
    attn = pv_kernel(p, v).view(p.size(1), -1)
    return attn, dw_kernel(attn.permute(1, 0), g)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
