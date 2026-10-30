"""Softmax -> P @ V -> out-proj + residual + NoNorm -> FFN + ReLU (MobileBERT, 24 occ).

Graph: masked softmax over 128 keys of f32 scores (all-masked rows -> 0),
bf16 probs @ V (4 heads x 32), token-major context, addmm (K = 128) + residual,
* gamma + beta (NoNorm), addmm (N = 512) + ReLU. Outputs: NoNorm result and
ReLU output.

One program per (batch, token tile): the head loop produces each head's
32 context columns and is also the K split of the out-proj, so the chain
stays on chip (F(sum) = 4 per token, context F = 128). No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (b, token tile), head loop = out-proj K split."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(m, s, v, wo, bo, res, gamma, beta, w1, b1):
    B, H, S, J = s.shape
    J = hl.specialize(J)
    D = hl.specialize(v.size(2))
    C = hl.specialize(wo.size(2))
    N = w1.size(1)
    y_out = torch.empty([B, S, C], dtype=torch.bfloat16, device=s.device)
    out = torch.empty([B, S, N], dtype=torch.bfloat16, device=s.device)
    for tile_b, tile_i in hl.tile([B, S], block_size=[1, None]):
        b = tile_b.begin
        h1 = hl.zeros([tile_i, C], dtype=torch.float32)
        for tile_h in hl.tile(H, block_size=1):
            h = tile_h.begin
            row = s[b, h, tile_i, :]
            live = torch.amax(m[b, h, tile_i, :].to(torch.float32), dim=-1, keepdim=True)
            e = torch.exp(row - torch.amax(row, dim=-1, keepdim=True))
            p = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
            p = torch.where(live > 0, p, 0.0).to(torch.bfloat16)
            c = hl.dot(p, v[b * H + h, :, :], out_dtype=torch.float32).to(torch.bfloat16)
            h1 = torch.addmm(h1, c, wo[h, :, :])
        a = (h1 + bo[:].to(torch.float32)).to(torch.bfloat16)
        a = (a + res[b, tile_i, :]).to(torch.bfloat16)
        a = (a * gamma[:]).to(torch.bfloat16)
        y = (a + beta[:]).to(torch.bfloat16)
        y_out[b, tile_i, :] = y
        for tile_n in hl.tile(N):
            z = hl.dot(y, w1[:, tile_n], out_dtype=torch.float32)
            z = (z + b1[tile_n].to(torch.float32)).to(torch.bfloat16)
            out[b, tile_i, tile_n] = torch.relu(z)
    return y_out, out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def attn_kernel(m, s, v):
    """Stage 1: masked softmax + P @ V, token-major context."""
    B, H, S, J = s.shape
    J = hl.specialize(J)
    D = hl.specialize(v.size(2))
    ctx = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=s.device)
    for tile_b, tile_h, tile_i in hl.tile([B, H, S], block_size=[1, 1, None]):
        b, h = tile_b.begin, tile_h.begin
        row = s[b, h, tile_i, :]
        live = torch.amax(m[b, h, tile_i, :].to(torch.float32), dim=-1, keepdim=True)
        e = torch.exp(row - torch.amax(row, dim=-1, keepdim=True))
        p = (e / torch.sum(e, dim=-1, keepdim=True)).to(torch.bfloat16)
        p = torch.where(live > 0, p, 0.0).to(torch.bfloat16)
        ctx[b, tile_i, h, :] = hl.dot(p, v[b * H + h, :, :], out_dtype=torch.float32).to(
            torch.bfloat16
        )
    return ctx


@helion.kernel(static_shapes=True, autotune_effort="quick")
def proj_kernel(x, w, bo, res, gamma, beta):
    """Stage 2: out-proj + residual + NoNorm."""
    M, K = x.shape
    C = w.size(1)
    y_out = torch.empty([M, C], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_c in hl.tile([M, C]):
        acc = hl.zeros([tile_m, tile_c], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_c])
        a = (acc + bo[tile_c].to(torch.float32)).to(torch.bfloat16)
        a = (a + res[tile_m, tile_c]).to(torch.bfloat16)
        a = (a * gamma[tile_c]).to(torch.bfloat16)
        y_out[tile_m, tile_c] = (a + beta[tile_c]).to(torch.bfloat16)
    return y_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ffn_kernel(y, w1, b1):
    """Stage 3: FFN up-proj + ReLU."""
    M, K = y.shape
    N = w1.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=y.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, y[tile_m, tile_k], w1[tile_k, tile_n])
        out[tile_m, tile_n] = torch.relu((acc + b1[tile_n].to(torch.float32)).to(torch.bfloat16))
    return out


def fused(m, s, v, wo, bo, res, gamma, beta, w1, b1, *shape_params):
    H = s.size(1)
    y, out = fused_kernel(m, s, v, wo.view(H, -1, wo.size(1)), bo, res, gamma, beta, w1, b1)
    return y, out.view(-1, out.size(2))


def unfused(m, s, v, wo, bo, res, gamma, beta, w1, b1, *shape_params):
    B, S, C = res.shape
    ctx = attn_kernel(m, s, v).view(B * S, -1)
    y = proj_kernel(ctx, wo, bo, res.view(B * S, C), gamma, beta)
    return y.view(B, S, C), ffn_kernel(y, w1, b1)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
