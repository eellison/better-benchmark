"""Context bmm -> out-proj + residual + NoNorm -> FFN up-proj (MobileBERT, 24 occ).

Graph: ctx[bh, d, s] = a[bh] @ P[bh] (K = 128 keys), * scale, token-major
[32768, 128] (4 heads x 32); h = ctx @ W1 (K = 128); add = f32(h) + res;
y = bf16(add * gamma); out = y @ W2 (N = 512). Outputs: ctx, add, y, out.

One program per (batch, token tile): each token's 128 context values
(F = 128) come from 4 per-head products that are also the K split of the
out-proj, so the whole chain stays on chip. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (batch, token tile), head loop = out-proj K split."
SCALE = 0.4204482076268573


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(at, pm, w1h, res, gamma, w2):
    BH, T, D = at.shape
    B, S, C = res.shape
    D = hl.specialize(D)
    C = hl.specialize(C)
    H = BH // B
    N = w2.size(1)
    ctx = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=at.device)
    add = torch.empty([B, S, C], dtype=torch.float32, device=at.device)
    y_out = torch.empty([B, S, C], dtype=torch.bfloat16, device=at.device)
    out = torch.empty([B, S, N], dtype=torch.bfloat16, device=at.device)
    for tile_b, tile_s in hl.tile([B, S], block_size=[1, None]):
        b = tile_b.begin
        h1 = hl.zeros([tile_s, C], dtype=torch.float32)
        for tile_h in hl.tile(H, block_size=1):
            h = tile_h.begin
            c = hl.dot(pm[b * H + h, :, tile_s].permute(1, 0), at[b * H + h, :, :], out_dtype=torch.float32)
            c = (c.to(torch.bfloat16) * SCALE).to(torch.bfloat16)
            ctx[b, tile_s, h, :] = c
            h1 = torch.addmm(h1, c, w1h[h, :, :])
        a = h1.to(torch.bfloat16).to(torch.float32) + res[b, tile_s, :]
        add[b, tile_s, :] = a
        y = (a * gamma[:]).to(torch.bfloat16)
        y_out[b, tile_s, :] = y
        for tile_n in hl.tile(N):
            out[b, tile_s, tile_n] = hl.dot(y, w2[:, tile_n], out_dtype=torch.float32).to(torch.bfloat16)
    return ctx, add, y_out, out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ctx_kernel(at, pm, B: hl.constexpr):
    BH, T, D = at.shape
    S = pm.size(2)
    D = hl.specialize(D)
    H = BH // B
    ctx = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=at.device)
    for tile_bh, tile_s in hl.tile([BH, S], block_size=[1, None]):
        bh = tile_bh.begin
        acc = hl.zeros([tile_s, D], dtype=torch.float32)
        for tile_t in hl.tile(T):
            acc = torch.addmm(acc, pm[bh, tile_t, tile_s].permute(1, 0), at[bh, tile_t, :])
        ctx[bh // H, tile_s, bh % H, :] = (acc.to(torch.bfloat16) * SCALE).to(torch.bfloat16)
    return ctx


@helion.kernel(static_shapes=True, autotune_effort="quick")
def proj_kernel(x, w1, res, gamma):
    M, K = x.shape
    C = w1.size(1)
    add = torch.empty([M, C], dtype=torch.float32, device=x.device)
    y_out = torch.empty([M, C], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_c in hl.tile([M, C]):
        acc = hl.zeros([tile_m, tile_c], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[tile_m, tile_k], w1[tile_k, tile_c])
        a = acc.to(torch.bfloat16).to(torch.float32) + res[tile_m, tile_c]
        add[tile_m, tile_c] = a
        y_out[tile_m, tile_c] = (a * gamma[tile_c]).to(torch.bfloat16)
    return add, y_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def mm_kernel(x, w):
    M, K = x.shape
    N = w.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, x[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = acc.to(torch.bfloat16)
    return out


def fused(a, pm, w1, res, gamma, w2, *shape_params):
    H = a.size(0) // res.size(0)
    at = a.permute(0, 2, 1)  # [BH, T, D], contiguous
    w1h = w1.view(H, -1, w1.size(1))
    ctx, add, y, out = fused_kernel(at, pm, w1h, res, gamma, w2)
    M = res.size(0) * res.size(1)
    return ctx.view(M, -1), add, y.view(M, -1), out.view(M, -1)


def unfused(a, pm, w1, res, gamma, w2, *shape_params):
    B = res.size(0)
    M = B * res.size(1)
    ctx = ctx_kernel(a.permute(0, 2, 1), pm, B).view(M, -1)
    add, y = proj_kernel(ctx, w1, res.view(M, -1), gamma)
    return ctx, add.view(res.shape), y, mm_kernel(y, w2)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
