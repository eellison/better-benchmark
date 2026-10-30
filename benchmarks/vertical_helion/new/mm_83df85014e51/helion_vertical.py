"""Norm backward (two row sums over 144) -> residual add -> mm (N = 288) -> SiLU bwd.

MobileViT, 2 occurrences. Graph: dx = d/144 * (a3 - sum(a0) - a2 * sum(a1))
per row of 144, h = bf16(a5 + dx) (output, also viewed and transposed),
h @ W (K = 144, N = 288), times the SiLU derivative of a7.

One program per row tile: both sums (R = 144, F = 1), dx and the 144-wide
row of h stay on chip; the mm loops over the 288 columns. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per row tile, two row sums and the 144-wide row on chip."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(a0, a1, a2, a3, a4, a5, w, a7):
    M, C = a0.shape
    C = hl.specialize(C)
    N = w.size(1)
    h_out = torch.empty([M, C], dtype=torch.bfloat16, device=a0.device)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=a0.device)
    for tile_m in hl.tile(M):
        s1 = torch.sum(a0[tile_m, :], dim=-1, keepdim=True)
        s2 = torch.sum(a1[tile_m, :], dim=-1, keepdim=True)
        d = a4[tile_m, :] / 144
        dx = d * ((a3[tile_m, :] - s1) - a2[tile_m, :] * s2)
        h = (a5[tile_m, :] + dx.to(torch.bfloat16)).to(torch.bfloat16)
        h_out[tile_m, :] = h
        for tile_n in hl.tile(N):
            z = hl.dot(h, w[:, tile_n], out_dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
            g = a7[tile_m, tile_n].to(torch.float32)
            sg = torch.sigmoid(g)
            y = (z * sg) * (g * (1 - sg) + 1)
            out[tile_m, tile_n] = y.to(torch.bfloat16)
    return h_out, out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def norm_bwd_kernel(a0, a1, a2, a3, a4, a5):
    M, C = a0.shape
    C = hl.specialize(C)
    h_out = torch.empty([M, C], dtype=torch.bfloat16, device=a0.device)
    for tile_m in hl.tile(M):
        s1 = torch.sum(a0[tile_m, :], dim=-1, keepdim=True)
        s2 = torch.sum(a1[tile_m, :], dim=-1, keepdim=True)
        d = a4[tile_m, :] / 144
        dx = d * ((a3[tile_m, :] - s1) - a2[tile_m, :] * s2)
        h_out[tile_m, :] = (a5[tile_m, :] + dx.to(torch.bfloat16)).to(torch.bfloat16)
    return h_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def mm_silu_bwd_kernel(h, w, a7):
    M, K = h.shape
    N = w.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=h.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, h[tile_m, tile_k], w[tile_k, tile_n])
        z = acc.to(torch.bfloat16).to(torch.float32)
        g = a7[tile_m, tile_n].to(torch.float32)
        sg = torch.sigmoid(g)
        out[tile_m, tile_n] = ((z * sg) * (g * (1 - sg) + 1)).to(torch.bfloat16)
    return out


def _flat(*tensors):
    return [t.reshape(-1, t.size(-1)) for t in tensors]


def _outputs(h, out, a5):
    view = h
    return h.view(a5.shape), view, view.permute(1, 0), out


def fused(a0, a1, a2, a3, a4, a5, w, a7, *shape_params):
    f = _flat(a0, a1, a2, a3, a4, a5)
    h, out = fused_kernel(*f, w, a7)
    return _outputs(h, out, a5)


def unfused(a0, a1, a2, a3, a4, a5, w, a7, *shape_params):
    h = norm_bwd_kernel(*_flat(a0, a1, a2, a3, a4, a5))
    return _outputs(h, mm_silu_bwd_kernel(h, w, a7), a5)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
