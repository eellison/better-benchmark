"""Out-proj + residual -> LayerNorm(192) -> fc1 + GELU (DeiT-tiny, 13 occurrences).

Graph: h = residual + (a @ W1 + b1) (K = 192, output `add`), LN over 192,
fc1 = LN(h) @ W2 + b2 (N = 768), exact GELU. Outputs: h and GELU(fc1).

One program per row tile: the first MM's full 192-wide rows (F = 192) stay on
chip for the LN, then the second MM loops over its 768 columns. No
recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per row tile keeps the 192-wide MM rows for the LN."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(a, w1, b1, res, g, beta, w2, b2):
    M, K = a.shape
    C = hl.specialize(w1.size(1))
    N = w2.size(1)
    h_out = torch.empty([M, C], dtype=torch.bfloat16, device=a.device)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=a.device)
    for tile_m in hl.tile(M):
        acc = hl.zeros([tile_m, C], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, a[tile_m, tile_k], w1[tile_k, :])
        h1 = (acc + b1[:].to(torch.float32)).to(torch.bfloat16)
        h = (res[tile_m, :] + h1).to(torch.bfloat16)
        h_out[tile_m, :] = h
        hf = h.to(torch.float32)
        xc = hf - torch.mean(hf, dim=-1, keepdim=True)
        var = torch.mean(xc * xc, dim=-1, keepdim=True)
        y = xc * torch.rsqrt(var + 1e-6) * g[:].to(torch.float32) + beta[:].to(torch.float32)
        y = y.to(torch.bfloat16)
        for tile_n in hl.tile(N):
            z = hl.dot(y, w2[:, tile_n], out_dtype=torch.float32)
            z = (z + b2[tile_n].to(torch.float32)).to(torch.bfloat16).to(torch.float32)
            z = (z * 0.5) * (torch.erf(z * 0.7071067811865476) + 1)
            out[tile_m, tile_n] = z.to(torch.bfloat16)
    return h_out, out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def proj_residual_kernel(a, w1, b1, res):
    M, K = a.shape
    C = w1.size(1)
    h_out = torch.empty([M, C], dtype=torch.bfloat16, device=a.device)
    for tile_m, tile_c in hl.tile([M, C]):
        acc = hl.zeros([tile_m, tile_c], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, a[tile_m, tile_k], w1[tile_k, tile_c])
        h1 = (acc + b1[tile_c].to(torch.float32)).to(torch.bfloat16)
        h_out[tile_m, tile_c] = (res[tile_m, tile_c] + h1).to(torch.bfloat16)
    return h_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ln_kernel(h, g, beta):
    M, C = h.shape
    C = hl.specialize(C)
    y_out = torch.empty([M, C], dtype=torch.bfloat16, device=h.device)
    for tile_m in hl.tile(M):
        hf = h[tile_m, :].to(torch.float32)
        xc = hf - torch.mean(hf, dim=-1, keepdim=True)
        var = torch.mean(xc * xc, dim=-1, keepdim=True)
        y = xc * torch.rsqrt(var + 1e-6) * g[:].to(torch.float32) + beta[:].to(torch.float32)
        y_out[tile_m, :] = y.to(torch.bfloat16)
    return y_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fc1_gelu_kernel(y, w2, b2):
    M, K = y.shape
    N = w2.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=y.device)
    for tile_m, tile_n in hl.tile([M, N]):
        z = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            z = torch.addmm(z, y[tile_m, tile_k], w2[tile_k, tile_n])
        z = (z + b2[tile_n].to(torch.float32)).to(torch.bfloat16).to(torch.float32)
        z = (z * 0.5) * (torch.erf(z * 0.7071067811865476) + 1)
        out[tile_m, tile_n] = z.to(torch.bfloat16)
    return out


def _inputs(a, w1, b1, res, g, beta, w2, b2):
    C = w1.size(0)
    return a.reshape(-1, C), w1, b1, res.view(-1, res.size(2)), g, beta, w2, b2


def fused(*inputs):
    a, w1, b1, res, g, beta, w2, b2 = _inputs(*inputs[:8])
    h, out = fused_kernel(a, w1, b1, res, g, beta, w2, b2)
    return h.view(inputs[3].shape), out


def unfused(*inputs):
    a, w1, b1, res, g, beta, w2, b2 = _inputs(*inputs[:8])
    h = proj_residual_kernel(a, w1, b1, res)
    out = fc1_gelu_kernel(ln_kernel(h, g, beta), w2, b2)
    return h.view(inputs[3].shape), out


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
