"""LayerNorm(192) -> qkv linear (DeiT-tiny, 30 occurrences in 12 repros).

Graph: var_mean over 192 channels of f32 x [128, 197, 192], affine, bf16,
addmm with W [192, 576] + bias; output is a view of the [25216, 576] result.

One program per row tile normalizes full 192-wide rows (R = 192, F = 1) and
loops over the 576 output columns reusing them. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per row tile, LN on full rows, loop over N."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, g, b, w, bias):
    M, C = x.shape
    C = hl.specialize(C)
    N = w.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
    for tile_m in hl.tile(M):
        row = x[tile_m, :]
        xc = row - torch.mean(row, dim=-1, keepdim=True)
        var = torch.mean(xc * xc, dim=-1, keepdim=True)
        y = xc * torch.rsqrt(var + 1e-6) * g[:].to(torch.float32) + b[:].to(torch.float32)
        y = y.to(torch.bfloat16)
        for tile_n in hl.tile(N):
            acc = hl.dot(y, w[:, tile_n], out_dtype=torch.float32)
            out[tile_m, tile_n] = (acc + bias[tile_n].to(torch.float32)).to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_ktiled_kernel(x, g, b, w, bias):
    """Same single kernel, but K = 192 is tiled (no padding to 256): LN
    statistics in a first pass over the row, then every N tile re-reads the
    row chunk by chunk and normalizes it on the fly (x re-read from L2)."""
    M, C = x.shape
    N = w.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=x.device)
    for tile_m in hl.tile(M):
        s1 = hl.zeros([tile_m], dtype=torch.float32)
        s2 = hl.zeros([tile_m], dtype=torch.float32)
        for tile_c in hl.tile(C):
            v = x[tile_m, tile_c]
            s1 = s1 + torch.sum(v, dim=-1)
            s2 = s2 + torch.sum(v * v, dim=-1)
        mean = s1 / C
        rstd = torch.rsqrt(s2 / C - mean * mean + 1e-6)
        for tile_n in hl.tile(N):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_c in hl.tile(C):
                v = (x[tile_m, tile_c] - mean[:, None]) * rstd[:, None]
                y = v * g[tile_c].to(torch.float32) + b[tile_c].to(torch.float32)
                acc = torch.addmm(acc, y.to(torch.bfloat16), w[tile_c, tile_n])
            out[tile_m, tile_n] = (acc + bias[tile_n].to(torch.float32)).to(torch.bfloat16)
    return out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ln_kernel(x, g, b):
    M, C = x.shape
    C = hl.specialize(C)
    y_out = torch.empty([M, C], dtype=torch.bfloat16, device=x.device)
    for tile_m in hl.tile(M):
        row = x[tile_m, :]
        xc = row - torch.mean(row, dim=-1, keepdim=True)
        var = torch.mean(xc * xc, dim=-1, keepdim=True)
        y = xc * torch.rsqrt(var + 1e-6) * g[:].to(torch.float32) + b[:].to(torch.float32)
        y_out[tile_m, :] = y.to(torch.bfloat16)
    return y_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def addmm_kernel(y, w, bias):
    M, K = y.shape
    N = w.size(1)
    out = torch.empty([M, N], dtype=torch.bfloat16, device=y.device)
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = torch.addmm(acc, y[tile_m, tile_k], w[tile_k, tile_n])
        out[tile_m, tile_n] = (acc + bias[tile_n].to(torch.float32)).to(torch.bfloat16)
    return out


def _view(out, x):
    B, T, _ = x.shape
    return out.view(B, T, 3, 3, -1).permute(2, 0, 3, 1, 4)


def fused(x, g, b, w, bias, *shape_params):
    return _view(fused_kernel(x.view(-1, x.size(2)), g, b, w, bias), x)


def fused_ktiled(x, g, b, w, bias, *shape_params):
    return _view(fused_ktiled_kernel(x.view(-1, x.size(2)), g, b, w, bias), x)


def unfused(x, g, b, w, bias, *shape_params):
    y = ln_kernel(x.view(-1, x.size(2)), g, b)
    return _view(addmm_kernel(y, w, bias), x)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES, variants={"fused_ktiled": fused_ktiled})
