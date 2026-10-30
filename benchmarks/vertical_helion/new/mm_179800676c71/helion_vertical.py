"""Tiny MLP backward: mm -> tanh' -> weight grad + bias grad (lennard_jones, 2 occ).

Graph: d = bf16(f32(g @ W) * (1 - h * h)) [128, 16] (output), dW = d^T @ x
(K = 128 rows) and db = column sums of d.

Slice = output column (16 of them): dW and db need d[:, n] for all 128 rows
(F = 128). The whole problem fits one program, which computes d once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: d (128 x 16) on chip, dW and db reduce over its rows."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(g, w, h, x):
    M, K = g.shape
    M = hl.specialize(M)
    K = hl.specialize(K)
    N = hl.specialize(w.size(1))
    Nx = hl.specialize(x.size(1))
    d_out = torch.empty([M, N], dtype=torch.bfloat16, device=g.device)
    dw = torch.empty([N, Nx], dtype=torch.float32, device=g.device)
    db = torch.empty([N], dtype=torch.float32, device=g.device)
    for tile_z in hl.tile(1, block_size=1):
        z = hl.dot(g[:, :], w[:, :], out_dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        hf = h[:, :].to(torch.float32)
        d = (z * (1 - hf * hf)).to(torch.bfloat16)
        d_out[:, :] = d
        dw[:, :] = hl.dot(d.permute(1, 0), x[:, :], out_dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        db[:] = torch.sum(d.to(torch.float32), dim=0).to(torch.bfloat16).to(torch.float32)
    return d_out, dw, db


@helion.kernel(static_shapes=True, autotune_effort="quick")
def d_kernel(g, w, h):
    M, K = g.shape
    K = hl.specialize(K)
    N = hl.specialize(w.size(1))
    d_out = torch.empty([M, N], dtype=torch.bfloat16, device=g.device)
    for tile_m in hl.tile(M):
        z = hl.dot(g[tile_m, :], w[:, :], out_dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        hf = h[tile_m, :].to(torch.float32)
        d_out[tile_m, :] = (z * (1 - hf * hf)).to(torch.bfloat16)
    return d_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def grads_kernel(d, x):
    M, N = d.shape
    M = hl.specialize(M)
    N = hl.specialize(N)
    Nx = hl.specialize(x.size(1))
    dw = torch.empty([N, Nx], dtype=torch.float32, device=d.device)
    db = torch.empty([N], dtype=torch.float32, device=d.device)
    for tile_z in hl.tile(1, block_size=1):
        dd = d[:, :]
        dw[:, :] = hl.dot(dd.permute(1, 0), x[:, :], out_dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        db[:] = torch.sum(dd.to(torch.float32), dim=0).to(torch.bfloat16).to(torch.float32)
    return dw, db


def fused(g, w, h, x, *shape_params):
    return fused_kernel(g, w, h, x)


def unfused(g, w, h, x, *shape_params):
    d = d_kernel(g, w, h)
    return (d, *grads_kernel(d, x))


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
