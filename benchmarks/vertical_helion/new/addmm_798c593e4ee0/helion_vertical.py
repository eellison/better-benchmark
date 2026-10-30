"""Training BatchNorm over 5 samples -> linear 64 -> 5 -> log_softmax (maml, 1 occ).

Graph: var_mean over the batch of [5, 64] (R = 5 per channel), running
stats updates (outputs), normalize + affine, addmm (K = 64, N = 5, output),
log_softmax over 5 (output).

The whole region is one slice ([5, 64] input): one program does all of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: the whole [5, 64] region is a single slice."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, rm, rv, g, b, w, bias):
    B, C = x.shape
    B = hl.specialize(B)
    C = hl.specialize(C)
    N = hl.specialize(w.size(1))
    rm_out = torch.empty([C], dtype=torch.bfloat16, device=x.device)
    rv_out = torch.empty([C], dtype=torch.bfloat16, device=x.device)
    lin = torch.empty([B, N], dtype=torch.bfloat16, device=x.device)
    lsm = torch.empty([B, N], dtype=torch.bfloat16, device=x.device)
    for tile_z in hl.tile(1, block_size=1):
        xf = x[:, :].to(torch.float32)
        mean = torch.mean(xf, dim=0)
        xc = xf - mean[None, :]
        var = torch.mean(xc * xc, dim=0)
        rm_out[:] = (mean * 0.1 + (rm[:] * 0.9).to(torch.bfloat16)).to(torch.bfloat16)
        rv_out[:] = ((var * 1.25) * 0.1 + (rv[:] * 0.9).to(torch.bfloat16)).to(torch.bfloat16)
        y = xc * torch.rsqrt(var + 1e-05)[None, :] * g[:][None, :] + b[:][None, :]
        y = y.to(torch.bfloat16)
        # N = 5 is below tl.dot's minimum: broadcast-multiply and reduce over C
        z = torch.sum(y.to(torch.float32)[:, :, None] * w[:, :].to(torch.float32)[None, :, :], dim=1)
        zb = (z + bias[:].to(torch.float32)[None, :]).to(torch.bfloat16)
        lin[:, :] = zb
        zf = zb.to(torch.float32)
        sh = zf - torch.amax(zf, dim=-1, keepdim=True)
        lsm[:, :] = (sh - torch.log(torch.sum(torch.exp(sh), dim=-1, keepdim=True))).to(torch.bfloat16)
    return rm_out, rv_out, lin, lsm


@helion.kernel(static_shapes=True, autotune_effort="quick")
def bn_kernel(x, rm, rv, g, b):
    B, C = x.shape
    B = hl.specialize(B)
    C = hl.specialize(C)
    rm_out = torch.empty([C], dtype=torch.bfloat16, device=x.device)
    rv_out = torch.empty([C], dtype=torch.bfloat16, device=x.device)
    y_out = torch.empty([B, C], dtype=torch.bfloat16, device=x.device)
    for tile_z in hl.tile(1, block_size=1):
        xf = x[:, :].to(torch.float32)
        mean = torch.mean(xf, dim=0)
        xc = xf - mean[None, :]
        var = torch.mean(xc * xc, dim=0)
        rm_out[:] = (mean * 0.1 + (rm[:] * 0.9).to(torch.bfloat16)).to(torch.bfloat16)
        rv_out[:] = ((var * 1.25) * 0.1 + (rv[:] * 0.9).to(torch.bfloat16)).to(torch.bfloat16)
        y = xc * torch.rsqrt(var + 1e-05)[None, :] * g[:][None, :] + b[:][None, :]
        y_out[:, :] = y.to(torch.bfloat16)
    return rm_out, rv_out, y_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def linear_kernel(y, w, bias):
    B, C = y.shape
    B = hl.specialize(B)
    C = hl.specialize(C)
    N = hl.specialize(w.size(1))
    lin = torch.empty([B, N], dtype=torch.bfloat16, device=y.device)
    for tile_z in hl.tile(1, block_size=1):
        z = torch.sum(y[:, :].to(torch.float32)[:, :, None] * w[:, :].to(torch.float32)[None, :, :], dim=1)
        lin[:, :] = (z + bias[:].to(torch.float32)[None, :]).to(torch.bfloat16)
    return lin


@helion.kernel(static_shapes=True, autotune_effort="quick")
def log_softmax_kernel(z):
    B, N = z.shape
    B = hl.specialize(B)
    N = hl.specialize(N)
    out = torch.empty([B, N], dtype=torch.bfloat16, device=z.device)
    for tile_z in hl.tile(1, block_size=1):
        zf = z[:, :].to(torch.float32)
        sh = zf - torch.amax(zf, dim=-1, keepdim=True)
        out[:, :] = (sh - torch.log(torch.sum(torch.exp(sh), dim=-1, keepdim=True))).to(torch.bfloat16)
    return out


def fused(x, rm, rv, g, b, w, bias, *shape_params):
    return fused_kernel(x.view(x.size(0), -1), rm, rv, g, b, w, bias)


def unfused(x, rm, rv, g, b, w, bias, *shape_params):
    rm_out, rv_out, y = bn_kernel(x.view(x.size(0), -1), rm, rv, g, b)
    lin = linear_kernel(y, w, bias)
    return rm_out, rv_out, lin, log_softmax_kernel(lin)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
