"""Softmax backward -> causal mask -> dk and dq bmms (GPT-Neo, 44 occurrences).

Graph: ds = g - p * rowsum(g) over 128 keys, bf16, where(causal mask, ds,
fill), dk^T = q^T @ ds, dq = ds @ k (head_dim 128), both token-major
[4096, 2048] (plus transposed views).

Same structure as bmm_c044adeba521 (dq reads rows of ds, dk reads columns):
the slice is a whole (batch, head) - here 512 of them instead of 16.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (b, h): 512 slices of 128 x 128."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(g, p, mask, fill, qt, kt, H: hl.constexpr):
    BH, S, J = g.shape
    S = hl.specialize(S)
    J = hl.specialize(J)
    D = qt.size(1)
    B = BH // H
    dk_out = torch.empty([B, J, H, D], dtype=torch.bfloat16, device=g.device)
    dq_out = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=g.device)
    for tile_bh in hl.tile(BH, block_size=1):
        bh = tile_bh.begin
        b, h = bh // H, bh % H
        gh = g[bh, :, :]
        ds = (gh - p[bh, :, :] * torch.sum(gh, dim=-1, keepdim=True)).to(torch.bfloat16)
        ds = torch.where(mask[:, :], ds, fill[0]).to(torch.bfloat16)
        for tile_d in hl.tile(D):
            dkt = hl.dot(qt[bh, tile_d, :], ds, out_dtype=torch.float32).to(torch.bfloat16)
            dk_out[b, :, h, tile_d] = dkt.permute(1, 0)
            dq = hl.dot(ds, kt[bh, :, tile_d], out_dtype=torch.float32).to(torch.bfloat16)
            dq_out[b, :, h, tile_d] = dq
    return dk_out, dq_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ds_kernel(g, p, mask, fill):
    BH, S, J = g.shape
    J = hl.specialize(J)
    ds_out = torch.empty([BH, S, J], dtype=torch.bfloat16, device=g.device)
    for tile_bh, tile_i in hl.tile([BH, S]):
        gh = g[tile_bh, tile_i, :]
        ds = (gh - p[tile_bh, tile_i, :] * torch.sum(gh, dim=-1, keepdim=True)).to(torch.bfloat16)
        ds_out[tile_bh, tile_i, :] = torch.where(mask[tile_i, :], ds, fill[0]).to(torch.bfloat16)
    return ds_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dk_kernel(qt, ds, H: hl.constexpr):
    BH, S, J = ds.shape
    D = qt.size(1)
    B = BH // H
    dk_out = torch.empty([B, J, H, D], dtype=torch.bfloat16, device=ds.device)
    for tile_bh, tile_d, tile_j in hl.tile([BH, D, J], block_size=[1, None, None]):
        bh = tile_bh.begin
        acc = hl.zeros([tile_d, tile_j], dtype=torch.float32)
        for tile_i in hl.tile(S):
            acc = torch.addmm(acc, qt[bh, tile_d, tile_i], ds[bh, tile_i, tile_j])
        dk_out[bh // H, tile_j, bh % H, tile_d] = acc.to(torch.bfloat16).permute(1, 0)
    return dk_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dq_kernel(ds, kt, H: hl.constexpr):
    BH, S, J = ds.shape
    D = kt.size(2)
    B = BH // H
    dq_out = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=ds.device)
    for tile_bh, tile_i, tile_d in hl.tile([BH, S, D], block_size=[1, None, None]):
        bh = tile_bh.begin
        acc = hl.zeros([tile_i, tile_d], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc = torch.addmm(acc, ds[bh, tile_i, tile_j], kt[bh, tile_j, tile_d])
        dq_out[bh // H, tile_i, bh % H, tile_d] = acc.to(torch.bfloat16)
    return dq_out


def _inputs(g, p, mask, fill, qt, kt):
    B, H, S, J = g.shape
    return g.view(-1, S, J), p.view(-1, S, J), mask[0, 0, :S, :J], fill.view(1), qt, kt, H


def _outputs(dk, dq):
    B, S, H, D = dq.shape
    dk2 = dk.view(B * S, H * D)
    dq2 = dq.view(B * S, H * D)
    return dk2, dk2.permute(1, 0), dq2, dq2.permute(1, 0)


def fused(g, p, mask, fill, qt, kt, *shape_params):
    return _outputs(*fused_kernel(*_inputs(g, p, mask, fill, qt, kt)))


def unfused(g, p, mask, fill, qt, kt, *shape_params):
    g3, p3, m2, f, qt, kt, H = _inputs(g, p, mask, fill, qt, kt)
    ds = ds_kernel(g3, p3, m2, f)
    return _outputs(dk_kernel(qt, ds, H), dq_kernel(ds, kt, H))


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
