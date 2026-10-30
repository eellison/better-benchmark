"""Softmax backward -> bias-grad accumulation, dk and dq bmms (MT5, 14 occurrences).

Graph: ds = g - p * rowsum(g) over 128 keys, bf16; add = acc + f32(ds)
(output); dk^T = q^T @ ds, dq = ds @ k (head_dim 64), token-major [4096, 384].

Same structure as bmm_c044adeba521: the slice is a whole (batch, head),
here 192 of them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (b, h): 192 slices of 128 x 128."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(g, p, acc_in, qt, kt, H: hl.constexpr):
    BH, S, J = g.shape
    S = hl.specialize(S)
    J = hl.specialize(J)
    D = hl.specialize(qt.size(1))
    B = BH // H
    add = torch.empty([BH, S, J], dtype=torch.float32, device=g.device)
    dk_out = torch.empty([B, J, H, D], dtype=torch.bfloat16, device=g.device)
    dq_out = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=g.device)
    for tile_bh in hl.tile(BH, block_size=1):
        bh = tile_bh.begin
        b, h = bh // H, bh % H
        gh = g[bh, :, :]
        ds = (gh - p[bh, :, :] * torch.sum(gh, dim=-1, keepdim=True)).to(torch.bfloat16)
        add[bh, :, :] = acc_in[bh, :, :] + ds.to(torch.float32)
        dkt = hl.dot(qt[bh, :, :], ds, out_dtype=torch.float32).to(torch.bfloat16)
        dk_out[b, :, h, :] = dkt.permute(1, 0)
        dq_out[b, :, h, :] = hl.dot(ds, kt[bh, :, :], out_dtype=torch.float32).to(torch.bfloat16)
    return add, dk_out, dq_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ds_kernel(g, p, acc_in):
    BH, S, J = g.shape
    J = hl.specialize(J)
    add = torch.empty([BH, S, J], dtype=torch.float32, device=g.device)
    ds_out = torch.empty([BH, S, J], dtype=torch.bfloat16, device=g.device)
    for tile_bh, tile_i in hl.tile([BH, S]):
        gh = g[tile_bh, tile_i, :]
        ds = (gh - p[tile_bh, tile_i, :] * torch.sum(gh, dim=-1, keepdim=True)).to(torch.bfloat16)
        ds_out[tile_bh, tile_i, :] = ds
        add[tile_bh, tile_i, :] = acc_in[tile_bh, tile_i, :] + ds.to(torch.float32)
    return add, ds_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dk_kernel(qt, ds, H: hl.constexpr):
    BH, S, J = ds.shape
    D = hl.specialize(qt.size(1))
    B = BH // H
    dk_out = torch.empty([B, J, H, D], dtype=torch.bfloat16, device=ds.device)
    for tile_bh, tile_j in hl.tile([BH, J], block_size=[1, None]):
        bh = tile_bh.begin
        acc = hl.zeros([D, tile_j], dtype=torch.float32)
        for tile_i in hl.tile(S):
            acc = torch.addmm(acc, qt[bh, :, tile_i], ds[bh, tile_i, tile_j])
        dk_out[bh // H, tile_j, bh % H, :] = acc.to(torch.bfloat16).permute(1, 0)
    return dk_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dq_kernel(ds, kt, H: hl.constexpr):
    BH, S, J = ds.shape
    D = hl.specialize(kt.size(2))
    B = BH // H
    dq_out = torch.empty([B, S, H, D], dtype=torch.bfloat16, device=ds.device)
    for tile_bh, tile_i in hl.tile([BH, S], block_size=[1, None]):
        bh = tile_bh.begin
        acc = hl.zeros([tile_i, D], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc = torch.addmm(acc, ds[bh, tile_i, tile_j], kt[bh, tile_j, :])
        dq_out[bh // H, tile_i, bh % H, :] = acc.to(torch.bfloat16)
    return dq_out


def _inputs(g, p, acc_in, qt, kt):
    B, H, S, J = g.shape
    return g.view(-1, S, J), p.view(-1, S, J), acc_in.view(-1, S, J), qt, kt, H


def _outputs(add, dk, dq, g):
    B, S, H, D = dq.shape
    dq2 = dq.view(B * S, H * D)
    return add.view(g.shape), dk.view(B * S, H * D), dq2, dq2.permute(1, 0)


def fused(g, p, acc_in, qt, kt, *shape_params):
    return _outputs(*fused_kernel(*_inputs(g, p, acc_in, qt, kt)), g)


def unfused(g, p, acc_in, qt, kt, *shape_params):
    g3, p3, a3, qt, kt, H = _inputs(g, p, acc_in, qt, kt)
    add, ds = ds_kernel(g3, p3, a3)
    return _outputs(add, dk_kernel(qt, ds, H), dq_kernel(ds, kt, H), g)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
