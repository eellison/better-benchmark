"""Window-attention softmax backward -> dq and dk bmms (Swin, 24 occurrences).

Graph: ds = g - p * rowsum(g) over 49 keys (f32 output), bf16 ds feeds
dk = q^T @ ds and dq = ds @ k (scaled). Outputs: ds (f32), dk^T, dq * scale.

Same structure as GPT-J's bmm_c044adeba521 (dq reads ROWS of ds, dk reads
COLUMNS, every element needs its row sum), so the slice is a whole
(window, head) - but here there are 4096 of them, each only 49 x 49.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (window, head); 4096 small slices."
SCALE = 0.1767766952966369


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(g, p, qt, kt):
    BH, S, J = g.shape
    S = hl.specialize(S)
    J = hl.specialize(J)
    D = hl.specialize(qt.size(1))
    ds_out = torch.empty([BH, S, J], dtype=torch.float32, device=g.device)
    dkt = torch.empty([BH, J, D], dtype=torch.bfloat16, device=g.device)
    dq = torch.empty([BH, S, D], dtype=torch.bfloat16, device=g.device)
    for tile_bh in hl.tile(BH, block_size=1):
        bh = tile_bh.begin
        gh = g[bh, :, :]
        ds = gh - p[bh, :, :] * torch.sum(gh, dim=-1, keepdim=True)
        ds_out[bh, :, :] = ds
        dsb = ds.to(torch.bfloat16)
        dk = hl.dot(qt[bh, :, :], dsb, out_dtype=torch.float32)
        dkt[bh, :, :] = dk.to(torch.bfloat16).permute(1, 0)
        dqv = hl.dot(dsb, kt[bh, :, :], out_dtype=torch.float32).to(torch.bfloat16)
        dq[bh, :, :] = (dqv * SCALE).to(torch.bfloat16)
    return ds_out, dkt, dq


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ds_kernel(g, p):
    BH, S, J = g.shape
    J = hl.specialize(J)
    ds_out = torch.empty([BH, S, J], dtype=torch.float32, device=g.device)
    dsb = torch.empty([BH, S, J], dtype=torch.bfloat16, device=g.device)
    for tile_bh, tile_i in hl.tile([BH, S]):
        gh = g[tile_bh, tile_i, :]
        ds = gh - p[tile_bh, tile_i, :] * torch.sum(gh, dim=-1, keepdim=True)
        ds_out[tile_bh, tile_i, :] = ds
        dsb[tile_bh, tile_i, :] = ds.to(torch.bfloat16)
    return ds_out, dsb


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dk_kernel(qt, dsb):
    BH, D, S = qt.shape
    J = hl.specialize(dsb.size(2))
    D = hl.specialize(D)
    dkt = torch.empty([BH, J, D], dtype=torch.bfloat16, device=dsb.device)
    for tile_bh in hl.tile(BH, block_size=1):
        bh = tile_bh.begin
        acc = hl.zeros([D, J], dtype=torch.float32)
        for tile_s in hl.tile(S):
            acc = torch.addmm(acc, qt[bh, :, tile_s], dsb[bh, tile_s, :])
        dkt[bh, :, :] = acc.to(torch.bfloat16).permute(1, 0)
    return dkt


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dq_kernel(dsb, kt):
    BH, S, J = dsb.shape
    S = hl.specialize(S)
    D = hl.specialize(kt.size(2))
    dq = torch.empty([BH, S, D], dtype=torch.bfloat16, device=dsb.device)
    for tile_bh in hl.tile(BH, block_size=1):
        bh = tile_bh.begin
        acc = hl.zeros([S, D], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc = torch.addmm(acc, dsb[bh, :, tile_j], kt[bh, tile_j, :])
        dq[bh, :, :] = (acc.to(torch.bfloat16) * SCALE).to(torch.bfloat16)
    return dq


def _inputs(g, p, qt, kt):
    shape = g.shape
    return g.view(-1, shape[2], shape[3]), p.view(-1, shape[2], shape[3]), qt, kt


def _outputs(ds, dkt, dq, g):
    B, H, S, J = g.shape
    return ds.view(B, H, S, J), dkt.view(B, H, J, -1), dq.view(B, H, S, -1)


def fused(g, p, qt, kt, *shape_params):
    return _outputs(*fused_kernel(*_inputs(g, p, qt, kt)), g)


def unfused(g, p, qt, kt, *shape_params):
    g3, p3, qt, kt = _inputs(g, p, qt, kt)
    ds, dsb = ds_kernel(g3, p3)
    return _outputs(ds, dk_kernel(qt, dsb), dq_kernel(dsb, kt), g)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
