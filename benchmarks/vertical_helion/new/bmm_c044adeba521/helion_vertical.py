"""Softmax backward -> dq and dk bmms -> RoPE epilogue (GPT-J, 56 occurrences).

Graph: ds = g - p * rowsum(g) (f32), dsb = bf16(ds) / 16,
dk = q^T @ dsb ([16, 256, 128]), dq = dsb @ k ([16, 128, 256]);
outputs: dk^T (token-major view), dq[..., 64:], dq[..., :64] * rope_a,
dq[..., :64] * rope_b.

Slice analysis: dq needs ROWS of ds, dk needs COLUMNS of ds, and every ds
element needs its row sum. The only independent slice is the whole head
(the extractor reported F(sum) = 128, slice = head). One kernel without
recomputation therefore runs one program per head: 16 programs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = (
    "One kernel only with one program per head (16 programs): dq reads rows "
    "and dk reads columns of ds, which needs every row sum of the head."
)


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(g, p, qt, k, k2, ra, rb):
    H, S, J = g.shape
    S = hl.specialize(S)
    J = hl.specialize(J)
    D = qt.size(1)
    R2 = hl.specialize(ra.size(1))
    dq = torch.empty([S, H, D], dtype=torch.bfloat16, device=g.device)
    dkt = torch.empty([S, H, D], dtype=torch.bfloat16, device=g.device)
    mul = torch.empty([S, H, R2, 2], dtype=torch.bfloat16, device=g.device)
    mul_1 = torch.empty([S, H, R2, 2], dtype=torch.bfloat16, device=g.device)
    for tile_h in hl.tile(H, block_size=1):
        h = tile_h.begin
        gh = g[h, :, :]
        ds = gh - p[h, :, :] * torch.sum(gh, dim=-1, keepdim=True)
        dsb = ds.to(torch.bfloat16) / 16.0
        for tile_d in hl.tile(D):
            dq_t = hl.dot(dsb, k[h, :, tile_d], out_dtype=torch.float32).to(torch.bfloat16)
            dq[:, h, tile_d] = dq_t
            dk_t = hl.dot(qt[h, tile_d, :], dsb, out_dtype=torch.float32).to(torch.bfloat16)
            dkt[:, h, tile_d] = dk_t.permute(1, 0)
        # rope: the first 2 * R2 columns of dq times tables repeated twice
        dq_r = hl.dot(dsb, k2[h, :, :, 0], out_dtype=torch.float32).to(torch.bfloat16)
        dq_s = hl.dot(dsb, k2[h, :, :, 1], out_dtype=torch.float32).to(torch.bfloat16)
        mul[:, h, :, 0] = dq_r * ra[:, :]
        mul[:, h, :, 1] = dq_s * ra[:, :]
        mul_1[:, h, :, 0] = dq_r * rb[:, :]
        mul_1[:, h, :, 1] = dq_s * rb[:, :]
    return dq, dkt, mul, mul_1


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ds_kernel(g, p):
    """Stage 1: row sum and ds (bf16, scaled)."""
    H, S, J = g.shape
    J = hl.specialize(J)
    dsb = torch.empty([H, S, J], dtype=torch.bfloat16, device=g.device)
    for tile_h, tile_i in hl.tile([H, S]):
        gh = g[tile_h, tile_i, :]
        ds = gh - p[tile_h, tile_i, :] * torch.sum(gh, dim=-1, keepdim=True)
        dsb[tile_h, tile_i, :] = ds.to(torch.bfloat16) / 16.0
    return dsb


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dq_kernel(dsb, k, k2, ra, rb):
    """Stage 2: dq = ds @ k, token-major, with the RoPE products."""
    H, S, J = dsb.shape
    D = k.size(2)
    R2 = ra.size(1)
    dq = torch.empty([S, H, D], dtype=torch.bfloat16, device=dsb.device)
    mul = torch.empty([S, H, R2, 2], dtype=torch.bfloat16, device=dsb.device)
    mul_1 = torch.empty([S, H, R2, 2], dtype=torch.bfloat16, device=dsb.device)
    for tile_h, tile_i, tile_d in hl.tile([H, S, D], block_size=[1, None, None]):
        h = tile_h.begin
        acc = hl.zeros([tile_i, tile_d], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc = torch.addmm(acc, dsb[h, tile_i, tile_j], k[h, tile_j, tile_d])
        dq[tile_i, h, tile_d] = acc.to(torch.bfloat16)
    for tile_h, tile_i, tile_r in hl.tile([H, S, R2], block_size=[1, None, None]):
        h = tile_h.begin
        acc0 = hl.zeros([tile_i, tile_r], dtype=torch.float32)
        acc1 = hl.zeros([tile_i, tile_r], dtype=torch.float32)
        for tile_j in hl.tile(J):
            acc0 = torch.addmm(acc0, dsb[h, tile_i, tile_j], k2[h, tile_j, tile_r, 0])
            acc1 = torch.addmm(acc1, dsb[h, tile_i, tile_j], k2[h, tile_j, tile_r, 1])
        a0, a1 = acc0.to(torch.bfloat16), acc1.to(torch.bfloat16)
        mul[tile_i, h, tile_r, 0] = a0 * ra[tile_i, tile_r]
        mul[tile_i, h, tile_r, 1] = a1 * ra[tile_i, tile_r]
        mul_1[tile_i, h, tile_r, 0] = a0 * rb[tile_i, tile_r]
        mul_1[tile_i, h, tile_r, 1] = a1 * rb[tile_i, tile_r]
    return dq, mul, mul_1


@helion.kernel(static_shapes=True, autotune_effort="quick")
def dk_kernel(qt, dsb):
    """Stage 3: dk = q^T @ ds, stored transposed (token-major)."""
    H, S, J = dsb.shape
    D = qt.size(1)
    dkt = torch.empty([S, H, D], dtype=torch.bfloat16, device=dsb.device)
    for tile_h, tile_d, tile_j in hl.tile([H, D, J], block_size=[1, None, None]):
        h = tile_h.begin
        acc = hl.zeros([tile_d, tile_j], dtype=torch.float32)
        for tile_i in hl.tile(S):
            acc = torch.addmm(acc, qt[h, tile_d, tile_i], dsb[h, tile_i, tile_j])
        dkt[tile_j, h, tile_d] = acc.to(torch.bfloat16).permute(1, 0)
    return dkt


def _inputs(g, p, qt, k, a, b):
    H, S = g.shape[1], g.shape[2]
    # rope tables [1, S, 1, 32, 1] are used with each value repeated twice:
    # view k's first 64 columns as [.., 32, 2] instead of expanding the tables.
    R2 = a.size(3)
    k2 = k[:, :, : 2 * R2].unflatten(2, (R2, 2))
    return g.view(H, S, -1), p.view(H, S, -1), qt, k, k2, a[0, :, 0, :, 0], b[0, :, 0, :, 0]


def _outputs(dq, dkt, mul, mul_1):
    S, H, D = dq.shape
    R = mul.size(2) * mul.size(3)
    return (
        dkt.view(1, S, H, D),
        dq.view(1, S, H, D)[..., R:],
        mul.view(1, S, H, R),
        mul_1.view(1, S, H, R),
    )


def fused(g, p, qt, k, a, b, *shape_params):
    return _outputs(*fused_kernel(*_inputs(g, p, qt, k, a, b)))


def unfused(g, p, qt, k, a, b, *shape_params):
    g3, p3, qt, k, k2, ra, rb = _inputs(g, p, qt, k, a, b)
    dsb = ds_kernel(g3, p3)
    dq, mul, mul_1 = dq_kernel(dsb, k, k2, ra, rb)
    dkt = dk_kernel(qt, dsb)
    return _outputs(dq, dkt, mul, mul_1)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
