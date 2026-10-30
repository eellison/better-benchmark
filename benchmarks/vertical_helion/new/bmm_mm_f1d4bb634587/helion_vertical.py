"""Window-attention backward: projection -> dP -> softmax bwd -> ds @ k (Swin, 1 occ).

Graph: q = x @ W (K = 128, tokens of 8192 windows x 49, 4 heads x 32),
dP = q_h @ k_h^T (49 x 49), probs recomputed from saved scores + relative
position bias (gather from a [169, 4] table) with saved max/sum, ds =
probs * (dP - rowsum(dP * probs)), out = bf16(ds) @ v * scale.
Outputs: q (head-major), ds (f32), bf16 ds, out.

One program per (window, head): every stage only needs the window's own
49 rows (slice = window x head x row; F: 32, 49, 1). The bias gather reads
an external table. No recomputation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per (window, head), all four stages on the 49 rows."
SCALE = 0.1767766952966369


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x3, w3, kt, sp, idx, table, mx, sm, vt):
    NW, S, C = x3.shape
    _, H, D = w3.shape
    S = hl.specialize(S)
    D = hl.specialize(D)
    q_out = torch.empty([NW * H, S, D], dtype=torch.bfloat16, device=x3.device)
    ds_out = torch.empty([NW * H, S, S], dtype=torch.float32, device=x3.device)
    dsb_out = torch.empty([NW * H, S, S], dtype=torch.bfloat16, device=x3.device)
    out = torch.empty([NW * H, S, D], dtype=torch.bfloat16, device=x3.device)
    for tile_w, tile_h in hl.tile([NW, H], block_size=[1, 1]):
        w, h = tile_w.begin, tile_h.begin
        bh = w * H + h
        q = hl.zeros([S, D], dtype=torch.float32)
        for tile_c in hl.tile(C):
            q = torch.addmm(q, x3[w, :, tile_c], w3[tile_c, h, :])
        qb = q.to(torch.bfloat16)
        q_out[bh, :, :] = qb
        dp = hl.dot(qb, kt[bh, :, :], out_dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        s = sp[bh, :, :].to(torch.float32) + table[idx[:, :], h]
        probs = torch.exp(s - mx[bh, :, :]) / sm[bh, :, :]
        prod = dp * probs
        ds = prod - probs * torch.sum(prod, dim=-1, keepdim=True)
        ds_out[bh, :, :] = ds
        dsb = ds.to(torch.bfloat16)
        dsb_out[bh, :, :] = dsb
        o = hl.dot(dsb, vt[bh, :, :], out_dtype=torch.float32).to(torch.bfloat16)
        out[bh, :, :] = (o * SCALE).to(torch.bfloat16)
    return q_out, ds_out, dsb_out, out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def q_proj_kernel(x3, w3):
    NW, S, C = x3.shape
    _, H, D = w3.shape
    S = hl.specialize(S)
    D = hl.specialize(D)
    q_out = torch.empty([NW * H, S, D], dtype=torch.bfloat16, device=x3.device)
    for tile_w, tile_h in hl.tile([NW, H], block_size=[1, 1]):
        w, h = tile_w.begin, tile_h.begin
        q = hl.zeros([S, D], dtype=torch.float32)
        for tile_c in hl.tile(C):
            q = torch.addmm(q, x3[w, :, tile_c], w3[tile_c, h, :])
        q_out[w * H + h, :, :] = q.to(torch.bfloat16)
    return q_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def softmax_bwd_kernel(q, kt, sp, idx, table, mx, sm, H: hl.constexpr):
    BH, S, D = q.shape
    S = hl.specialize(S)
    D = hl.specialize(D)
    ds_out = torch.empty([BH, S, S], dtype=torch.float32, device=q.device)
    dsb_out = torch.empty([BH, S, S], dtype=torch.bfloat16, device=q.device)
    for tile_bh in hl.tile(BH, block_size=1):
        bh = tile_bh.begin
        dp = hl.dot(q[bh, :, :], kt[bh, :, :], out_dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
        s = sp[bh, :, :].to(torch.float32) + table[idx[:, :], bh % H]
        probs = torch.exp(s - mx[bh, :, :]) / sm[bh, :, :]
        prod = dp * probs
        ds = prod - probs * torch.sum(prod, dim=-1, keepdim=True)
        ds_out[bh, :, :] = ds
        dsb_out[bh, :, :] = ds.to(torch.bfloat16)
    return ds_out, dsb_out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def ds_v_kernel(dsb, vt):
    BH, S, J = dsb.shape
    S = hl.specialize(S)
    J = hl.specialize(J)
    D = hl.specialize(vt.size(2))
    out = torch.empty([BH, S, D], dtype=torch.bfloat16, device=dsb.device)
    for tile_bh in hl.tile(BH, block_size=1):
        bh = tile_bh.begin
        o = hl.dot(dsb[bh, :, :], vt[bh, :, :], out_dtype=torch.float32).to(torch.bfloat16)
        out[bh, :, :] = (o * SCALE).to(torch.bfloat16)
    return out


def _inputs(x, w, kt, sp, idx, table, mx, sm, vt, shape0):
    NW, S, H, D = shape0
    return (
        x.view(NW, S, -1), w.view(w.size(0), H, D), kt, sp[:, :S, :S], idx, table,
        mx.view(-1, S, 1), sm.view(-1, S, 1), vt,
    )


def _outputs(q, ds, dsb, out, shape0):
    NW, S, H, D = shape0
    return q, ds.view(NW, H, S, S), dsb, out.view(NW, H, S, D)


def fused(x, w, kt, sp, idx, table, mx, sm, vt, shape0, *shape_params):
    return _outputs(*fused_kernel(*_inputs(x, w, kt, sp, idx, table, mx, sm, vt, shape0)), shape0)


def unfused(x, w, kt, sp, idx, table, mx, sm, vt, shape0, *shape_params):
    x3, w3, kt, sp, idx, table, mx, sm, vt = _inputs(x, w, kt, sp, idx, table, mx, sm, vt, shape0)
    q = q_proj_kernel(x3, w3)
    ds, dsb = softmax_bwd_kernel(q, kt, sp, idx, table, mx, sm, shape0[2])
    return _outputs(q, ds, dsb, ds_v_kernel(dsb, vt), shape0)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
