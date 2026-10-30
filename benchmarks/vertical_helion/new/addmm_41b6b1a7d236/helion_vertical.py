"""Global average pool -> classifier linear (DenseNet/ResNet heads, 4 occurrences).

Graph: mean over the 4x4 (or 8x8) spatial positions of bf16 [128, C, H, W]
-> [128, C] (output), addmm with W [C, 10] + bias (f32 parameters cast to
bf16) -> [128, 10].

One program per sample tile holds its pooled C-wide rows (F = C = 184 or 64)
and does the 10-wide linear. No recomputation; the whole problem is tiny.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import helion
import helion.language as hl

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
from vertical_fusion_bench import run  # noqa: E402

NOTES = "One kernel: program per sample tile, pooled rows (F = C) feed the linear."


@helion.kernel(static_shapes=True, autotune_effort="quick")
def fused_kernel(x, w, bias):
    B, C, P = x.shape
    C = hl.specialize(C)
    P = hl.specialize(P)
    N = hl.specialize(w.size(1))
    pooled = torch.empty([B, C], dtype=torch.bfloat16, device=x.device)
    out = torch.empty([B, N], dtype=torch.bfloat16, device=x.device)
    for tile_b in hl.tile(B):
        p = torch.mean(x[tile_b, :, :].to(torch.float32), dim=-1).to(torch.bfloat16)
        pooled[tile_b, :] = p
        acc = hl.dot(p, w[:, :].to(torch.bfloat16), out_dtype=torch.float32)
        out[tile_b, :] = (acc + bias[:].to(torch.bfloat16).to(torch.float32)).to(torch.bfloat16)
    return pooled, out


@helion.kernel(static_shapes=True, autotune_effort="quick")
def pool_kernel(x):
    B, C, P = x.shape
    P = hl.specialize(P)
    pooled = torch.empty([B, C], dtype=torch.bfloat16, device=x.device)
    for tile_b, tile_c in hl.tile([B, C]):
        pooled[tile_b, tile_c] = torch.mean(x[tile_b, tile_c, :].to(torch.float32), dim=-1).to(torch.bfloat16)
    return pooled


@helion.kernel(static_shapes=True, autotune_effort="quick")
def linear_kernel(p, w, bias):
    B, C = p.shape
    C = hl.specialize(C)
    N = hl.specialize(w.size(1))
    out = torch.empty([B, N], dtype=torch.bfloat16, device=p.device)
    for tile_b in hl.tile(B):
        acc = hl.dot(p[tile_b, :], w[:, :].to(torch.bfloat16), out_dtype=torch.float32)
        out[tile_b, :] = (acc + bias[:].to(torch.bfloat16).to(torch.float32)).to(torch.bfloat16)
    return out


def fused(x, bias, w, *shape_params):
    return fused_kernel(x.flatten(2), w, bias)


def unfused(x, bias, w, *shape_params):
    p = pool_kernel(x.flatten(2))
    return p, linear_kernel(p, w, bias)


if __name__ == "__main__":
    run(__file__, fused, unfused, NOTES)
