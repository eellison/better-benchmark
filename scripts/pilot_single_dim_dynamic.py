"""Pilot: SINGLE-dim dynamism — the realistic case (it is rare for all dims
to be dynamic; usually exactly one is: batch, or seq-len, ...).

Captures three blocks, each with exactly ONE dim marked dynamic, into a
scratch canonical root (the checked-in corpus is never touched):

  ffn_batch_dyn        transformer FFN (GELU + residual + LayerNorm over a
                       STATIC 1024 feature dim); dynamic dim = batch.
                       The reduction numel is static -> the persistent-
                       reduction heuristic should survive dynamism.
  softmax_seq_dyn      attention-score softmax [16, 8, 128, K] over K;
                       dynamic dim = K = THE REDUCTION DIM. The contrast
                       case: reduction numel is symbolic.
  groupnorm_batch_dyn  the GroupNorm block with batch dynamic and spatial
                       STATIC (contrast with the all-spatial-dynamic
                       var_mean sweep, which paid 2-5x).

Also probes the batch=1 story mechanically (no timing): with the batch dim
marked dynamic, run batch=1 first, then batch=8 — does dynamo specialize a
separate 0/1 graph, does the batch=8 run land on the general kernel, and
how many graphs/compiles result?

Timing matrix (run after this script; repro self-bench, GPU lock,
CUDAGraph, do_bench min; see the sweep driver in the investigation doc):

  static per point:        repro.py --bind <p> ... --output static.json
  dynamic, hint-first:     repro.py --dynamic --prewarm <hint> --prewarm <2*hint> --bind <p> ...
  dynamic, small-first:    repro.py --dynamic --prewarm <small> --prewarm <hint> --bind <p> ...

The first --prewarm binding is the hint inductor tunes the general kernel
at, so hint-first vs small-first isolates compile-history sensitivity
("ran first at 2, then timed at 8 — does that mess up perf?").

Usage:
    python scripts/pilot_single_dim_dynamic.py [--output-root DIR]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import merge_one_capture


class FFNBlock(torch.nn.Module):
    """Linear -> GELU -> Linear -> +residual -> LayerNorm(1024).
    The Linears are extern (matmul); the captured fusible regions are the
    GELU pointwise and the residual+LayerNorm reduction over a STATIC
    feature dim with a dynamic outer (batch) dim."""

    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(1024, 4096)
        self.down = torch.nn.Linear(4096, 1024)
        self.ln = torch.nn.LayerNorm(1024)

    def forward(self, x):
        return self.ln(x + self.down(torch.nn.functional.gelu(self.up(x))))


class SoftmaxSeq(torch.nn.Module):
    """Attention-score softmax over the LAST (dynamic) dim: [16,8,128,K]."""

    def forward(self, scores):
        return torch.softmax(scores, dim=-1) * 0.125


class GroupNormBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gn = torch.nn.GroupNorm(32, 64)

    def forward(self, x):
        return torch.nn.functional.relu(self.gn(x))


def _mk(shape, dyn_dim):
    t = torch.randn(*shape, device="cuda")
    torch._dynamo.mark_dynamic(t, dyn_dim)
    return t


CASES = [
    # (label, module factory, input factory) — exactly ONE dynamic dim each.
    ("ffn_batch_dyn", lambda: FFNBlock().cuda(),
     lambda: [_mk((8, 1024), 0)]),
    ("softmax_seq_dyn", lambda: SoftmaxSeq().cuda(),
     lambda: [_mk((16, 8, 128, 256), 3)]),
    ("groupnorm_batch_dyn", lambda: GroupNormBlock().cuda(),
     lambda: [_mk((8, 64, 16, 16), 0)]),
]


def batch1_probe():
    """Mechanism probe for 'run first at batch=1, then 8': marked-dynamic
    batch, feed 1 then 8 then 16. Reports graph/recompile counts and whether
    8/16 share one general graph (i.e. batch=1 specializes SEPARATELY and
    does not poison the general kernel's hint)."""
    import torch._dynamo as dynamo
    dynamo.reset()
    counters = dynamo.utils.counters
    counters.clear()
    m = torch.compile(FFNBlock().cuda())
    out = {}
    with torch.no_grad():
        for b in (1, 8, 16):
            x = torch.randn(b, 1024, device="cuda")
            torch._dynamo.mark_dynamic(x, 0)
            try:
                m(x)
                out[b] = "ok"
            except Exception as e:
                out[b] = f"{type(e).__name__}: {str(e)[:80]}"
    stats = dict(counters.get("stats", {}))
    return out, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path,
                        default=Path("/tmp/scratch_space/pilot_single_dim"))
    args = parser.parse_args()
    root = args.output_root

    torch.manual_seed(0)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    canonical_root = root / "repros"
    print("================ SINGLE-DIM DYNAMIC CAPTURES ================")
    for label, mk_model, mk_inputs in CASES:
        cap_dir = root / f"cap_{label}"
        torch._dynamo.reset()
        install_capture_hook(str(cap_dir), label=label)
        try:
            m = mk_model()
            with torch.no_grad():
                torch.compile(m, dynamic=None)(*mk_inputs())
        finally:
            uninstall_capture_hook()
        merged = merge_one_capture(cap_dir, canonical_root, label,
                                   suite="other", mode="infer")
        print(f"[{label}] merged={merged}")

    print("\n================ MERGED FAMILIES ================")
    for d in sorted((canonical_root / "canonical").iterdir()):
        s = json.loads((d / "shapes.json").read_text())
        syms = {k: v.get("hint") for k, v in (s.get("symbols") or {}).items()}
        print(f"{d.name}: symbols={syms} guards={s.get('guards')} "
              f"points={[p['shape_hash'] for p in s['points']]}")

    print("\n================ BATCH=1 -> 8 -> 16 PROBE ================")
    torch._dynamo.reset()
    runs, stats = batch1_probe()
    for b, r in runs.items():
        print(f"batch={b}: {r}")
    print(f"dynamo stats: {stats}")
    print("\nDONE — bench the families per the module docstring.")


if __name__ == "__main__":
    main()
