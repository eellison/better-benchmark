"""Runnable end-to-end example of the dynamic-shape pipeline.

Captures a set of small models live (torch.compile(dynamic=True) with
mark_dynamic'd dims), merges them through merge_one_capture into a SCRATCH
canonical root, then loads every merged family and runs it eager at its
native binding and at a 2x rebind. Demonstrates, on live captures:

  * dynamic capture (symbols + guards harvested from dynamo's ShapeEnv)
  * family-identity grouping: the two GroupNorm variants (C=64 vs C=128)
    share a pattern_hash but bake different reshape split factors, so they
    land in var_mean_<hash> and var_mean_<hash>__2 — one symbolic family
    per dir, each with its own repro.py and guards
  * symbolization-aware shape_hash: no (pattern, point-hash) duplicates
  * point instantiation at arbitrary bindings via load_shape_configs,
    with out-of-range bindings rejected loudly

Everything is written under --output-root (default
/tmp/scratch_space/exercise_live) — the checked-in corpus is never touched.

Evaluating a family at DIFFERENT POINTS, and static-vs-dynamic perf
--------------------------------------------------------------------
Every merged repro.py is self-benchmarking. To measure a family at several
eval points (one STATIC specialized compile per binding):

    python <root>/repros/canonical/<dir>/repro.py \
        --bind s0=8,s1=8 --bind s0=16,s1=16 --bind s0=32,s1=32 \
        --output static.json

and to measure the ONE GENERAL dynamic artifact at the same points
(mark_dynamic + two-distinct-shape pre-warm, then per-binding timing):

    python <root>/repros/canonical/<dir>/repro.py --dynamic \
        --bind s0=8,s1=8 --bind s0=16,s1=16 --bind s0=32,s1=32 \
        --output dynamic.json

The per-point ratio dynamic/static is the cost of dynamism across the
family (the generalization curve). Both arms use the same methodology
(GPU lock, CUDAGraph capture, do_bench min).

Usage:
    python scripts/exercise_dynamic_shapes_live.py [--output-root DIR]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import merge_one_capture
from repro_harness import load_shape_configs, make_inputs_from_config


class GroupNormVarMean(torch.nn.Module):
    """Coupled reduction with a BAKED split factor (channels/groups): the
    C=64 and C=128 instances emit the same ops with a different constant —
    the family-split case."""

    def __init__(self, channels):
        super().__init__()
        self.c = channels
        self.w = torch.nn.Parameter(torch.randn(channels))
        self.b = torch.nn.Parameter(torch.randn(channels))

    def forward(self, x):
        v = x.view(64, 32, self.c // 32, x.shape[2] * x.shape[3])
        var, mean = torch.var_mean(v, [2, 3], correction=0, keepdim=True)
        n = (v - mean) * torch.rsqrt(var + 1e-5)
        return (n.view(x.shape) * self.w[None, :, None, None]
                + self.b[None, :, None, None])


class SoftmaxLastDim(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1) + 1.0


class LayerNormDyn(torch.nn.Module):
    def forward(self, x, w, b):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), w, b, 1e-5)


class AmaxSum(torch.nn.Module):
    def forward(self, x):
        m = torch.amax(x, dim=1, keepdim=True)
        return (x - m).exp().sum(dim=1)


class MatmulDynBatch(torch.nn.Module):
    """Extern bmm with a dynamic batch dim + fusible epilogue."""

    def forward(self, a, b):
        return torch.bmm(a, b).relu()


class PointwiseGelu(torch.nn.Module):
    def forward(self, x, bias):
        return torch.nn.functional.gelu(x) * x + bias


def _mk(shape, dyn_dims):
    t = torch.randn(*shape, device="cuda")
    for d in dyn_dims:
        torch._dynamo.mark_dynamic(t, d)
    return t


CASES = [
    ("var_mean_c64", lambda: GroupNormVarMean(64).cuda(),
     lambda: [_mk((64, 64, 16, 16), [2, 3])]),
    ("var_mean_c128", lambda: GroupNormVarMean(128).cuda(),
     lambda: [_mk((64, 128, 8, 8), [2, 3])]),
    ("softmax_lastdim", lambda: SoftmaxLastDim().cuda(),
     lambda: [_mk((32, 128, 256), [1, 2])]),
    ("layernorm_dyn", lambda: LayerNormDyn().cuda(),
     lambda: [_mk((16, 512, 768), [1]),
              torch.randn(768, device="cuda"),
              torch.randn(768, device="cuda")]),
    ("amax_sum", lambda: AmaxSum().cuda(),
     lambda: [_mk((128, 512), [0, 1])]),
    ("matmul_dyn_batch", lambda: MatmulDynBatch().cuda(),
     lambda: [_mk((32, 64, 128), [0]), _mk((32, 128, 96), [0])]),
    ("pointwise_gelu", lambda: PointwiseGelu().cuda(),
     lambda: [_mk((64, 1024), [0, 1]), torch.randn(1024, device="cuda")]),
]


def capture_and_merge(root: Path):
    canonical_root = root / "repros"
    summary = []
    for label, mk_model, mk_inputs in CASES:
        cap_dir = root / f"cap_{label}"
        torch._dynamo.reset()
        install_capture_hook(str(cap_dir), label=label)
        err = None
        try:
            m = mk_model()
            with torch.no_grad():
                torch.compile(m, dynamic=True)(*mk_inputs())
        except Exception as e:  # report per-case, keep exercising the rest
            err = f"{type(e).__name__}: {e}"
        finally:
            uninstall_capture_hook()
        n_captured = n_guards = merged = 0
        if err is None:
            idx_path = cap_dir / "index.json"
            if idx_path.exists():
                idx = json.loads(idx_path.read_text())
                entries = (idx.get("captured", idx)
                           if isinstance(idx, dict) else idx)
                n_captured = len(entries)
                n_guards = sum(len(e.get("guards") or []) for e in entries)
            merged = merge_one_capture(cap_dir, canonical_root, label,
                                       suite="other", mode="infer")
        summary.append((label, err, n_captured, n_guards, merged))
    return canonical_root, summary


def run_points(canonical_root: Path):
    """Every point eager at its own binding, then a 2x rebind; a 0-binding
    must be rejected loudly."""
    results = []
    for d in sorted((canonical_root / "canonical").iterdir()):
        repro_py = d / "repro.py"
        shapes = json.loads((d / "shapes.json").read_text())
        spec = importlib.util.spec_from_file_location(f"_ex_{d.name}", repro_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ran = rebound = 0
        notes = []
        for cfg in load_shape_configs(str(repro_py)).values():
            with torch.no_grad():
                mod.Repro()(*make_inputs_from_config(cfg))
            ran += 1
        syms = sorted(shapes.get("symbols") or {})
        if syms:
            hints = {s: (shapes["symbols"][s].get("hint") or 8) for s in syms}
            try:
                cfgs2 = load_shape_configs(
                    str(repro_py),
                    symbol_bindings={s: max(2, hints[s] * 2) for s in syms})
                for cfg in cfgs2.values():
                    with torch.no_grad():
                        mod.Repro()(*make_inputs_from_config(cfg))
                    rebound += 1
            except ValueError as e:
                notes.append(f"2x rebind rejected ({str(e)[:60]})")
            zero_rejected = False
            try:
                load_shape_configs(str(repro_py),
                                   symbol_bindings={s: 0 for s in syms})
            except (ValueError, RuntimeError):
                zero_rejected = True
            if not zero_rejected:
                notes.append("ZERO BINDING NOT REJECTED")
        results.append((d.name, len(shapes["points"]), ran, rebound,
                        len(shapes.get("guards") or []), "; ".join(notes)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path,
                        default=Path("/tmp/scratch_space/exercise_live"))
    args = parser.parse_args()
    root = args.output_root

    torch.manual_seed(0)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    canonical_root, cap_summary = capture_and_merge(root)
    print("\n================ LIVE CAPTURE + MERGE ================")
    for label, err, n_cap, n_guards, merged in cap_summary:
        if err:
            print(f"[{label}] CAPTURE ERROR: {err}")
        else:
            print(f"[{label}] captured={n_cap} guards={n_guards} merged={merged}")

    print("\n================ CANONICAL DIRS ================")
    key_counts = Counter()
    for d in sorted((canonical_root / "canonical").iterdir()):
        meta = json.loads((d / "meta.json").read_text())
        s = json.loads((d / "shapes.json").read_text())
        for p in s["points"]:
            key_counts[(meta["pattern_hash"], p["shape_hash"])] += 1
        pts = [(p["shape_hash"], p.get("bindings")) for p in s["points"]]
        print(f"{d.name}: {pts}")
    dupes = {k: c for k, c in key_counts.items() if c > 1}
    print("(pattern, point) duplicates:", dupes if dupes else "NONE")

    print("\n================ LOAD + EAGER RUNS ================")
    failed = False
    for name, n_pts, ran, rebound, n_guards, notes in run_points(canonical_root):
        line = (f"[{name}] points={n_pts} ran_at_native={ran} "
                f"ran_at_2x={rebound} guards={n_guards}")
        if notes:
            line += f"  NOTE: {notes}"
            failed = failed or "NOT REJECTED" in notes
        print(line)
    print("\nDONE" + (" (with failures)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
