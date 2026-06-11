#!/usr/bin/env python3
"""Per-model perf attribution: standalone parts vs end-to-end, exactly.

Closes the accounting identity on real hardware:

    e2e  ~=  SUM(fusible standalone_us x occurrences)
           + SUM(extern  standalone_us x occurrences)
           - G x (total_occurrences - 1)

where G is the CUDAGraph launch floor: the intercept of replay_us vs
n_kernels, measured at runtime on graphs of k no-op kernels (~4.5us on
B200). Every standalone point pays one graph launch; the real model pays
it once, so G is removed per occurrence and added back once. The
per-kernel-in-graph slope (~0.9us) appears identically on both sides
(same kernels, same counts) and cancels — only the intercept is
corrected.

Validated 2026-06-11 on 10 timm infer models:
raw parts/e2e 1.14-1.34 -> corrected 1.00-1.09. The two undershoots are
real perf findings, not accounting error: deit 0.75 (in-model layernorm
runs up to 4x slower than the identical standalone shape) and nfnet 0.92
(dominant conv slightly faster standalone). Residual +2-9% above 1.0 is
compile-context effects (epilogue fusion the standalone repro can't see).

Methodology (must match repro_harness.benchmark_repro):
  - coordinate_descent_tuning ON
  - fresh torch._dynamo.reset() + torch.compile per shape point
    (without it, dynamo marks dims dynamic after the 2nd shape and every
    later point gets a slow dynamic-shape kernel — the bug that produced
    0.12x-4.23x attribution scatter in the first validation round)
  - CUDAGraph capture + do_bench(replay, return_mode="min")

Dedup: nothing is benched twice within a run.
  - fusible: one bench per (pattern_hash, shape_hash) canonical point,
    shared across all models via occurrence-count joins
  - extern: one bench per (target, exact input signature incl. stride),
    run-level cache shared across models

Usage:
    python scripts/model_attribution.py --corpus-root /tmp/recapture_corpus/repros \
        --suite timm --mode infer --models mobilenetv2_100,resnet18
    python scripts/model_attribution.py --corpus-root ... --models all \
        --output results/attribution_b200.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import torch.fx as fx
from triton.testing import do_bench

from model_graph_accounting import analyze_graph, trace_full_graph
from full_graph_harness import load_full_graph
from repro_harness import parse_shapes_config

N_WARMUP = 10
N_REP = 50


# ============================================================================
# CUDAGraph launch floor (measured per run, never hardcoded)
# ============================================================================

def measure_graph_launch_floor() -> tuple[float, float]:
    """(intercept_us, slope_us) of replay_us = intercept + slope * n_kernels.

    Graphs of k no-op kernels (1-element add_). The intercept is the
    per-graph launch cost each standalone measurement pays; the slope is
    per-kernel-in-graph overhead, identical on both sides of the
    accounting identity and therefore not corrected.
    """
    x = torch.zeros(1, device="cuda")
    ks = [1, 2, 5, 10, 20, 50, 100]
    ts = []
    for k in ks:
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(k):
                x.add_(1.0)
        torch.cuda.synchronize()
        ts.append(do_bench(lambda: g.replay(), warmup=25, rep=100,
                           return_mode="min") * 1000)
    n = len(ks)
    mk = sum(ks) / n
    mt = sum(ts) / n
    slope = (sum((k - mk) * (t - mt) for k, t in zip(ks, ts))
             / sum((k - mk) ** 2 for k in ks))
    return mt - slope * mk, slope


# ============================================================================
# Shared bench primitive
# ============================================================================

def _bench_replay(fn) -> float:
    """CUDAGraph-capture fn() and return min replay time in us."""
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
    return do_bench(lambda: g.replay(), warmup=N_WARMUP, rep=N_REP,
                    return_mode="min") * 1000


# ============================================================================
# Fusible side: canonical repro points
# ============================================================================

def bench_fusible_points(needed: set[tuple[str, str]],
                         canonical_dir: Path) -> dict[tuple[str, str], float]:
    """Bench each needed (pattern_hash, shape_hash) point exactly once."""
    times: dict[tuple[str, str], float] = {}
    needed_patterns = {p for p, _ in needed}
    for d in sorted(canonical_dir.iterdir()):
        phash = d.name.rsplit("_", 1)[-1]
        if phash not in needed_patterns or not (d / "repro.py").exists():
            continue
        spec = importlib.util.spec_from_file_location("r", d / "repro.py")
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = float("nan")
        spec.loader.exec_module(mod)
        for pt in json.loads((d / "shapes.json").read_text())["points"]:
            key = (phash, pt["shape_hash"])
            if key not in needed or key in times:
                continue
            inputs = [t.cuda() if isinstance(t, torch.Tensor) else t
                      for t in parse_shapes_config(pt["signature"])]
            torch._dynamo.reset()
            compiled = torch.compile(mod.Repro())
            times[key] = _bench_replay(lambda: compiled(*inputs))
    return times


# ============================================================================
# Extern side: non-fusible ops at exact input metas
# ============================================================================

def _meta_sig(val) -> str:
    if torch.is_tensor(val):
        return f"T({list(val.shape)},{val.dtype},{list(val.stride())})"
    return repr(val)


def _generation_kind_for_arg(node, arg_pos) -> dict | None:
    """Generation spec for an integer tensor arg of an extern op.

    Same op semantics as full_graph_harness.infer_index_bounds_from_gm —
    here the consumer IS the node being benched, so inference is direct:
    embedding(weight, indices) bounds indices by weight.shape[0];
    gather/index_select/scatter by input.shape[dim]; _embedding_bag
    offsets are a sorted-"offsets" kind. Returns a make_inputs_from_config
    generation dict, or None (harness default: small non-negative ints).
    """
    def _arg_shape(i):
        if i >= len(node.args) or not isinstance(node.args[i], fx.Node):
            return None
        v = node.args[i].meta.get("val")
        return list(v.shape) if torch.is_tensor(v) else None

    t = node.target
    a = torch.ops.aten
    if t in (a.embedding.default, a._embedding_bag.default) and arg_pos == 1:
        w = _arg_shape(0)
        return {"kind": "index", "low": 0, "high": w[0]} if w else None
    if t == a._embedding_bag.default and arg_pos == 2:
        idx = _arg_shape(1)
        return {"kind": "offsets", "high": idx[0]} if idx else None
    if t in (a.gather.default, a.index_select.default,
             a.scatter.src, a.scatter.value, a.scatter_add.default):
        if arg_pos == 2:  # (input, dim, index, ...)
            inp = _arg_shape(0)
            dim = node.args[1] if len(node.args) > 1 else 0
            if inp and isinstance(dim, int):
                return {"kind": "index", "low": 0, "high": inp[dim]}
    if t == a.index.Tensor and arg_pos == 1:
        inp = _arg_shape(0)
        return {"kind": "index", "low": 0, "high": min(inp)} if inp else None
    return None


def _fabricate(val, dev="cuda", node=None, arg_pos=None):
    """Materialize a bench input from a fake-tensor meta val.

    REUSES repro_harness.make_inputs_from_config — the project's single
    input-generation implementation (randn floats, bounded randint index
    ints, permutation/offsets kinds) — by building the same spec dict the
    shapes.json pipeline produces. No value semantics live here.
    """
    if not torch.is_tensor(val):
        return val
    from repro_harness import make_inputs_from_config
    spec = {
        "shape": list(val.shape),
        "dtype": str(val.dtype),
        "stride": list(val.stride()),
        "device": dev,
    }
    if node is not None and arg_pos is not None:
        gen = _generation_kind_for_arg(node, arg_pos)
        if gen:
            spec["gen"] = gen
    return make_inputs_from_config({"inputs": [spec]})[0]


def _arg_sig(a) -> str:
    """Exact signature of one arg: tensor metas for Nodes, repr for
    literals, recursing into containers. (fx.node.map_arg only visits
    Nodes — literals like conv padding would be silently dropped, merging
    padding-only-different calls into one point. Caught by
    test_canonical_invariants.)"""
    if isinstance(a, fx.Node):
        return _meta_sig(a.meta.get("val"))
    if isinstance(a, (list, tuple)):
        inner = ",".join(_arg_sig(x) for x in a)
        return f"[{inner}]" if isinstance(a, list) else f"({inner})"
    if isinstance(a, dict):
        return "{" + ",".join(f"{k}:{_arg_sig(v)}"
                              for k, v in sorted(a.items())) + "}"
    return repr(a)


def collect_extern_points(gm) -> dict[tuple[str, str], dict]:
    """(target, exact input signature) -> {count, node} for non-fusible ops.

    The signature includes shape, dtype AND stride of every tensor arg
    plus the repr of every literal arg — two conv calls differing only in
    padding or layout are distinct points.
    """
    from capture_hook import get_fusion_partitions, partition_node_is_supported

    partitioned: set = set()
    for comp in get_fusion_partitions(gm):
        partitioned.update(comp)
    points: dict[tuple[str, str], dict] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or node in partitioned:
            continue
        if partition_node_is_supported(node):
            continue
        key = (str(node.target), _arg_sig((node.args, node.kwargs)))
        if key in points:
            points[key]["count"] += 1
        else:
            points[key] = {"count": 1, "node": node}
    return points


def bench_extern_point(node) -> float:
    """Bench one non-fusible op standalone at its exact input metas.

    Top-level positional args carry their position into _fabricate so
    integer index tensors get inferred valid bounds (embedding indices
    bounded by the weight's vocab dim, gather indices by input.shape[dim]).
    """
    args = tuple(
        _fabricate(a.meta.get("val"), node=node, arg_pos=i)
        if isinstance(a, fx.Node)
        else fx.node.map_arg(a, lambda x: _fabricate(x.meta.get("val")))
        for i, a in enumerate(node.args)
    )
    kwargs = fx.node.map_arg(
        node.kwargs, lambda x: _fabricate(x.meta.get("val")))
    return _bench_replay(lambda: node.target(*args, **kwargs))


# ============================================================================
# End-to-end reference
# ============================================================================

def bench_e2e(graph_path: Path) -> float:
    inst, inputs, _ = load_full_graph(graph_path, default_device="cuda")
    inputs = [t.cuda() if isinstance(t, torch.Tensor) else t for t in inputs]
    torch._dynamo.reset()
    compiled = torch.compile(inst)
    return _bench_replay(lambda: compiled(*inputs))


# ============================================================================
# Per-model attribution
# ============================================================================

def attribute_model(model_dir: Path, canonical_dir: Path,
                    fusible_cache: dict[tuple[str, str], float],
                    extern_cache: dict[tuple[str, str], float],
                    launch_floor_us: float) -> dict:
    graphs = sorted(model_dir.glob("full_graph_*.py"))
    fus_occ: Counter = Counter()
    extern_occ: Counter = Counter()
    extern_nodes: dict[tuple[str, str], object] = {}
    e2e_total = 0.0
    for g in graphs:
        gm = trace_full_graph(g)
        acc = analyze_graph(gm, str(g), g.stem)
        fus_occ.update((o.pattern_hash, o.shape_hash) for o in acc.occurrences)
        for key, info in collect_extern_points(gm).items():
            extern_occ[key] += info["count"]
            extern_nodes.setdefault(key, info["node"])
        e2e_total += bench_e2e(g)

    # Fusible: bench only points not already in the run cache.
    missing = set(fus_occ) - set(fusible_cache)
    fusible_cache.update(bench_fusible_points(missing, canonical_dir))
    fus_unmatched = {k: n for k, n in fus_occ.items() if k not in fusible_cache}
    fus_sum = sum(fusible_cache[k] * n for k, n in fus_occ.items()
                  if k in fusible_cache)

    # Extern: run-level cache keyed on (target, exact signature).
    extern_failed: list[str] = []
    for key, node in extern_nodes.items():
        if key in extern_cache:
            continue
        try:
            extern_cache[key] = bench_extern_point(node)
        except Exception as e:  # noqa: BLE001 -- record, never silently drop
            extern_failed.append(f"{key[0]}: {type(e).__name__}: {e}")
    ext_unmatched = {k: n for k, n in extern_occ.items()
                     if k not in extern_cache}
    ext_sum = sum(extern_cache[k] * n for k, n in extern_occ.items()
                  if k in extern_cache)

    n_occ = sum(fus_occ.values()) + sum(extern_occ.values())
    parts = fus_sum + ext_sum
    # Each standalone occurrence embeds one graph-launch intercept G; the
    # e2e side embeds G per FULL GRAPH (e2e_total sums one bench_e2e per
    # graph). Correction is therefore G*(n_occ - n_graphs) — using (n_occ-1)
    # over-subtracts on multi-graph models (opus verifier, 2026-06-11;
    # coincidentally equal for the 1-graph timm infer validation set).
    corrected = parts - launch_floor_us * (n_occ - len(graphs))
    return {
        "e2e_us": round(e2e_total, 1),
        "sum_fusible_us": round(fus_sum, 1),
        "sum_extern_us": round(ext_sum, 1),
        "sum_parts_us": round(parts, 1),
        "corrected_parts_us": round(corrected, 1),
        "ratio_raw": round(parts / e2e_total, 3),
        "ratio_corrected": round(corrected / e2e_total, 3),
        "n_graphs": len(graphs),
        "n_fusible_occurrences": sum(fus_occ.values()),
        "n_fusible_points": len(fus_occ),
        "n_extern_occurrences": sum(extern_occ.values()),
        "n_extern_points": len(extern_occ),
        "fusible_unmatched": [f"{p}/{s} x{n}"
                              for (p, s), n in sorted(fus_unmatched.items())],
        "extern_unmatched": [f"{t} x{n}"
                             for (t, _), n in sorted(ext_unmatched.items())],
        "extern_bench_failures": extern_failed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Standalone-parts vs e2e attribution per model")
    parser.add_argument("--corpus-root", type=Path, required=True,
                        help="Corpus root containing models/ and canonical/")
    parser.add_argument("--suite", default="timm")
    parser.add_argument("--mode", default="infer")
    parser.add_argument("--models", required=True,
                        help="Comma-separated model names, or 'all'")
    parser.add_argument("--output", "-o", type=Path, default=None)
    args = parser.parse_args()

    import torch._inductor.config as inductor_config
    inductor_config.coordinate_descent_tuning = True

    suite_dir = args.corpus_root / "models" / args.suite / args.mode
    canonical_dir = args.corpus_root / "canonical"
    if args.models == "all":
        model_dirs = sorted(d for d in suite_dir.iterdir() if d.is_dir())
    else:
        model_dirs = [suite_dir / m for m in args.models.split(",") if m]

    floor_us, slope_us = measure_graph_launch_floor()
    print(f"CUDAGraph floor: replay = {floor_us:.2f} + {slope_us:.3f} * "
          f"n_kernels (us)", file=sys.stderr)

    fusible_cache: dict[tuple[str, str], float] = {}
    extern_cache: dict[tuple[str, str], float] = {}
    out = {"_launch_floor_us": round(floor_us, 2),
           "_launch_slope_us": round(slope_us, 3),
           "_device": torch.cuda.get_device_name(0),
           "models": {}}
    for md in model_dirs:
        if not md.is_dir():
            out["models"][md.name] = {"error": "model dir not found"}
            continue
        try:
            r = attribute_model(md, canonical_dir, fusible_cache,
                                extern_cache, floor_us)
            out["models"][md.name] = r
            print(f"{md.name}: e2e={r['e2e_us']:.0f}us "
                  f"parts={r['sum_parts_us']:.0f}us "
                  f"corrected={r['corrected_parts_us']:.0f}us "
                  f"ratio={r['ratio_corrected']:.2f} "
                  f"unmatched={len(r['fusible_unmatched']) + len(r['extern_unmatched'])}",
                  flush=True)
        except Exception as e:  # noqa: BLE001 -- per-model isolation
            out["models"][md.name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{md.name}: ERROR {type(e).__name__}: {e}", flush=True)

    print(f"run cache: {len(fusible_cache)} fusible points, "
          f"{len(extern_cache)} extern points benched once", file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
