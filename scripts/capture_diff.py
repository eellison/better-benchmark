#!/usr/bin/env python3
"""Diff a static capture against a dynamic capture of the same models.

A dynamic family's static rows bench the dynamic graph evaluated at the
captured binding. That is what the model ran, but it is not always what a
static capture of the same model would bench: sizes read at runtime add ops
(an attention scale computed from a SymInt instead of a constant), partitions
can differ, and inferred input generators can differ. This reports those
differences per model:

  - regions paired across the two captures by origin-op overlap, with the ops
    only one side has;
  - for each pair, the inputs at the hint (dynamic points evaluated at their
    bindings) that differ in shape, dtype, stride or generator;
  - regions with no counterpart.

Usage:
  python scripts/capture_diff.py STATIC_ROOT DYNAMIC_ROOT [--json OUT]

Each root is a repros/ tree (it has canonical/).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from input_codec import evaluate_symbolic_entry, is_symbolic_entry  # noqa: E402


def _point_inputs(shapes: dict, point: dict) -> list:
    bindings = point.get("bindings") or {}
    if not shapes.get("symbols") or point.get("requires_binding"):
        return point.get("inputs") or []
    return [evaluate_symbolic_entry(e, bindings) if is_symbolic_entry(e) else e
            for e in point.get("inputs") or []]


def _input_key(entry) -> str | None:
    """A tensor input at the hint: shape, dtype, non-contiguous stride, offset
    and generator. Scalars and shape lists are left out: a static capture
    lifts sizes as shape-param lists where a dynamic one passes symint
    scalars, the same information spelled differently."""
    if not (isinstance(entry, list) and entry and isinstance(entry[0], list)):
        return None
    shape, opts = entry[0], (entry[2] if len(entry) > 2 else {})
    contiguous, step = [], 1
    for dim in reversed(shape):
        contiguous.insert(0, step)
        step *= max(dim, 1)
    stride = opts.get("st")
    return json.dumps([shape, entry[1], None if stride in (None, contiguous) else stride,
                       opts.get("off") or None, opts.get("gen")])


def load_regions(root: Path) -> dict[str, list[dict]]:
    """model -> regions, each {name, ops, inputs (per point, at the hint)}."""
    by_model: dict[str, list[dict]] = {}
    for d in sorted((root / "canonical").iterdir()):
        if not (d / "shapes.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        shapes = json.loads((d / "shapes.json").read_text())
        for point in shapes["points"]:
            for model in point.get("models") or {"?": {}}:
                by_model.setdefault(model, []).append({
                    "name": d.name,
                    "ops": Counter(meta.get("origin_ops") or []),
                    "inputs": [k for k in map(_input_key, _point_inputs(shapes, point)) if k],
                })
    return by_model


def _overlap(a: Counter, b: Counter) -> float:
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 1.0


def diff_model(static: list[dict], dynamic: list[dict], min_overlap: float = 0.5) -> dict:
    # Pair by op overlap; regions with the same ops pair by input overlap.
    candidates = sorted(
        ((_overlap(s["ops"], d["ops"]), _overlap(Counter(s["inputs"]), Counter(d["inputs"])), i, j)
         for i, s in enumerate(static) for j, d in enumerate(dynamic)),
        reverse=True)
    used_s, used_d, pairs = set(), set(), []
    for score, _inputs_score, i, j in candidates:
        if score < min_overlap or i in used_s or j in used_d:
            continue
        used_s.add(i)
        used_d.add(j)
        s, d = static[i], dynamic[j]
        s_in, d_in = Counter(s["inputs"]), Counter(d["inputs"])
        pairs.append({
            "static": s["name"],
            "dynamic": d["name"],
            "ops_only_static": dict(s["ops"] - d["ops"]),
            "ops_only_dynamic": dict(d["ops"] - s["ops"]),
            "inputs_only_static": sorted((s_in - d_in).elements()),
            "inputs_only_dynamic": sorted((d_in - s_in).elements()),
        })
    return {
        "pairs": pairs,
        "unmatched_static": [static[i]["name"] for i in range(len(static)) if i not in used_s],
        "unmatched_dynamic": [dynamic[j]["name"] for j in range(len(dynamic)) if j not in used_d],
    }


def diff_captures(static_root: Path, dynamic_root: Path) -> dict:
    static, dynamic = load_regions(static_root), load_regions(dynamic_root)
    return {model: diff_model(static.get(model, []), dynamic.get(model, []))
            for model in sorted(set(static) | set(dynamic))}


_DIFF_KEYS = ("ops_only_static", "ops_only_dynamic", "inputs_only_static", "inputs_only_dynamic")


def _differs(pair: dict) -> bool:
    return any(pair[k] for k in _DIFF_KEYS)


def render(report: dict) -> str:
    lines = []
    for model, r in report.items():
        same = sum(not _differs(p) for p in r["pairs"])
        n_static, n_dynamic = len(r["unmatched_static"]), len(r["unmatched_dynamic"])
        lines.append(f"{model}: {len(r['pairs'])} paired ({same} identical at the hint), "
                     f"{n_static} static-only, {n_dynamic} dynamic-only")
        for p in r["pairs"]:
            if not _differs(p):
                continue
            lines.append(f"  {p['static']} <-> {p['dynamic']}")
            for key in _DIFF_KEYS:
                if p[key]:
                    lines.append(f"    {key}: {p[key]}")
        for side in ("unmatched_static", "unmatched_dynamic"):
            if r[side]:
                lines.append(f"  {side}: {r[side]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("static_root", type=Path)
    ap.add_argument("dynamic_root", type=Path)
    ap.add_argument("--json", type=Path, help="also write the report as JSON")
    args = ap.parse_args(argv)
    report = diff_captures(args.static_root, args.dynamic_root)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
