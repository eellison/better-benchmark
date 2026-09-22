"""Merge two sweep result files (dir keys, points_by_shape structure), then
run comparison. Used to combine two sweeps that cover different (dir, shape)
subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(a: dict, b: dict) -> dict:
    """Merge two bench_parallel output dicts. b takes precedence on conflict."""
    out = {}
    all_keys = set(a.keys()) | set(b.keys())
    for k in all_keys:
        if k.startswith("__") or k == "_metadata":
            continue
        entry_a = a.get(k)
        entry_b = b.get(k)
        if entry_a is None:
            out[k] = entry_b
            continue
        if entry_b is None:
            out[k] = entry_a
            continue
        # Both have this dir. Merge points_by_shape from both.
        pbs_a = entry_a.get("points_by_shape", {})
        pbs_b = entry_b.get("points_by_shape", {})
        pbs = {**pbs_a, **pbs_b}  # b takes precedence per shape
        pts_a = entry_a.get("points", {})
        pts_b = entry_b.get("points", {})
        pts = {**pts_a, **pts_b}
        out[k] = {
            **entry_a,  # copy other fields
            "points_by_shape": pbs,
            "points": pts,
        }
    # Also merge __failures__ from both, but a valid entry beats a failure
    fail_a = a.get("__failures__", {}) or {}
    fail_b = b.get("__failures__", {}) or {}
    failures = {}
    for k in set(fail_a.keys()) | set(fail_b.keys()):
        if k in out:
            continue  # already merged as valid
        failures[k] = fail_b.get(k, fail_a.get(k))
    if failures:
        out["__failures__"] = failures
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triton", type=Path, nargs="+", required=True)
    ap.add_argument("--cutile", type=Path, nargs="+", required=True)
    ap.add_argument("--out-triton", type=Path, required=True)
    ap.add_argument("--out-cutile", type=Path, required=True)
    args = ap.parse_args()

    triton_merged = {}
    for p in args.triton:
        d = json.loads(p.read_text())
        triton_merged = merge(triton_merged, d)
    cutile_merged = {}
    for p in args.cutile:
        d = json.loads(p.read_text())
        cutile_merged = merge(cutile_merged, d)

    args.out_triton.write_text(json.dumps(triton_merged, indent=2))
    args.out_cutile.write_text(json.dumps(cutile_merged, indent=2))
    print(f"triton dirs: {len([k for k in triton_merged if not k.startswith('__')])}")
    print(f"cutile dirs: {len([k for k in cutile_merged if not k.startswith('__')])}")


if __name__ == "__main__":
    main()
