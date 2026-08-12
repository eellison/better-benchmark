#!/usr/bin/env python3
"""Compare base vs branch bench_parallel outputs per shape point."""
import json, sys, math

def load(path):
    with open(path) as f:
        return json.load(f)

def points(data):
    out = {}
    for k, v in data.items():
        if k.startswith("_") or k == "__summary__" or not isinstance(v, dict):
            continue
        for pt, rec in v.items():
            if isinstance(rec, dict) and rec.get("compiled_us"):
                out[(k, pt)] = rec["compiled_us"]
            elif isinstance(rec, dict):
                # full-graph payloads nest under "default"
                d = rec.get("default") if isinstance(rec.get("default"), dict) else None
                if d and d.get("compiled_us"):
                    out[(k, pt)] = d["compiled_us"]
    return out

def main(base_path, branch_path, label):
    b = points(load(base_path)); a = points(load(branch_path))
    common = sorted(set(b) & set(a))
    print(f"\n== {label}: {len(common)} common points ==")
    print(f"{'repro':42s} {'point':38s} {'base_us':>10s} {'branch_us':>10s} {'speedup':>8s}")
    logs = []
    for k in common:
        sp = b[k] / a[k]
        logs.append(math.log(sp))
        name = k[0].split('/')[-2] if '/' in k[0] else k[0]
        print(f"{name:42s} {k[1]:38s} {b[k]:10.1f} {a[k]:10.1f} {sp:7.2f}x")
    if logs:
        print(f"geomean speedup: {math.exp(sum(logs)/len(logs)):.3f}x  "
              f"min {math.exp(min(logs)):.2f}x  max {math.exp(max(logs)):.2f}x")

if __name__ == "__main__":
    main(*sys.argv[1:4])
