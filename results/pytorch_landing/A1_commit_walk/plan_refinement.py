#!/usr/bin/env python3
"""After the coarse pass, decide which SEGMENTS moved perf and emit the
fill-in commit list to bench (the commits strictly BETWEEN two benched
checkpoints whose segment |delta_geomean| exceeds the noise floor).

A flat segment (|delta| < NOISE) contributes ~0 per-commit -> no fill needed;
its commits are reported as noise-floor in bulk. A moving segment gets ALL its
interior commits queued so each commit inside it is individually attributed.

Reuses analyze_walk's per-segment deltas (walk_analysis.json) + walk_list.json.
"""
import json
from pathlib import Path

OUT = Path("/tmp/scratch_space/better_benchmark/results/pytorch_landing/A1_commit_walk")
NOISE = 0.82  # pp; segment movement threshold (per-model A/A floor)

def main():
    walk = json.loads((OUT / "walk_list.json").read_text())["walk"]
    analysis = json.loads((OUT / "walk_analysis.json").read_text())
    rows = analysis["per_segment_deltas_in_order"]

    by_order = {r["order"]: r for r in walk}
    benched_orders = [-1] + [r["order"] for r in rows]  # -1 = base

    fill = []          # interior commits to bench in moving segments
    moving_segs = []
    flat_segs = []
    for r in rows:
        o = r["order"]
        seg_n = r["segment_commits"]
        d = abs(r["delta_geomean"] or 0.0)
        # interior orders strictly between previous benched and this one
        prev = o - seg_n
        interior = [k for k in range(prev + 1, o)]  # exclude endpoint o (already benched)
        if seg_n == 1:
            # single-commit segment already fully attributed
            (moving_segs if d >= NOISE else flat_segs).append((o, o, d, []))
            continue
        if d >= NOISE and interior:
            moving_segs.append((prev + 1, o, d, interior))
            fill.extend(interior)
        else:
            flat_segs.append((prev + 1, o, d, interior))

    fill_full = [by_order[k]["sha_full"] for k in sorted(set(fill)) if k in by_order]
    (OUT / "refine_states.txt").write_text("\n".join(fill_full) + ("\n" if fill_full else ""))

    print(f"NOISE threshold: ±{NOISE}pp geomean")
    print(f"\nMOVING segments ({len(moving_segs)}) -> {len(fill_full)} interior commits to fill:")
    for lo, hi, d, interior in sorted(moving_segs, key=lambda x: -x[2]):
        names = ",".join(by_order[hi]["a0_features"][0].split(":")[0]
                         for _ in [0] if by_order[hi]["a0_features"])
        print(f"  seg orders {lo}..{hi}  Δgeo={d:+.3f}pp  interior={interior}")
    print(f"\nFLAT segments ({len(flat_segs)}) -> bulk noise-floor (no fill):")
    for lo, hi, d, interior in flat_segs:
        print(f"  seg orders {lo}..{hi}  Δgeo={d:+.3f}pp  ({len(interior)} interior commits)")
    print(f"\nrefine_states.txt: {len(fill_full)} commits "
          f"(~{len(fill_full)*36/60:.1f}h additional GPU at 36min/state)")

if __name__ == "__main__":
    main()
