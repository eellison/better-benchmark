#!/usr/bin/env python3
"""Segment a per-model A/B rollup by suite/mode, in the right units.

Reads a ``perf_ab_rollup.py --json`` output (or any
``{"per_model": {"<suite>/<mode>/<model>": pct}}`` dict) and reports the
per-model-e2e geomean / median / n overall, per suite, and per suite+mode, plus
the top/bottom model movers -- so "small overall, but +13% on timm inference,
+6.5% on HF inference" is one glance.

genai entries are EXCLUDED from the model/suite geomeans by default: a genai
entry is a single microbenchmark kernel, not a model, so folding it into a
per-model geomean inflates it (its whole "model" is the kernel under test).
They are listed separately under "genai (single-kernel workloads)". Pass
--include-genai to fold them in anyway.

Usage:
    python scripts/perf_ab_rollup.py --base B.json --head H.json --json > rollup.json
    python scripts/segment_perf.py rollup.json
    python scripts/segment_perf.py rollup.json --include-genai --top 12
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def geomean_pct(pcts):
    """Geomean of pct improvements: pct -> speedup ratio -> geomean -> pct.

    Drops points where (1 - pct/100) <= 0 (a >=100% "improvement" would make the
    ratio non-positive); those are reported via the dropped count so the caller
    can see when it happens.
    """
    finite = [p for p in pcts if p is not None]
    ratios = [1.0 / (1.0 - p / 100.0) for p in finite if (1.0 - p / 100.0) > 0]
    if len(ratios) != len(finite):
        import sys
        print(f"WARNING: {len(finite) - len(ratios)} model(s) with pct>=100% dropped "
              f"from geomean (over {len(ratios)}, not {len(finite)})", file=sys.stderr)
    if not ratios:
        return None
    return (statistics.geometric_mean(ratios) - 1.0) * 100.0


def seg_stats(pcts):
    pcts = [p for p in pcts if p is not None]
    if not pcts:
        return None
    return dict(n=len(pcts), geomean=geomean_pct(pcts),
                median=statistics.median(pcts), mean=statistics.mean(pcts))


def _fmt(label, st, width=16):
    if st is None:
        return f"    {label:{width}s} (no priced models)"
    g = f"{st['geomean']:+6.2f}%" if st["geomean"] is not None else "   n/a"
    return f"    {label:{width}s} geomean {g}  median {st['median']:+6.2f}%  n={st['n']}"


def main():
    ap = argparse.ArgumentParser(
        description="Segment a per-model A/B rollup by suite/mode (genai excluded "
                    "from model geomeans by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("rollup", help="perf_ab_rollup --json output (or a {per_model:{...}} JSON)")
    ap.add_argument("--include-genai", action="store_true",
                    help="Fold genai microbenchmarks into the model/suite geomeans "
                         "(off by default -- genai are single kernels, not models).")
    ap.add_argument("--top", type=int, default=8, help="How many top/bottom model movers to list")
    args = ap.parse_args()

    d = json.load(open(args.rollup))
    pm_all = {k: v for k, v in (d.get("per_model", d)).items() if v is not None}

    is_genai = lambda k: "genai" in k.split("/")[0]
    genai = {k: v for k, v in pm_all.items() if is_genai(k)}
    pm = pm_all if args.include_genai else {k: v for k, v in pm_all.items() if not is_genai(k)}

    by_suite = defaultdict(list)
    by_suite_mode = defaultdict(list)
    for k, v in pm.items():
        parts = k.split("/")
        if len(parts) < 3:
            continue
        by_suite[parts[0]].append(v)
        by_suite_mode[f"{parts[0]}/{parts[1]}"].append(v)

    scope = "genai INCLUDED" if args.include_genai else "genai excluded from model geomeans"
    print("=" * 70)
    print(f"SEGMENTED PERF   ({len(pm)} models, {scope})   [+ = faster]")
    print("=" * 70)
    print(_fmt("OVERALL", seg_stats(list(pm.values()))))
    print("  -- by suite --")
    for s in sorted(by_suite):
        print(_fmt(s, seg_stats(by_suite[s]), width=14))
    print("  -- by suite/mode --")
    for sm in sorted(by_suite_mode):
        print(_fmt(sm, seg_stats(by_suite_mode[sm])))

    if genai and not args.include_genai:
        print("  -- genai (single-kernel workloads; NOT in the model geomean above) --")
        for k, v in sorted(genai.items(), key=lambda kv: -kv[1]):
            print(f"    {v:+7.2f}%  {k}")

    if args.top:
        movers = sorted(pm.items(), key=lambda kv: -kv[1])
        print(f"  -- top {args.top} model movers --")
        for k, v in movers[:args.top]:
            print(f"    {v:+7.2f}%  {k}")
        print(f"  -- bottom {min(args.top, len(movers))} --")
        for k, v in movers[-args.top:][::-1]:
            print(f"    {v:+7.2f}%  {k}")


if __name__ == "__main__":
    main()
