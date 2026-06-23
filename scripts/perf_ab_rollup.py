#!/usr/bin/env python3
"""A/B two corpus sweep JSONs as a shape-matched, per-model-e2e rollup.

Thin orchestration wrapper -- runs NO new timing code. Consumes two corpus
sweep JSONs (as written by scripts/bench_parallel.py) and rolls the per-kernel
deltas up to a per-MODEL end-to-end estimate, reporting the geomean / median /
mean improvement across models (genai excluded by default). Can optionally
print the raw per-kernel A/B table for direction.

CAVEAT -- where the headline comes from:
  Headline the CAREFUL per-model rollup this tool computes (genai-excluded,
  shape-matched, diluted by each model's extern/GEMM/SDPA denominator), NOT the
  raw `bench_report.py --compare` all-points compiled_us diff. That all-points
  diff is compile-side noisy, not shape-matched, and mixes genai scope, so its
  geomean overstates model impact ~2-3x. Use --kernel-table only for DIRECTION
  (which kernel families moved), never for the headline number.

  The per-model occurrence accounting is read pre-baked from --occdir (no GPU,
  no graph re-tracing); the tool degrades gracefully when that dir is absent.
  The rollup arithmetic is inlined (same formula as the A1 commit-walk driver),
  so the tool is self-contained and runs on a clean checkout.

USAGE:
  # Careful per-model-e2e A/B from two existing sweep JSONs (no GPU):
  python scripts/perf_ab_rollup.py --base BASE.json --head HEAD.json
  # Add the raw per-kernel table (direction only), or include genai:
  python scripts/perf_ab_rollup.py --base BASE.json --head HEAD.json --kernel-table
  python scripts/perf_ab_rollup.py --base BASE.json --head HEAD.json --include-genai
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Per-model fusible/extern accounting; under results/ (gitignored from main).
DEFAULT_OCCDIR = ROOT / "results/perf_ab/occurrences"


# ---- rollup arithmetic (inlined; same formula as the A1 walk driver) ----
def _phash(repro_path: str) -> str:
    return Path(repro_path).parent.name.rsplit("_", 1)[-1]

def load_arm(path: Path):
    data = json.loads(Path(path).read_text())
    exact, by_pat = {}, defaultdict(list)
    for key, val in data.items():
        if key.startswith("_") or not isinstance(val, dict):
            continue
        ph = _phash(key)
        for label, m in val.items():
            if not isinstance(m, dict):
                continue
            us = m.get("coord_descent_us")
            if us is None:
                us = m.get("compiled_us")
            if us is None:
                continue
            sh = label.rsplit("_", 1)[-1]
            exact[(ph, sh)] = float(us)
            by_pat[ph].append(float(us))
    rep = {ph: statistics.median(v) for ph, v in by_pat.items() if v}
    return exact, rep

def fusible_us(exact, rep, ph, sh):
    v = exact.get((ph, sh))
    if v is not None:
        return v, "exact"
    v = rep.get(ph)
    if v is not None:
        return v, "rep"
    return None, "unmatched"

def rollup_models(base_path: Path, br_path: Path, occdir: Path):
    """Per-model pct_improvement of br vs base."""
    be, brp = load_arm(base_path)
    bre, brrp = load_arm(br_path)
    models = {}
    for sc in sorted(Path(occdir).glob("*.json")):
        rec = json.loads(sc.read_text())
        suite, mode, model = rec["suite"], rec["mode"], rec["model"]
        fus_base = fus_branch = delta = ext_total = 0.0
        unmatched = []
        for ph, shapes in rec["fusible"].items():
            for sh, cnt in shapes.items():
                b, bk = fusible_us(be, brp, ph, sh)
                r, rk = fusible_us(bre, brrp, ph, sh)
                if b is None or r is None:
                    unmatched.append(f"{ph}/{sh}x{cnt}")
                    continue
                fus_base += b * cnt
                fus_branch += r * cnt
                delta += (b - r) * cnt
        for e in rec.get("extern", []):
            eu = e.get("baseline_us")
            if eu is None:
                continue
            ext_total += eu * e["count"]
        mb = fus_base + ext_total
        pct = (delta / mb * 100.0) if mb > 0 else None
        models[f"{suite}/{mode}/{model}"] = pct
    return models

def cut(models, genai_excl=True):
    pcts = [p for m, p in models.items()
            if p is not None and (("genai" not in m) if genai_excl else True)]
    if not pcts:
        return None
    ratios = [1.0/(1.0 - p/100.0) for p in pcts if (1.0 - p/100.0) > 0]
    geo = statistics.geometric_mean(ratios) if ratios else None
    return dict(
        n=len(pcts),
        median=round(statistics.median(pcts), 4),
        mean=round(statistics.mean(pcts), 4),
        geomean=round((geo-1.0)*100.0, 4) if geo is not None else None,
    )


def careful_rollup(base_json: Path, head_json: Path, occdir: Path,
                   include_genai: bool):
    """Run the rollup. Returns (per_model dict, summaries {genai_excl, genai_incl})."""
    per_model = rollup_models(base_json, head_json, occdir)
    summaries = {
        "genai_excl": cut(per_model, genai_excl=True),
        "genai_incl": cut(per_model, genai_excl=False),
    }
    return per_model, summaries


def _fmt_summary(label: str, s: dict | None) -> str:
    if not s:
        return f"  {label:12s} (no priced models)"
    geo = f"{s['geomean']:+.3f}%" if s.get("geomean") is not None else "n/a"
    return (
        f"  {label:12s} median {s['median']:+.3f}%   "
        f"mean {s['mean']:+.3f}%   geomean {geo}   (n={s['n']})"
    )


def print_careful(per_model: dict, summaries: dict, include_genai: bool,
                  top_n: int, base_json: Path, head_json: Path):
    headline = summaries["genai_incl"] if include_genai else summaries["genai_excl"]
    label = "genai-incl" if include_genai else "genai-excl"

    print("=" * 78)
    print("CAREFUL per-model-e2e A/B  (shape-matched, per-model rollup)")
    print(f"  base: {base_json}")
    print(f"  head: {head_json}")
    print("=" * 78)
    print("HEADLINE (trust this number):")
    print(_fmt_summary(label + " *", headline))
    other = summaries["genai_excl"] if include_genai else summaries["genai_incl"]
    print(_fmt_summary(("genai-excl" if include_genai else "genai-incl"), other))
    print()
    print("  geomean = geometric mean of per-model speedup ratios (1/(1-pct/100)),")
    print("            reported as a percent. median/mean are over per-model pct.")
    print("  Per-model pct = delta_fusible / (fusible_base + extern), shape-matched.")

    if top_n:
        items = [
            (m, p) for m, p in per_model.items()
            if p is not None and (include_genai or "genai" not in m)
        ]
        items.sort(key=lambda x: -x[1])
        print()
        print(f"Top {top_n} per-model improvements (concentration check):")
        for m, p in items[:top_n]:
            print(f"  {p:+7.2f}%  {m}")
        if len(items) > top_n:
            print(f"Bottom {min(top_n, len(items))} (regressions / no-ops):")
            for m, p in items[-top_n:]:
                print(f"  {p:+7.2f}%  {m}")
    print()


def print_kernel_table(base_json: Path, head_json: Path):
    """Shell out to the locked bench_report.py for the raw per-kernel A/B table."""
    print("=" * 78)
    print("RAW per-kernel A/B table  (bench_report.py --compare)  -- DIRECTION ONLY")
    print("=" * 78)
    print("CAVEAT: this is the all-points raw compiled_us diff. compiled_us has")
    print("run-to-run variance on the compile side, this table is NOT genai-")
    print("excluded and NOT shape-weighted. Read it for WHICH KERNEL FAMILIES")
    print("moved (e.g. var_mean/rsqrt batchnorm), never for the headline number.")
    print("The trusted number is the CAREFUL per-model rollup above.")
    print("-" * 78)
    report = ROOT / "scripts" / "bench_report.py"
    proc = subprocess.run(
        [sys.executable, str(report), "--compare", str(base_json), str(head_json)],
        capture_output=True, text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)


def main():
    p = argparse.ArgumentParser(
        description="Careful A/B + per-model-e2e rollup of two corpus sweeps "
                    "(shape-matched, genai-excluded). Reuses the locked tools; "
                    "runs no new timing code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", type=Path, help="BASE sweep JSON (bench_parallel output)")
    p.add_argument("--head", type=Path, help="HEAD sweep JSON (bench_parallel output)")
    p.add_argument("--occdir", type=Path, default=DEFAULT_OCCDIR,
                   help="Per-model occurrence accounting dir "
                        "(default: results/perf_ab/occurrences/)")
    p.add_argument("--include-genai", action="store_true",
                   help="Include genai microbenchmarks (off by default)")
    p.add_argument("--kernel-table", action="store_true",
                   help="Also print the raw per-kernel bench_report A/B table "
                        "(direction only; correctly caveated)")
    p.add_argument("--top", type=int, default=10,
                   help="Show top/bottom N per-model movers (0 to suppress)")
    p.add_argument("--json", action="store_true",
                   help="Emit the per-model dict + summaries as JSON")
    args = p.parse_args()

    if not args.base or not args.head:
        p.error("provide --base and --head sweep JSONs")
    if not args.base.exists():
        p.error(f"base JSON not found: {args.base}")
    if not args.head.exists():
        p.error(f"head JSON not found: {args.head}")
    if not args.occdir.exists():
        print(
            f"[perf_ab_rollup] occurrence dir not found: {args.occdir}\n"
            "  This is the per-model fusible/extern accounting (a perf artifact\n"
            "  under results/, gitignored from `main`). Point --occdir at a\n"
            "  checkout that has results/perf_ab/occurrences/, or regenerate it\n"
            "  with scripts/model_graph_accounting.py over the corpus.",
            file=sys.stderr,
        )
        sys.exit(2)

    per_model, summaries = careful_rollup(
        args.base, args.head, args.occdir, args.include_genai
    )
    if per_model is None:
        sys.exit(2)

    if args.json:
        print(json.dumps({
            "base": str(args.base),
            "head": str(args.head),
            "include_genai": args.include_genai,
            "summaries": summaries,
            "per_model": per_model,
        }, indent=2))
        return

    print_careful(per_model, summaries, args.include_genai, args.top,
                  args.base, args.head)
    if args.kernel_table:
        print_kernel_table(args.base, args.head)


if __name__ == "__main__":
    main()
