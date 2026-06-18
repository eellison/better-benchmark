#!/usr/bin/env python3
"""compute_ab.py — DERIVE the A/B (perf-branch model improvement) from two
per-commit measurement-sets. The A/B is NOT a stored artifact: it is computed
on demand from two `kernels.json` files, each annotated with a single
`pytorch_commit`.

A "result" in results/b200/ is a MEASUREMENT at one pytorch commit ("the kernels
at commit C on hardware H ran in T us"). The improvement of the perf branch over
its baseline is a DERIVATION over two such measurement-sets — exactly what this
script does:

  improvement(model) = Σ (baseline_us[k] - branch_us[k]) * occ[k]
                       ───────────────────────────────────────────
                                    model_baseline_us

shape-matched per (pattern_hash, shape_hash). This reuses the EXACT per-model
join arithmetic of results/perf_ab/rollup.py (the established model-accounting
fusible join + coord_descent_us preference + rep-median fallback + extern
denominator from the occurrence sidecars).

EXTERN CORRECTNESS: the SAME extern numbers are used for BOTH arms. Externs
(cuBLAS/cuDNN/flash) are inductor-invariant — torch.compile does not touch them —
so each sidecar's `extern[].baseline_us` is the single shared extern price (the
sidecars are shared across commits, NOT per-arm). It enters only the denominator
(model_baseline_us); it CANCELS in the numerator, so each per-model
%-improvement is independent of the extern term entirely. This mirrors rollup.py,
which adds the identical ext_total to both model_baseline_us and model_branch_us.

genai entries are MICROBENCHMARKS (raw kernel-level wins on directly-targeted
ops) and are EXCLUDED from the headline aggregate; they are reported separately.

SELF-CONTAINED: the only inputs read are under this script's directory
(results/b200/): the two per-commit kernels.json and the local occurrences/
sidecars. Copy results/b200/ anywhere and `python compute_ab.py` still works.

Usage:
  python compute_ab.py [BASELINE_DIR] [BRANCH_DIR]
  # defaults: results/b200/244fdb379d11  results/b200/daa79cd25ca

Each DIR must contain a kernels.json with _metadata.pytorch_commit. The script
prints the headline (median / mean / geomean, win/regress/flat counts) and cites
the two pytorch commits it derived from.

NB: geomean (of per-model speedup ratios baseline/branch) is computed on the fly
here; it is NOT yet emitted by a committed metric script (new 2026-06-18).
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Self-contained: every input lives under this script's own directory
# (results/b200/), so the snapshot can be copied anywhere and still derive the
# A/B. OCCDIR is the LOCAL occurrence-sidecar copy (occurrence counts + extern
# prices); the two kernels.json are the per-commit measurement-sets.
HERE = Path(__file__).resolve().parent
OCCDIR = HERE / "occurrences"

DEFAULT_BASELINE = HERE / "244fdb379d11"
DEFAULT_BRANCH = HERE / "daa79cd25ca"


# ---- the rollup.py fusible join (reused verbatim) ---------------------------

def _phash(repro_path: str) -> str:
    # repros/canonical/<family>_<phash>/repro.py -> dir name last "_" segment
    return Path(repro_path).parent.name.rsplit("_", 1)[-1]


def load_arm(path: Path):
    """(exact[(phash,shape)]->us, rep[phash]->median_us). coord_descent pref,
    compiled_us fallback. EXACTLY rollup.py:load_arm / reconcile.py:load_arm."""
    data = json.loads(path.read_text())
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
        return v
    return rep.get(ph)  # pattern representative-median fallback, else None


def per_model_improvement(base_kernels: Path, branch_kernels: Path):
    """Replicate rollup.py: per-model pct_improvement, shape-matched on
    (pattern_hash, shape_hash) across the two measurement-sets."""
    be, brp = load_arm(base_kernels)
    bre, brrp = load_arm(branch_kernels)
    models = {}
    for sc in sorted(OCCDIR.glob("*.json")):
        rec = json.loads(sc.read_text())
        suite, mode, model = rec["suite"], rec["mode"], rec["model"]
        fus_base = fus_branch = delta = ext_total = 0.0
        unmatched = []
        for ph, shapes in rec["fusible"].items():
            for sh, cnt in shapes.items():
                b = fusible_us(be, brp, ph, sh)
                r = fusible_us(bre, brrp, ph, sh)
                if b is None or r is None:
                    unmatched.append(f"{ph}/{sh}x{cnt}")
                    continue
                fus_base += b * cnt
                fus_branch += r * cnt
                delta += (b - r) * cnt
        for e in rec.get("extern", []):
            eu = e.get("baseline_us")
            if eu is not None:
                ext_total += eu * e["count"]  # cancels in numerator; denom only
        mb = fus_base + ext_total
        pct = (delta / mb * 100.0) if mb > 0 else None
        models[f"{suite}/{mode}/{model}"] = pct
    return models


# ---- the A/B derivation ------------------------------------------------------

def pytorch_commit_of(commit_dir: Path) -> str:
    meta = json.loads((commit_dir / "kernels.json").read_text())["_metadata"]
    sha = meta.get("pytorch_commit")
    if not sha:
        raise SystemExit(f"{commit_dir}/kernels.json has no _metadata.pytorch_commit")
    return sha


def summarize(pcts):
    """median / mean / geomean(speedup) + win/regress/flat over a pct list.
    pct_improvement p>0 == faster. speedup ratio = baseline/branch = 1/(1-p/100).
    geomean is over those ratios; reported as (geo-1)*100 percent."""
    pos = sum(1 for p in pcts if p > 0.05)
    neg = sum(1 for p in pcts if p < -0.05)
    flat = sum(1 for p in pcts if -0.05 <= p <= 0.05)
    ratios = [1.0 / (1.0 - p / 100.0) for p in pcts]
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return {
        "n": len(pcts),
        "median_pct": round(statistics.median(pcts), 3),
        "mean_pct": round(statistics.mean(pcts), 3),
        "geomean_pct": round((geo - 1) * 100.0, 3),
        "improved": pos, "regressed": neg, "flat": flat,
        "min_pct": round(min(pcts), 3), "max_pct": round(max(pcts), 3),
    }


def main():
    baseline_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BASELINE
    branch_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_BRANCH

    base_sha = pytorch_commit_of(baseline_dir)
    branch_sha = pytorch_commit_of(branch_dir)

    models = per_model_improvement(baseline_dir / "kernels.json",
                                   branch_dir / "kernels.json")

    # split models into the headline cut (non-genai) and the genai
    # microbenchmark cut. genai is NEVER folded into the model aggregate.
    model_rows = sorted(  # (name, pct) for real models, worst-first
        ((m, p) for m, p in models.items() if p is not None and "genai" not in m),
        key=lambda kv: kv[1])
    micro_rows = sorted(  # genai microbenchmarks, best-first
        ((m, p) for m, p in models.items() if p is not None and "genai" in m),
        key=lambda kv: kv[1], reverse=True)

    excl = [p for _, p in model_rows]
    incl = [p for p in models.values() if p is not None]
    s_excl, s_incl = summarize(excl), summarize(incl)

    print("=" * 76)
    print("DELIVERABLE 1 — perf-branch model improvement (A/B), DERIVED on the fly")
    print("=" * 76)
    print(f"  baseline pytorch commit : {base_sha}")
    print(f"            measurement-set: {baseline_dir}/kernels.json")
    print(f"  branch   pytorch commit : {branch_sha}")
    print(f"            measurement-set: {branch_dir}/kernels.json")
    print(f"  extern denominator      : occurrences/ sidecars (LOCAL); SAME extern "
          f"both arms, cancels in numerator")
    print(f"  join                    : (pattern_hash, shape_hash) shape-match; "
          f"rollup.py fusible arithmetic")
    print()
    s = s_excl
    print(f"  [genai-EXCLUDED (HEADLINE)]  n={s['n']}")
    print(f"     median +{s['median_pct']:.3f}%   mean +{s['mean_pct']:.3f}%   "
          f"geomean +{s['geomean_pct']:.3f}%  (geomean computed on the fly, "
          f"not yet in a committed metric script)")
    print(f"     improved {s['improved']} / regressed {s['regressed']} / "
          f"flat {s['flat']}   (range {s['min_pct']:+.1f}%..{s['max_pct']:+.1f}%)")
    print()

    # ---- largest regressions / best improvements (genai-EXCLUDED) -----------
    n_worst = min(8, len(model_rows))
    n_best = min(5, len(model_rows))
    print("-" * 76)
    print(f"  LARGEST REGRESSIONS (genai-excluded, worst {n_worst})")
    print("-" * 76)
    for m, p in model_rows[:n_worst]:
        print(f"     {p:+7.2f}%   {m}")
    print()
    print(f"  BEST IMPROVEMENTS (genai-excluded, top {n_best})")
    print("-" * 76)
    for m, p in reversed(model_rows[-n_best:]):
        print(f"     {p:+7.2f}%   {m}")
    print()

    # ---- microbenchmarks (genai) — SEPARATE, never in the headline ----------
    print("=" * 76)
    print("MICROBENCHMARKS (raw kernel-level wins on directly-targeted ops;")
    print("NOT model-representative — EXCLUDED from headline mean/median/geomean)")
    print("=" * 76)
    for m, p in micro_rows:
        print(f"     {p:+7.2f}%   {m}")
    print()
    print(f"  [genai-included reference only — the MICROBENCHMARK view]  n={s_incl['n']}")
    print(f"     median +{s_incl['median_pct']:.3f}%   mean +{s_incl['mean_pct']:.3f}%   "
          f"geomean +{s_incl['geomean_pct']:.3f}%")
    print(f"     (this line folds the {len(micro_rows)} microbenchmarks back in and is "
          f"NOT the headline)")
    print()
    print("  To reproduce: this is a DERIVATION, not a stored result. Re-run with the")
    print("  two commit dirs above; the per-commit kernels.json + local occurrences/")
    print("  are the only stored inputs (self-contained).")
    return s_excl


if __name__ == "__main__":
    main()
