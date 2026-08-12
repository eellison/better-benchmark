#!/usr/bin/env python3
"""Compare ORACLE-floor vs MIN-floor model projections.

Reads the two coverage-gated projection JSONs (lists of per-model reports from
model_graph_accounting.py) and reports:
  - how many of the ORACLE-floor INCOMPLETE models flip to COMPLETE under
    min-floor (the headline two-floor result),
  - top-15 models by projected fusible time under EACH floor (complete only),
  - the per-model side-by-side table (written to JSON for the README).
"""
import json

BASE = "results/oracle_floors_2026-06-18_v4/model_projections"
ORACLE = f"{BASE}/projections_v4_coveragegated.json"
MINFLOOR = f"{BASE}/projections_v4_minfloor.json"
OUT_JSON = f"{BASE}/two_floor_comparison.json"


def model_id(r):
    """Stable model id = suite/mode/name from model_dir tail, fallback name."""
    md = r.get("model_dir", "")
    parts = [p for p in md.split("/") if p]
    # .../repros/models/<suite>/<mode>/<name>
    if "models" in parts:
        i = parts.index("models")
        tail = parts[i + 1:]
        if tail:
            return "/".join(tail)
    return r["model_name"]


def index(reports):
    out = {}
    for r in reports:
        out[model_id(r)] = r
    return out


def main():
    oracle = index(json.load(open(ORACLE)))
    minf = index(json.load(open(MINFLOOR)))

    common = sorted(set(oracle) & set(minf))
    only_oracle = sorted(set(oracle) - set(minf))
    only_minf = sorted(set(minf) - set(oracle))

    rows = []
    flipped = []          # incomplete under oracle -> complete under min-floor
    still_incomplete = []  # incomplete under both
    complete_both = []
    for mid in common:
        o, m = oracle[mid], minf[mid]
        o_complete = bool(o.get("projection_complete"))
        m_complete = bool(m.get("projection_complete"))
        row = {
            "model": mid,
            "oracle_complete": o_complete,
            "minfloor_complete": m_complete,
            "oracle_us": o.get("fusible_oracle_us_total"),
            "oracle_us_partial": o.get("fusible_oracle_us_total_partial"),
            "minfloor_us": m.get("fusible_oracle_us_total"),
            "minfloor_us_partial": m.get("fusible_oracle_us_total_partial"),
            "oracle_unpriced_occ": o.get("unpriced_occurrences"),
            "minfloor_unpriced_occ": m.get("unpriced_occurrences"),
            "total_occ": m.get("total_occurrences"),
        }
        rows.append(row)
        if not o_complete and m_complete:
            flipped.append(row)
        elif not o_complete and not m_complete:
            still_incomplete.append(row)
        elif o_complete and m_complete:
            complete_both.append(row)

    oracle_incomplete = [r for r in rows if not r["oracle_complete"]]
    minf_incomplete = [r for r in rows if not r["minfloor_complete"]]

    print("=" * 70)
    print("TWO-FLOOR PROJECTION COMPARISON")
    print("=" * 70)
    print(f"models in oracle-floor projection : {len(oracle)}")
    print(f"models in min-floor  projection   : {len(minf)}")
    print(f"common models                     : {len(common)}")
    if only_oracle:
        print(f"  only in oracle: {only_oracle}")
    if only_minf:
        print(f"  only in minfloor: {only_minf}")
    print()
    o_comp = sum(1 for r in rows if r["oracle_complete"])
    m_comp = sum(1 for r in rows if r["minfloor_complete"])
    print(f"ORACLE-floor: {o_comp} COMPLETE / {len(oracle_incomplete)} INCOMPLETE")
    print(f"MIN-floor:    {m_comp} COMPLETE / {len(minf_incomplete)} INCOMPLETE")
    print()
    print(f">>> FLIPPED incomplete->complete under min-floor: {len(flipped)}")
    print(f"    still incomplete under both: {len(still_incomplete)}")
    print(f"    complete under both:         {len(complete_both)}")
    print()

    def top15(rows, key, label):
        comp = [r for r in rows if r.get(key) is not None]
        comp.sort(key=lambda r: -r[key])
        print(f"--- TOP 15 by {label} (complete projections only) ---")
        print(f"  {'model':46}{'us':>12}{'occ':>8}")
        for r in comp[:15]:
            print(f"  {r['model']:46}{r[key]:12.1f}{r.get('total_occ') or 0:8d}")
        print()
        return comp[:15]

    top_oracle = top15(rows, "oracle_us", "ORACLE-floor projected fusible us")
    top_minf = top15(rows, "minfloor_us", "MIN-floor projected fusible us")

    # Largest absolute reduction oracle_us -> minfloor_us among models complete
    # under BOTH (apples to apples).
    both = [r for r in rows
            if r["oracle_us"] is not None and r["minfloor_us"] is not None]
    for r in both:
        r["_drop_us"] = round(r["oracle_us"] - r["minfloor_us"], 2)
    both.sort(key=lambda r: -r["_drop_us"])
    print("--- Largest floor DROP (oracle_us - minfloor_us), complete-in-both ---")
    print(f"  {'model':46}{'oracle':>10}{'minfloor':>10}{'drop':>10}")
    for r in both[:15]:
        print(f"  {r['model']:46}{r['oracle_us']:10.1f}{r['minfloor_us']:10.1f}"
              f"{r['_drop_us']:10.1f}")
    print()

    out = {
        "counts": {
            "oracle_models": len(oracle),
            "minfloor_models": len(minf),
            "common": len(common),
            "oracle_complete": o_comp,
            "oracle_incomplete": len(oracle_incomplete),
            "minfloor_complete": m_comp,
            "minfloor_incomplete": len(minf_incomplete),
            "flipped_incomplete_to_complete": len(flipped),
            "still_incomplete_both": len(still_incomplete),
            "complete_both": len(complete_both),
        },
        "flipped_models": sorted(flipped, key=lambda r: -(r["minfloor_us"] or 0)),
        "still_incomplete_models": still_incomplete,
        "top15_oracle_floor": top_oracle,
        "top15_min_floor": top_minf,
        "largest_drop": both[:30],
        "all_rows": sorted(rows, key=lambda r: -(r["minfloor_us"] or r["minfloor_us_partial"] or 0)),
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"WROTE {OUT_JSON}")


if __name__ == "__main__":
    main()
