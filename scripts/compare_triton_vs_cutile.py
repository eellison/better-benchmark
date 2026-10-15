"""Compare Triton and cuTile oracle sweep results.

Both sweeps produced by:
    python scripts/bench_parallel.py --oracles <root> --output <out>.json

Emits per-(dir, shape_hash) rows with:
  * inductor / compile_us   (from either sweep — they should agree)
  * triton_oracle_us
  * cutile_oracle_us
  * triton_status / cutile_status
  * ratio_cutile_over_triton  = cutile_us / triton_us

Also computes summary stats.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


_INVALID_STATUSES = frozenset({
    "UNVERIFIED_NUMERICS",
    "INVALID_CUDAGRAPH_WARNING",
    "NUMERICS_WORSE_THAN_COMPILED",
    "NO_ORACLE_FOR_SHAPE",
})


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_shape_rows(payload: dict):
    """Iterate (dir_name, shape_hash, entry) rows from a bench_parallel oracle
    output payload. Entry is one dict {oracle_us, compile_us, status, ...}.

    Handles both top-level valid dirs (which carry ``points_by_shape``) and
    dirs moved to ``__failures__`` (all-bad-oracle case), which only carry
    ``points`` keyed by ``<model>_<shape_hash>``.
    """
    for dir_name, entry in payload.items():
        if dir_name == "__failures__":
            for fdir, fentry in (entry or {}).items():
                for label, row in (fentry.get("points") or {}).items():
                    shape_hash = label.rsplit("_", 1)[-1]
                    yield fdir, shape_hash, row
            continue
        if dir_name.startswith("__") or dir_name == "_metadata":
            continue
        if not isinstance(entry, dict):
            continue
        by_shape = entry.get("points_by_shape") or {}
        if by_shape:
            for shape_hash, row in by_shape.items():
                yield dir_name, shape_hash, row
        else:
            # Fallback: some legacy dirs may only have ``points``.
            for label, row in (entry.get("points") or {}).items():
                shape_hash = label.rsplit("_", 1)[-1]
                yield dir_name, shape_hash, row


def _valid(row: dict) -> bool:
    return (
        row.get("status") not in _INVALID_STATUSES
        and isinstance(row.get("oracle_us"), (int, float))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triton", type=Path, required=True,
                    help="triton oracle sweep JSON")
    ap.add_argument("--cutile", type=Path, required=True,
                    help="cutile oracle sweep JSON")
    ap.add_argument("--out-json", type=Path, required=False,
                    help="write per-row comparison to a JSON file")
    ap.add_argument("--out-csv", type=Path, required=False,
                    help="write per-row comparison to a CSV file")
    ap.add_argument("--print-top", type=int, default=25,
                    help="how many top rows to print")
    args = ap.parse_args()

    triton = _load(args.triton)
    cutile = _load(args.cutile)

    triton_rows = {(d, s): r for d, s, r in _iter_shape_rows(triton)}
    cutile_rows = {(d, s): r for d, s, r in _iter_shape_rows(cutile)}

    all_keys = sorted(set(triton_rows.keys()) | set(cutile_rows.keys()))

    rows = []
    for key in all_keys:
        dir_name, shape_hash = key
        t = triton_rows.get(key, {})
        c = cutile_rows.get(key, {})
        row = {
            "dir": dir_name,
            "shape_hash": shape_hash,
            "triton_us": t.get("oracle_us"),
            "cutile_us": c.get("oracle_us"),
            "triton_status": t.get("status"),
            "cutile_status": c.get("status"),
            "compile_us_triton": t.get("compile_us"),
            "compile_us_cutile": c.get("compile_us"),
            "triton_ratio": t.get("ratio"),  # compile / triton oracle
            "cutile_ratio": c.get("ratio"),  # compile / cutile oracle
        }
        # cutile / triton (>1 => cutile slower than triton)
        if _valid(t) and _valid(c):
            row["cutile_over_triton"] = row["cutile_us"] / row["triton_us"]
        rows.append(row)

    # -- Summary stats
    both_valid = [r for r in rows if r.get("cutile_over_triton") is not None]
    n_all = len(rows)
    n_triton_valid = sum(1 for r in rows if _valid(triton_rows.get((r["dir"], r["shape_hash"]), {})))
    n_cutile_valid = sum(1 for r in rows if _valid(cutile_rows.get((r["dir"], r["shape_hash"]), {})))
    n_both_valid = len(both_valid)
    cutile_wins = sum(1 for r in both_valid if r["cutile_over_triton"] < 1.0)
    triton_wins = sum(1 for r in both_valid if r["cutile_over_triton"] > 1.0)
    ties = n_both_valid - cutile_wins - triton_wins
    ratios = [r["cutile_over_triton"] for r in both_valid]
    geomean = math.exp(sum(math.log(x) for x in ratios) / len(ratios)) if ratios else float("nan")
    median = statistics.median(ratios) if ratios else float("nan")

    # cuTile status breakdown
    from collections import Counter
    cutile_status_counts = Counter(r["cutile_status"] for r in rows)
    triton_status_counts = Counter(r["triton_status"] for r in rows)

    print()
    print("=" * 70)
    print(f"Triton vs cuTile comparison")
    print(f"  Triton sweep:  {args.triton}")
    print(f"  cuTile sweep:  {args.cutile}")
    print("=" * 70)
    print(f"Total unique (dir, shape) points:     {n_all}")
    print(f"  Valid Triton oracle timing:         {n_triton_valid}")
    print(f"  Valid cuTile oracle timing:         {n_cutile_valid}")
    print(f"  Valid on BOTH (comparable):         {n_both_valid}")
    print()
    print("On the comparable subset:")
    print(f"  cuTile faster than Triton:          {cutile_wins}")
    print(f"  Triton faster than cuTile:          {triton_wins}")
    print(f"  Tied (within measurement noise):    {ties}")
    print(f"  Geomean cuTile / Triton:            {geomean:.3f}")
    print(f"  Median cuTile / Triton:             {median:.3f}")
    print()
    print("Triton status breakdown:")
    for s, n in sorted(triton_status_counts.items(),
                       key=lambda kv: (-kv[1], kv[0] or "")):
        print(f"  {s or '(missing)':<32} {n:>6}")
    print()
    print("cuTile status breakdown:")
    for s, n in sorted(cutile_status_counts.items(),
                       key=lambda kv: (-kv[1], kv[0] or "")):
        print(f"  {s or '(missing)':<32} {n:>6}")
    print()

    if args.print_top and both_valid:
        top = sorted(both_valid, key=lambda r: r["cutile_over_triton"])[: args.print_top]
        bot = sorted(both_valid, key=lambda r: r["cutile_over_triton"])[-args.print_top:]

        print(f"Top {args.print_top} rows where cuTile beats Triton the most:")
        print(f"  {'dir':<44} {'shape':<10} {'triton_us':>10} {'cutile_us':>10} {'ratio':>8}")
        for r in top:
            print(f"  {r['dir']:<44} {r['shape_hash']:<10} {r['triton_us']:>10.2f} "
                  f"{r['cutile_us']:>10.2f} {r['cutile_over_triton']:>8.3f}")
        print()
        print(f"Top {args.print_top} rows where cuTile loses the most:")
        print(f"  {'dir':<44} {'shape':<10} {'triton_us':>10} {'cutile_us':>10} {'ratio':>8}")
        for r in bot[::-1]:
            print(f"  {r['dir']:<44} {r['shape_hash']:<10} {r['triton_us']:>10.2f} "
                  f"{r['cutile_us']:>10.2f} {r['cutile_over_triton']:>8.3f}")

    if args.out_json:
        args.out_json.write_text(json.dumps({
            "summary": {
                "n_all": n_all,
                "n_triton_valid": n_triton_valid,
                "n_cutile_valid": n_cutile_valid,
                "n_both_valid": n_both_valid,
                "cutile_wins": cutile_wins,
                "triton_wins": triton_wins,
                "ties": ties,
                "geomean_cutile_over_triton": geomean,
                "median_cutile_over_triton": median,
                "cutile_status_counts": dict(cutile_status_counts),
                "triton_status_counts": dict(triton_status_counts),
            },
            "rows": rows,
        }, indent=2))
        print(f"wrote per-row comparison -> {args.out_json}")

    if args.out_csv:
        import csv
        with args.out_csv.open("w", newline="") as fh:
            fields = [
                "dir", "shape_hash", "triton_us", "cutile_us",
                "cutile_over_triton", "triton_status", "cutile_status",
                "compile_us_triton", "compile_us_cutile",
                "triton_ratio", "cutile_ratio",
            ]
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        print(f"wrote CSV -> {args.out_csv}")


if __name__ == "__main__":
    main()
