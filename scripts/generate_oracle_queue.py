#!/usr/bin/env python3
"""Generate the oracle migration work queue from the recaptured corpus.

Walks the NEW corpus canonical dirs and produces one CSV row per pattern dir,
ordered by total_occurrences descending. Matches each new pattern against the
old corpus by normalized origin_ops multiset to find oracle candidates.

Usage:
    python scripts/generate_oracle_queue.py

Output:
    investigation_results/oracle_migration_queue.csv
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_CORPUS = Path("/tmp/scratch_space/recapture_corpus/repros/canonical")
OLD_CORPUS = ROOT / "repros" / "canonical"
OLD_QUEUE_CSV = ROOT / "investigation_results" / "oracle_kernel_work_queue.csv"
OUTPUT_CSV = ROOT / "investigation_results" / "oracle_migration_queue.csv"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_op(op: str) -> str:
    """Normalize an op name for multiset comparison.

    - aten.reshape.default -> aten.view.default
    - prims.convert_element_type.default -> aten.convert_element_type.default
      (strip prims. vs aten. prefix difference for convert_element_type only)
    """
    if op == "aten.reshape.default":
        return "aten.view.default"
    # Normalize convert_element_type across prims/aten prefix
    if "convert_element_type" in op:
        return "aten.convert_element_type.default"
    return op


def normalized_multiset(origin_ops: list[str]) -> tuple[str, ...]:
    """Return a sorted tuple of normalized ops (usable as dict key)."""
    return tuple(sorted(normalize_op(op) for op in origin_ops))


# ---------------------------------------------------------------------------
# Load old corpus index
# ---------------------------------------------------------------------------

def load_old_corpus_index() -> dict[tuple[str, ...], list[dict]]:
    """Build multiset -> list of old dir info mappings."""
    index: dict[tuple[str, ...], list[dict]] = {}
    if not OLD_CORPUS.exists():
        return index
    for d in sorted(OLD_CORPUS.iterdir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        origin_ops = meta.get("origin_ops", [])
        if not origin_ops:
            continue
        ms = normalized_multiset(origin_ops)
        has_oracle = any(f.name.startswith("oracle_") and f.suffix == ".py"
                        for f in d.iterdir())
        entry = {"dir_name": d.name, "has_oracle": has_oracle, "meta": meta}
        index.setdefault(ms, []).append(entry)
    return index


def load_old_queue_families() -> dict[str, str]:
    """Load repro_id -> family mapping from the old work queue CSV."""
    families: dict[str, str] = {}
    if not OLD_QUEUE_CSV.exists():
        return families
    with open(OLD_QUEUE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repro_id = row.get("repro_id", "").strip()
            family = row.get("family", "").strip()
            if repro_id and family:
                families[repro_id] = family
    return families


# ---------------------------------------------------------------------------
# Near-match helpers
# ---------------------------------------------------------------------------

def multiset_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    """Symmetric difference size between two multisets (as sorted tuples)."""
    ca = Counter(a)
    cb = Counter(b)
    diff = 0
    all_keys = set(ca.keys()) | set(cb.keys())
    for k in all_keys:
        diff += abs(ca.get(k, 0) - cb.get(k, 0))
    return diff


# ---------------------------------------------------------------------------
# Family derivation
# ---------------------------------------------------------------------------

def derive_family(kind: str, origin_ops: list[str]) -> str:
    """Derive a coarse family from kind + dominant origin_ops."""
    ops_str = " ".join(origin_ops)

    # Scatter/gather patterns
    if any(kw in ops_str for kw in ("scatter", "index_put", "index_select",
                                     "gather", "index_copy")):
        return "scatter_gather"

    # Softmax-like: amax + exp + sum present
    has_amax = "amax" in ops_str
    has_exp = "exp" in ops_str
    has_sum = "sum" in ops_str
    if has_amax and has_exp and has_sum:
        return "reduction_softmax_like"

    # Var/mean reductions
    if "var_mean" in ops_str or ("var" in ops_str and "mean" in ops_str):
        return "reduction_var_mean"

    # General reduction
    if kind == "reduction":
        return "reduction_general"

    # Pointwise
    if kind == "pointwise":
        return "pointwise"

    return "unclassified"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading old corpus index...")
    old_index = load_old_corpus_index()
    old_families = load_old_queue_families()
    print(f"  Old corpus: {sum(len(v) for v in old_index.values())} dirs indexed "
          f"across {len(old_index)} unique multisets")
    print(f"  Old queue families: {len(old_families)} entries")

    # Collect all old multisets for near-match scanning
    old_multisets = list(old_index.keys())

    print(f"\nScanning new corpus: {NEW_CORPUS}")
    rows: list[dict] = []

    for pattern_dir in sorted(NEW_CORPUS.iterdir()):
        meta_path = pattern_dir / "meta.json"
        shapes_path = pattern_dir / "shapes.json"
        if not meta_path.exists():
            continue

        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        # Extract fields from meta
        pattern_hash = meta.get("pattern_hash", "")
        kind = meta.get("kind", "")
        n_models = meta.get("n_models", 0)
        origin_ops = meta.get("origin_ops", [])

        # Extract points info from shapes.json
        n_points = 0
        total_occurrences = 0
        if shapes_path.exists():
            try:
                shapes = json.loads(shapes_path.read_text())
                points = shapes.get("points", [])
                n_points = len(points)
                for pt in points:
                    models_info = pt.get("models", {})
                    for model_data in models_info.values():
                        total_occurrences += model_data.get("occurrences", 0)
            except (json.JSONDecodeError, OSError):
                pass

        # Match against old corpus
        new_ms = normalized_multiset(origin_ops)
        exact_matches = old_index.get(new_ms, [])

        # Sort: prefer dirs with oracles
        exact_matches_sorted = sorted(exact_matches,
                                      key=lambda e: (not e["has_oracle"], e["dir_name"]))

        candidates: list[str] = []
        near_candidates: list[str] = []

        if exact_matches_sorted:
            candidates = [e["dir_name"] for e in exact_matches_sorted[:3]]
        else:
            # Try near-matches (distance <= 2)
            near_hits: list[tuple[int, dict]] = []
            for old_ms in old_multisets:
                dist = multiset_distance(new_ms, old_ms)
                if dist <= 2:
                    for entry in old_index[old_ms]:
                        near_hits.append((dist, entry))
            # Sort by distance then oracle presence
            near_hits.sort(key=lambda x: (x[0], not x[1]["has_oracle"], x[1]["dir_name"]))
            near_candidates = ["~" + h[1]["dir_name"] for h in near_hits[:3]]

        old_oracle_candidates_list = candidates if candidates else near_candidates
        old_oracle_candidates = ";".join(old_oracle_candidates_list)

        # Determine family
        family = ""
        if candidates:
            # Use family from old queue for the best candidate
            for cand in candidates:
                if cand in old_families:
                    family = old_families[cand]
                    break
        if not family:
            # Also check near candidates (strip ~ prefix)
            for cand in near_candidates:
                bare = cand.lstrip("~")
                if bare in old_families:
                    family = old_families[bare]
                    break
        if not family:
            family = derive_family(kind, origin_ops)

        rows.append({
            "queue_rank": 0,  # filled after sorting
            "repro_dir": pattern_dir.name,
            "pattern_hash": pattern_hash,
            "kind": kind,
            "family": family,
            "n_points": n_points,
            "n_models": n_models,
            "total_occurrences": total_occurrences,
            "old_oracle_candidates": old_oracle_candidates,
            "status": "unclaimed",
            "owner": "",
            "notes": "",
        })

    # Sort by total_occurrences descending (stable sort preserves dir name order for ties)
    rows.sort(key=lambda r: (-r["total_occurrences"], r["repro_dir"]))

    # Assign queue ranks
    for i, row in enumerate(rows, 1):
        row["queue_rank"] = i

    # Write CSV atomically
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(OUTPUT_CSV.parent), suffix=".csv.tmp", prefix="oracle_mig_"
    )
    os.close(fd)
    try:
        fieldnames = [
            "queue_rank", "repro_dir", "pattern_hash", "kind", "family",
            "n_points", "n_models", "total_occurrences", "old_oracle_candidates",
            "status", "owner", "notes",
        ]
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.rename(tmp_path, str(OUTPUT_CSV))
    except BaseException:
        # Clean up on failure
        if Path(tmp_path).exists():
            os.unlink(tmp_path)
        raise

    # Summary
    n_exact = sum(1 for r in rows if r["old_oracle_candidates"]
                  and not r["old_oracle_candidates"].startswith("~"))
    n_near = sum(1 for r in rows if r["old_oracle_candidates"]
                 and r["old_oracle_candidates"].startswith("~"))
    n_none = sum(1 for r in rows if not r["old_oracle_candidates"])

    print(f"\n{'='*60}")
    print(f"Oracle Migration Queue Summary")
    print(f"{'='*60}")
    print(f"Total rows:              {len(rows)}")
    print(f"Rows with exact match:   {n_exact}")
    print(f"Rows with near match:    {n_near}")
    print(f"Rows with no candidate:  {n_none}")
    print(f"\nFamily histogram (top 10):")

    family_counts = Counter(r["family"] for r in rows)
    for fam, count in family_counts.most_common(10):
        print(f"  {fam:45s} {count:5d}")

    print(f"\nOutput: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
