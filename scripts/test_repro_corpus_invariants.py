"""Check repository-level repro corpus invariants.

Usage:
    python scripts/test_repro_corpus_invariants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import report_repro_corpus


def _check(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    report = report_repro_corpus.build_report(
        repo_root=ROOT,
        max_examples=20,
    )
    canonical = report["canonical"]
    manifests = report["model_manifest_references"]

    errors: list[str] = []
    total = canonical["total_repros"]

    _check(
        errors,
        canonical["current_version"]["count"] == total,
        "not all canonical repros use the current format version",
    )
    _check(errors, canonical["unversioned"]["count"] == 0, "unversioned repros found")
    _check(errors, canonical["old_versioned"]["count"] == 0, "old versioned repros found")
    _check(errors, canonical["newer_versioned"]["count"] == 0, "newer versioned repros found")
    _check(
        errors,
        canonical["invalid_version_markers"]["count"] == 0,
        "invalid repro version markers found",
    )
    _check(
        errors,
        canonical["missing_shapes_config"]["count"] == 0,
        "canonical repros missing _shapes_config found",
    )
    _check(
        errors,
        canonical["hash_collisions"]["count"] == 0,
        "duplicate canonical pattern_hash entries found",
    )
    _check(
        errors,
        manifests["references_missing_canonical"]["unique_count"] == 0,
        "model manifests reference missing canonical hashes",
    )
    _check(
        errors,
        manifests["canonical_hashes_unreferenced_by_manifests"]["count"] == 0,
        "canonical hashes unreferenced by model manifests found",
    )
    _check(
        errors,
        manifests["invalid_json"]["count"] == 0,
        "invalid model manifest JSON found",
    )

    if errors:
        print("FAILED: repro corpus invariant violations")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        "PASSED: "
        f"{total} current canonical repros, "
        f"{canonical['unique_pattern_hashes']} unique hashes, "
        f"{manifests['unique_pattern_references']} manifest refs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
