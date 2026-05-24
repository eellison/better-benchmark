#!/usr/bin/env python3
"""Validate corpus invariants that should NEVER regress.

This script enforces hard invariants (exit 1 on failure) and soft invariants
(warn but still exit 0). Designed to run in CI or as a pre-commit hook on
changes touching repros/.

Usage:
    python scripts/validate_corpus_invariants.py

Exit codes:
    0 - all hard invariants pass (soft warnings may be present)
    1 - one or more hard invariants violated
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "repros" / "canonical"
MODELS_DIR = ROOT / "repros" / "models"
BASELINE_PATH = ROOT / ".corpus_baseline.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_baseline() -> dict:
    """Load .corpus_baseline.json from repo root."""
    if not BASELINE_PATH.exists():
        print(f"ERROR: baseline file not found: {BASELINE_PATH}")
        sys.exit(1)
    return json.loads(BASELINE_PATH.read_text())


def _read_json_safe(path: Path) -> tuple[dict | None, str | None]:
    """Read JSON file, return (data, error)."""
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return None, "top-level value is not a JSON object"
        return data, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Hard invariants
# ---------------------------------------------------------------------------

def check_repro_count(baseline: dict) -> list[str]:
    """Hard invariant 1: repro count never decreases below baseline minimum."""
    errors = []
    min_count = baseline["min_repro_count"]
    repro_files = list(CANONICAL_DIR.glob("*/repro.py"))
    actual = len(repro_files)
    if actual < min_count:
        errors.append(
            f"Repro count regressed: found {actual} repro.py files, "
            f"baseline minimum is {min_count}. "
            f"This looks like data loss (cf. commit 1b61300e)."
        )
    else:
        print(f"  [PASS] repro count: {actual} >= {min_count} (baseline)")
    return errors


def check_repros_parse() -> list[str]:
    """Hard invariant 2: every repro.py must parse (valid Python syntax)."""
    errors = []
    bad = []
    for repro_py in sorted(CANONICAL_DIR.glob("*/repro.py")):
        try:
            ast.parse(repro_py.read_text())
        except SyntaxError as exc:
            bad.append(f"{repro_py.parent.name}: {exc}")
    if bad:
        errors.append(
            f"{len(bad)} repro.py file(s) have syntax errors:\n"
            + "\n".join(f"    {b}" for b in bad[:10])
        )
    else:
        count = len(list(CANONICAL_DIR.glob("*/repro.py")))
        print(f"  [PASS] all {count} repro.py files parse successfully")
    return errors


def check_shapes_config() -> list[str]:
    """Hard invariant 3: every repro.py must have _shapes_config (v2 format)."""
    errors = []
    missing = []
    for repro_py in sorted(CANONICAL_DIR.glob("*/repro.py")):
        source = repro_py.read_text()
        if "_shapes_config" not in source:
            missing.append(repro_py.parent.name)
            continue
        # Verify it's a real top-level assignment
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue  # covered by invariant 2
        found = False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_shapes_config":
                        found = True
                        break
        if not found:
            missing.append(repro_py.parent.name)

    if missing:
        errors.append(
            f"{len(missing)} repro.py file(s) missing _shapes_config:\n"
            + "\n".join(f"    {m}" for m in missing[:10])
        )
    else:
        count = len(list(CANONICAL_DIR.glob("*/repro.py")))
        print(f"  [PASS] all {count} repro.py files have _shapes_config")
    return errors


def check_manifest_consistency() -> list[str]:
    """Hard invariant 4: every pattern in a manifest must resolve to canonical/."""
    errors = []
    canonical_hashes = set()
    for d in CANONICAL_DIR.iterdir():
        if d.is_dir():
            meta = d / "meta.json"
            if meta.exists():
                data, _ = _read_json_safe(meta)
                if data and "pattern_hash" in data:
                    canonical_hashes.add(data["pattern_hash"])
            # Also extract hash from directory name as fallback
            name = d.name
            if len(name) >= 12:
                suffix = name.split("_")[-1]
                if len(suffix) == 12 and all(c in "0123456789abcdef" for c in suffix):
                    canonical_hashes.add(suffix)

    missing_refs = []
    for manifest_path in sorted(MODELS_DIR.rglob("*.json")):
        data, err = _read_json_safe(manifest_path)
        if data is None:
            continue
        patterns = data.get("patterns", [])
        if not isinstance(patterns, list):
            continue
        for pat in patterns:
            if isinstance(pat, str) and pat not in canonical_hashes:
                missing_refs.append(
                    f"{manifest_path.relative_to(ROOT)}: {pat}"
                )

    if missing_refs:
        errors.append(
            f"{len(missing_refs)} manifest pattern reference(s) "
            f"do not resolve to canonical/:\n"
            + "\n".join(f"    {r}" for r in missing_refs[:10])
        )
    else:
        print(f"  [PASS] all manifest patterns resolve to canonical/ "
              f"({len(canonical_hashes)} hashes)")
    return errors


def check_no_empty_canonical_dirs() -> list[str]:
    """Hard invariant 5: every canonical dir must have repro.py."""
    errors = []
    empty = []
    for d in sorted(CANONICAL_DIR.iterdir()):
        if d.is_dir() and not (d / "repro.py").exists():
            empty.append(d.name)

    if empty:
        errors.append(
            f"{len(empty)} canonical dir(s) missing repro.py:\n"
            + "\n".join(f"    {e}" for e in empty[:10])
        )
    else:
        count = sum(1 for d in CANONICAL_DIR.iterdir() if d.is_dir())
        print(f"  [PASS] all {count} canonical directories contain repro.py")
    return errors


def check_shapes_eager_validation() -> list[str]:
    """Hard invariant 6: a random sample of shapes must pass eager validation.

    Runs validate_shapes_eager.py --quick (50 random shapes) and fails if any
    shape entry does not work in eager mode.
    """
    import subprocess
    errors = []

    validate_script = ROOT / "scripts" / "validate_shapes_eager.py"
    if not validate_script.exists():
        # Script not present, skip gracefully
        print("  [SKIP] validate_shapes_eager.py not found")
        return errors

    try:
        result = subprocess.run(
            [sys.executable, str(validate_script), "--quick", "--timeout", "30"],
            capture_output=True, text=True, timeout=300,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            # Extract failure count from output
            output_lines = result.stdout.strip().split("\n")
            fail_summary = [l for l in output_lines if "FAIL" in l.upper()]
            sample = "\n".join(fail_summary[:5]) if fail_summary else result.stdout[-500:]
            errors.append(
                f"Eager validation of sampled shapes failed (exit {result.returncode}):\n"
                f"    {sample}"
            )
        else:
            print("  [PASS] sampled shapes pass eager validation")
    except subprocess.TimeoutExpired:
        errors.append("Eager validation timed out after 300s")
    except Exception as exc:
        errors.append(f"Eager validation failed to run: {exc}")

    return errors


# ---------------------------------------------------------------------------
# Soft invariants (warnings only)
# ---------------------------------------------------------------------------

def warn_multi_model_shapes() -> list[str]:
    """Soft invariant 6: multi-model patterns should have shapes data."""
    warnings = []
    for d in sorted(CANONICAL_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta = d / "meta.json"
        if not meta.exists():
            continue
        data, _ = _read_json_safe(meta)
        if data is None:
            continue
        n_models = data.get("n_models", 0)
        if n_models > 3:
            has_shapes = (d / "shapes.json").exists() or (d / "shapes.txt").exists()
            if not has_shapes:
                warnings.append(
                    f"{d.name}: n_models={n_models} but no shapes.json/shapes.txt"
                )
    return warnings


def warn_shapes_config_parseable() -> list[str]:
    """Soft invariant 7: _shapes_config values should be parseable."""
    warnings = []
    # Import parse_shapes_config if available (needs torch, so may fail)
    try:
        sys.path.insert(0, str(ROOT))
        from repro_harness import parse_shapes_config
    except Exception:
        # If we can't import (no torch), skip this check gracefully
        return []

    for repro_py in sorted(CANONICAL_DIR.glob("*/repro.py")):
        source = repro_py.read_text()
        # Extract _shapes_config value
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_shapes_config":
                        try:
                            config_val = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            continue
                        if isinstance(config_val, str):
                            try:
                                parse_shapes_config(config_val)
                            except Exception as exc:
                                warnings.append(
                                    f"{repro_py.parent.name}: {exc}"
                                )
    return warnings


def warn_pattern_count_per_model(baseline: dict) -> list[str]:
    """Soft invariant 8: pattern count per suite shouldn't drop."""
    warnings = []
    expected_suites = set(baseline.get("expected_suites", []))
    actual_suites = set()

    if MODELS_DIR.exists():
        for d in MODELS_DIR.iterdir():
            if d.is_dir():
                actual_suites.add(d.name)

    missing_suites = expected_suites - actual_suites
    if missing_suites:
        warnings.append(
            f"Expected model suites missing: {sorted(missing_suites)}"
        )

    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("Corpus Invariant Validation")
    print("=" * 60)

    if not CANONICAL_DIR.exists():
        print(f"ERROR: canonical directory not found: {CANONICAL_DIR}")
        return 1

    baseline = load_baseline()

    # --- Hard invariants ---
    print("\n--- Hard Invariants (must pass) ---\n")
    hard_errors: list[str] = []
    hard_errors.extend(check_repro_count(baseline))
    hard_errors.extend(check_repros_parse())
    hard_errors.extend(check_shapes_config())
    hard_errors.extend(check_manifest_consistency())
    hard_errors.extend(check_no_empty_canonical_dirs())
    hard_errors.extend(check_shapes_eager_validation())

    # --- Soft invariants ---
    print("\n--- Soft Invariants (warnings) ---\n")
    soft_warnings: list[str] = []

    w = warn_multi_model_shapes()
    if w:
        soft_warnings.extend(w)
        print(f"  [WARN] {len(w)} multi-model pattern(s) missing shapes data")
        for item in w[:5]:
            print(f"         {item}")
    else:
        print("  [OK] multi-model patterns have shapes data")

    w = warn_shapes_config_parseable()
    if w:
        soft_warnings.extend(w)
        print(f"  [WARN] {len(w)} _shapes_config value(s) failed to parse")
        for item in w[:5]:
            print(f"         {item}")
    else:
        print("  [OK] _shapes_config values parseable (or check skipped)")

    w = warn_pattern_count_per_model(baseline)
    if w:
        soft_warnings.extend(w)
        print(f"  [WARN] {len(w)} suite coverage issue(s)")
        for item in w[:5]:
            print(f"         {item}")
    else:
        print("  [OK] all expected suites present")

    # --- Summary ---
    print("\n" + "=" * 60)
    if hard_errors:
        print(f"FAILED: {len(hard_errors)} hard invariant(s) violated:\n")
        for err in hard_errors:
            print(f"  ERROR: {err}\n")
        if soft_warnings:
            print(f"  (also {len(soft_warnings)} soft warning(s))")
        return 1
    else:
        msg = "PASSED: all hard invariants satisfied"
        if soft_warnings:
            msg += f" ({len(soft_warnings)} soft warning(s))"
        print(msg)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
