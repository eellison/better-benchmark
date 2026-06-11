#!/usr/bin/env python3
"""Validate corpus invariants that should NEVER regress.

This script enforces hard invariants (exit 1 on failure) and soft invariants
(warn but still exit 0). Designed to run in CI or as a pre-commit hook on
changes touching repros/.

Usage:
    python scripts/validate_corpus_invariants.py

    # Validate a freshly captured corpus at another root
    # instead of the checked-in repros/. The baseline count check is skipped
    # for non-default roots (a fresh corpus has no baseline yet).
    python scripts/validate_corpus_invariants.py --corpus-root /tmp/recapture_corpus/repros

    # Full-graph round-trip invariants (A: input round-trip, B: partition
    # determinism, C: partition round-trip vs manifest). Samples 2 models per
    # suite by default; --all validates every model dir with full_graph_*.py.
    python scripts/validate_corpus_invariants.py --full-graph-roundtrip
    python scripts/validate_corpus_invariants.py --full-graph-roundtrip --all

Exit codes:
    0 - all hard invariants pass (soft warnings may be present)
    1 - one or more hard invariants violated
       (in --full-graph-roundtrip mode: any A/B violation, trace failure, or
        non-spelling C mismatch)
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = ROOT / "repros"
# Reassigned by --corpus-root (set_corpus_root) before any invariant runs.
CANONICAL_DIR = DEFAULT_CORPUS_ROOT / "canonical"
MODELS_DIR = DEFAULT_CORPUS_ROOT / "models"
BASELINE_PATH = ROOT / ".corpus_baseline.json"

sys.path.insert(0, str(ROOT))


def set_corpus_root(corpus_root: Path) -> bool:
    """Point all invariants at another corpus tree (a recapture output root).

    Returns True if this is a NON-default root — callers then skip the
    baseline count check (a fresh corpus has no baseline; every other
    invariant is root-relative and applies unchanged).
    """
    global CANONICAL_DIR, MODELS_DIR
    corpus_root = Path(corpus_root).resolve()
    CANONICAL_DIR = corpus_root / "canonical"
    MODELS_DIR = corpus_root / "models"
    return corpus_root != DEFAULT_CORPUS_ROOT.resolve()


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


_SHAPES_CONFIG_LINE = re.compile(
    r'^_shapes_config\s*=\s*["\'](.+?)["\']\s*$', re.MULTILINE)


def check_shapes_config() -> list[str]:
    """Hard invariant 3: every repro must have a loadable input config.

    v3 repros (no inline _shapes_config; default = first shapes.json
    point): the dir MUST have shapes.json with at least one point carrying
    structured "inputs", and every point's "signature" string must equal
    render_signature(inputs) (the string is a rendering of the data —
    drift means a writer bypassed the codec).

    v2 repros (inline _shapes_config string): the legacy check — the
    string must be a real top-level assignment, and where shapes.json
    exists it must match one point's signature.
    """
    errors = []
    missing = []
    drifted = []
    for repro_py in sorted(CANONICAL_DIR.glob("*/repro.py")):
        source = repro_py.read_text()
        shapes_path = repro_py.parent / "shapes.json"
        is_v3 = "_repro_version = 3" in source

        if is_v3:
            if not shapes_path.exists():
                missing.append(f"{repro_py.parent.name} (v3, no shapes.json)")
                continue
            try:
                points = json.loads(shapes_path.read_text()).get("points", [])
            except Exception:
                continue  # malformed shapes.json caught elsewhere
            if not points or not any(p.get("inputs") for p in points):
                missing.append(
                    f"{repro_py.parent.name} (v3, no structured inputs)")
                continue
            try:
                from input_codec import render_signature
                for p in points:
                    if p.get("inputs") and p.get("signature"):
                        if render_signature(p["inputs"]) != p["signature"]:
                            drifted.append(
                                f"{repro_py.parent.name}/{p.get('shape_hash')}")
            except ImportError:
                pass
            continue

        # v2 legacy path
        if "_shapes_config" not in source:
            missing.append(repro_py.parent.name)
            continue
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
            continue
        if shapes_path.exists():
            m = _SHAPES_CONFIG_LINE.search(source)
            if m:
                try:
                    sigs = {p.get("signature") for p in
                            json.loads(shapes_path.read_text()).get("points", [])}
                except Exception:
                    sigs = None
                if sigs is not None and m.group(1) not in sigs:
                    drifted.append(repro_py.parent.name)

    if drifted:
        errors.append(
            f"{len(drifted)} signature/inputs drift(s) (string is not the "
            f"rendering of the data):\n"
            + "\n".join(f"    {d}" for d in drifted[:10])
        )

    if missing:
        errors.append(
            f"{len(missing)} repro(s) without a loadable input config:\n"
            + "\n".join(f"    {m}" for m in missing[:10])
        )
    else:
        count = len(list(CANONICAL_DIR.glob("*/repro.py")))
        print(f"  [PASS] all {count} repros have loadable input configs")
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
            [sys.executable, str(validate_script), "--quick", "--timeout", "30",
             # Pass the ACTIVE canonical dir — without this the child script
             # hardcodes the checked-in repros/ and silently validates the
             # wrong tree under --corpus-root (opus verifier gap #6).
             "--canonical-dir", str(CANONICAL_DIR)],
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
# Full-graph round-trip invariants (--full-graph-roundtrip)
# ---------------------------------------------------------------------------

def _model_dirs_with_full_graphs() -> dict[str, list[Path]]:
    """suite -> sorted list of model dirs containing full_graph_*.py."""
    by_suite: dict[str, list[Path]] = {}
    seen: set[Path] = set()
    for g in MODELS_DIR.rglob("full_graph_*.py"):
        d = g.parent
        if d in seen:
            continue
        seen.add(d)
        suite = d.relative_to(MODELS_DIR).parts[0]
        by_suite.setdefault(suite, []).append(d)
    for dirs in by_suite.values():
        dirs.sort()
    return by_suite


def _manifest_only_dirs() -> int:
    """Count model dirs that have manifest.json but NO full_graph_*.py.

    These are a separate KNOWN class: the full post-grad graph was never
    saved (or was captured before graph_dir support), so round-trip
    validation cannot apply. They need recapture, not validation.
    """
    count = 0
    for m in MODELS_DIR.rglob("manifest.json"):
        if not list(m.parent.glob("full_graph_*.py")):
            count += 1
    return count


def validate_model_roundtrip(
    model_dir: Path,
    *,
    max_graphs: int | None = None,
) -> dict:
    """Run invariants A/B/C for one model dir. Returns a result dict."""
    from roundtrip_validation import (
        canonical_repro_ops,
        check_input_roundtrip,
        check_partition_determinism,
        compare_pattern_sets,
        derive_partitions,
        trace_full_graph_fake,
    )
    from full_graph_harness import load_full_graph_definition

    result = {
        "model_dir": str(model_dir.relative_to(ROOT)),
        "suite": model_dir.relative_to(MODELS_DIR).parts[0],
        "graphs": [],
        "a_failures": [],      # invariant A: input round-trip
        "b_failures": [],      # invariant B: partition determinism
        "c": None,             # invariant C: vs manifest.json
        "c_known_class": None, # "no-manifest" when C cannot apply
        "trace_failures": [],  # graphs that could not be re-traced at all
    }

    graph_paths = sorted(model_dir.glob("full_graph_*.py"))
    if max_graphs is not None:
        graph_paths = graph_paths[:max_graphs]

    derived_parts_all: list[dict] = []
    for gpath in graph_paths:
        gname = gpath.name
        result["graphs"].append(gname)

        # --- Invariant A: input round-trip ---
        try:
            definition = load_full_graph_definition(gpath)
        except Exception as exc:
            result["a_failures"].append(
                f"{gname}: load failed: {type(exc).__name__}: {exc}"
            )
            continue
        a_failures = check_input_roundtrip(definition)
        if a_failures:
            result["a_failures"].extend(f"{gname}: {f}" for f in a_failures)
            continue

        # --- Re-trace (required for B and C) ---
        try:
            gm = trace_full_graph_fake(definition)
        except Exception as exc:
            result["trace_failures"].append(
                f"{gname}: {type(exc).__name__}: {str(exc)[:300]}"
            )
            continue

        # --- Invariant B: partition determinism ---
        ok, diff = check_partition_determinism(gm)
        if not ok:
            result["b_failures"].append(f"{gname}: {diff}")

        derived_parts_all.extend(derive_partitions(gm))

    # --- Invariant C: partition round-trip vs manifest.json ---
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        result["c_known_class"] = "no-manifest"
        return result
    try:
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get("patterns", [])
    except Exception as exc:
        result["c"] = {
            "ok": False,
            "missing": [],
            "extra": [],
            "spelling": [],
            "hard": [f"manifest.json unreadable: {exc}"],
        }
        return result
    if result["a_failures"] or result["trace_failures"]:
        # C would report bogus "missing" hashes if graphs failed upstream.
        result["c_known_class"] = "skipped-upstream-failure"
        return result

    result["c"] = compare_pattern_sets(
        derived_parts_all,
        expected,
        expected_ops_lookup=lambda h: canonical_repro_ops(h, CANONICAL_DIR),
    )
    return result


def run_full_graph_roundtrip(
    *,
    sample_per_suite: int = 2,
    validate_all: bool = False,
    max_graphs: int | None = 10,
    seed: int = 0,
    json_out: Path | None = None,
) -> int:
    print("=" * 60)
    print("Full-Graph Round-Trip Invariant Validation")
    print("=" * 60)

    by_suite = _model_dirs_with_full_graphs()
    n_total = sum(len(v) for v in by_suite.values())
    n_manifest_only = _manifest_only_dirs()
    print(f"\n{n_total} model dirs have full_graph_*.py artifacts "
          f"({', '.join(f'{s}={len(d)}' for s, d in sorted(by_suite.items()))})")
    print(f"[KNOWN CLASS] {n_manifest_only} model dirs are manifest-only "
          f"(no full_graph_*.py saved; round-trip N/A, need recapture)\n")

    rng = random.Random(seed)
    selected: list[Path] = []
    for suite in sorted(by_suite):
        dirs = by_suite[suite]
        if validate_all or len(dirs) <= sample_per_suite:
            selected.extend(dirs)
        else:
            selected.extend(sorted(rng.sample(dirs, sample_per_suite)))

    if validate_all:
        max_graphs = None

    results = []
    a_violations = b_violations = trace_violations = c_hard = 0
    for model_dir in selected:
        res = validate_model_roundtrip(model_dir, max_graphs=max_graphs)
        results.append(res)

        status_bits = []
        if res["a_failures"]:
            a_violations += 1
            status_bits.append("A:FAIL")
        else:
            status_bits.append("A:ok")
        if res["b_failures"]:
            b_violations += 1
            status_bits.append("B:FAIL")
        elif res["trace_failures"]:
            status_bits.append("B:skip")
        else:
            status_bits.append("B:ok")
        if res["trace_failures"]:
            trace_violations += 1
            status_bits.append("trace:FAIL")
        c = res["c"]
        if c is None:
            status_bits.append(f"C:n/a({res['c_known_class']})")
        elif c["hard"]:
            c_hard += 1
            status_bits.append("C:FAIL")
        elif c["spelling"]:
            status_bits.append("C:ok(spelling)")
        else:
            status_bits.append("C:ok")

        print(f"  [{' '.join(status_bits)}] {res['model_dir']} "
              f"({len(res['graphs'])} graph(s))")
        for f in res["a_failures"][:5]:
            print(f"      A: {f}")
        for f in res["b_failures"][:5]:
            print(f"      B: {f}")
        for f in res["trace_failures"][:5]:
            print(f"      trace: {f}")
        if c:
            for f in c["hard"][:5]:
                print(f"      C(hard): {f}")
            for f in c["spelling"][:5]:
                print(f"      C(spelling, known class): {f}")

    print("\n" + "=" * 60)
    print(f"Validated {len(results)} model dir(s) "
          f"({'all' if validate_all else f'sampled {sample_per_suite}/suite, seed={seed}'})")
    print(f"  A (input round-trip) violations:      {a_violations}")
    print(f"  B (partition determinism) violations: {b_violations}")
    print(f"  re-trace failures:                    {trace_violations}")
    print(f"  C hard (non-spelling) mismatches:     {c_hard}")
    n_spelling = sum(
        1 for r in results
        if r["c"] and r["c"]["spelling"] and not r["c"]["hard"]
    )
    print(f"  C spelling-only mismatches (known):   {n_spelling}")
    print(f"  manifest-only dirs (known class):     {n_manifest_only}")

    if json_out is not None:
        payload = {
            "manifest_only_dirs": n_manifest_only,
            "sampled": not validate_all,
            "sample_per_suite": sample_per_suite,
            "seed": seed,
            "results": results,
        }
        json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote JSON results to {json_out}")

    failed = a_violations or b_violations or trace_violations or c_hard
    print("\n" + ("FAILED" if failed else "PASSED"))
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-graph-roundtrip", action="store_true",
        help="Run full-graph round-trip invariants (A/B/C) instead of the "
             "standard corpus invariants.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="(roundtrip mode) validate ALL model dirs with full graphs, "
             "not a per-suite sample, and all graphs per model.",
    )
    parser.add_argument(
        "--sample", type=int, default=2,
        help="(roundtrip mode) models sampled per suite (default 2).",
    )
    parser.add_argument(
        "--max-graphs", type=int, default=10,
        help="(roundtrip mode, sampled) cap on graphs validated per model "
             "(default 10; --all removes the cap).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="(roundtrip mode) sampling seed (default 0, deterministic).",
    )
    parser.add_argument(
        "--json", type=Path, default=None,
        help="(roundtrip mode) write per-model results JSON to this path.",
    )
    parser.add_argument(
        "--corpus-root", type=Path, default=None,
        help="Validate the corpus at this root (containing canonical/ and "
             "models/) instead of the checked-in repros/. Baseline count "
             "check is skipped for non-default roots.",
    )
    args = parser.parse_args()

    skip_baseline = False
    if args.corpus_root is not None:
        skip_baseline = set_corpus_root(args.corpus_root)

    if args.full_graph_roundtrip:
        return run_full_graph_roundtrip(
            sample_per_suite=args.sample,
            validate_all=args.all,
            max_graphs=args.max_graphs,
            seed=args.seed,
            json_out=args.json,
        )
    return run_standard_invariants(skip_baseline=skip_baseline)


def run_standard_invariants(skip_baseline: bool = False) -> int:
    print("=" * 60)
    print("Corpus Invariant Validation")
    print(f"  canonical: {CANONICAL_DIR}")
    print(f"  models:    {MODELS_DIR}")
    print("=" * 60)

    if not CANONICAL_DIR.exists():
        print(f"ERROR: canonical directory not found: {CANONICAL_DIR}")
        return 1

    # --- Hard invariants ---
    print("\n--- Hard Invariants (must pass) ---\n")
    hard_errors: list[str] = []
    if skip_baseline:
        print("  [SKIP] baseline repro-count check (non-default corpus root)")
    else:
        hard_errors.extend(check_repro_count(load_baseline()))
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

    if skip_baseline:
        print("  [SKIP] suite-coverage check (baseline-relative)")
    else:
        w = warn_pattern_count_per_model(load_baseline())
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
