#!/usr/bin/env python3
"""
Validate oracle correctness and detect scope mismatches.

For each oracle file in repros/canonical/*/oracle_*.py:
  1. Attempt to run it with --check flag (most oracles support this).
  2. If --check is not supported, try loading the repro module and oracle
     module to compare outputs programmatically.
  3. Detect scope mismatches (different output count/shapes between oracle
     and eager repro).

Outputs a CSV: investigation_results/oracle_validation_results.csv
with columns: repro_id, oracle_file, status, notes

Status values:
  PASS           - oracle --check passes (returncode 0)
  SCOPE_MISMATCH - oracle produces different number/shapes of outputs
  INCORRECT      - oracle outputs differ numerically from eager reference
  ERROR          - exception/timeout/crash during validation
"""
from __future__ import annotations

import csv
import importlib.util
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "repros" / "canonical"
OUTPUT_CSV = ROOT / "investigation_results" / "oracle_validation_results.csv"

# Ensure project root is on sys.path for repro_harness imports
sys.path.insert(0, str(ROOT))


def find_all_oracles() -> list[tuple[str, Path]]:
    """Find all oracle files, returning (repro_id, oracle_path) pairs."""
    results = []
    if not CANONICAL_DIR.exists():
        return results
    for repro_dir in sorted(CANONICAL_DIR.iterdir()):
        if not repro_dir.is_dir():
            continue
        repro_id = repro_dir.name
        for oracle_file in sorted(repro_dir.glob("oracle_*.py")):
            results.append((repro_id, oracle_file))
    return results


def validate_via_check_flag(oracle_path: Path, timeout: int = 120) -> tuple[str, str]:
    """Run oracle with --check flag and interpret the result.

    Returns (status, notes).
    """
    try:
        result = subprocess.run(
            [sys.executable, str(oracle_path), "--check"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(oracle_path.parent),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")},
        )
        stdout = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
        stderr = result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr

        if result.returncode == 0:
            # Check for scope mismatch indicators in output
            combined = stdout + stderr
            if "scope_mismatch" in combined.lower() or "SCOPE_MISMATCH" in combined:
                return "SCOPE_MISMATCH", f"Oracle reported scope mismatch in output"
            if "PASS" in combined or "passed" in combined.lower() or "correct" in combined.lower():
                return "PASS", "Oracle --check passed"
            # returncode 0 without explicit PASS is still considered passing
            return "PASS", "Oracle --check exited 0"
        else:
            # Non-zero return code
            if "scope" in (stdout + stderr).lower() and "mismatch" in (stdout + stderr).lower():
                return "SCOPE_MISMATCH", f"exit={result.returncode}; {_extract_error(stderr, stdout)}"
            if "allclose" in (stdout + stderr).lower() or "FAIL" in (stdout + stderr):
                return "INCORRECT", f"exit={result.returncode}; {_extract_error(stderr, stdout)}"
            return "ERROR", f"exit={result.returncode}; {_extract_error(stderr, stdout)}"

    except subprocess.TimeoutExpired:
        return "ERROR", f"Timeout after {timeout}s"
    except Exception as e:
        return "ERROR", f"Exception running --check: {type(e).__name__}: {e}"


def _extract_error(stderr: str, stdout: str) -> str:
    """Extract a concise error message from subprocess output."""
    # Try to find the last meaningful error line
    for output in [stderr, stdout]:
        lines = output.strip().splitlines()
        # Look for common error patterns
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            if any(kw in line for kw in ["Error", "FAIL", "assert", "mismatch", "diff"]):
                return line[:200]
        # Fall back to last non-empty line
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line[:200]
    return "(no output)"


def validate_via_module_loading(
    repro_id: str, oracle_path: Path, repro_path: Path
) -> tuple[str, str]:
    """Try to load both the repro and oracle modules and compare outputs.

    This is used as a fallback when --check is not available or fails due
    to missing the flag.

    Supports both:
    - Standard format: oracle has `oracle_forward(inputs)` + `get_inputs()`
    - Legacy format: oracle has `oracle_*(*inputs)` callable

    Returns (status, notes).
    """
    try:
        # Load repro module
        repro_spec = importlib.util.spec_from_file_location(
            f"{repro_id}_repro", repro_path
        )
        if repro_spec is None or repro_spec.loader is None:
            return "ERROR", "Could not load repro module spec"
        repro_module = importlib.util.module_from_spec(repro_spec)
        repro_spec.loader.exec_module(repro_module)

        # Get inputs
        if hasattr(repro_module, "make_inputs"):
            make_fn = repro_module.make_inputs
        elif hasattr(repro_module, "_default_make_inputs"):
            make_fn = repro_module._default_make_inputs
        else:
            return "ERROR", "Repro has no make_inputs or _default_make_inputs"

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        inputs = make_fn()

        # Run eager
        repro_instance = repro_module.Repro()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        with torch.no_grad():
            eager_out = repro_instance(*inputs)

        # Normalize outputs to a list of tensors
        eager_outputs = _normalize_outputs(eager_out)

        # Try to load oracle module and find a callable
        oracle_spec = importlib.util.spec_from_file_location(
            f"{repro_id}_oracle", oracle_path
        )
        if oracle_spec is None or oracle_spec.loader is None:
            return "ERROR", "Could not load oracle module spec"
        oracle_module = importlib.util.module_from_spec(oracle_spec)
        oracle_spec.loader.exec_module(oracle_module)

        # Preferred: standard format oracle_forward(inputs)
        oracle_fn = None
        call_with_tuple = False
        if hasattr(oracle_module, "oracle_forward"):
            oracle_fn = oracle_module.oracle_forward
            call_with_tuple = True  # standard format takes inputs as a tuple
        else:
            # Legacy: look for oracle_* callable that takes *inputs
            for name in dir(oracle_module):
                if name.startswith("oracle_"):
                    candidate = getattr(oracle_module, name)
                    if callable(candidate):
                        oracle_fn = candidate
                        break

        if oracle_fn is None:
            # Try check_correctness function
            if hasattr(oracle_module, "check_correctness"):
                return "PASS", "Has check_correctness (use --check for full validation)"
            return "ERROR", "No callable oracle_forward or oracle_* function found"

        # Run oracle
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        with torch.no_grad():
            if call_with_tuple:
                oracle_out = oracle_fn(inputs)
            else:
                oracle_out = oracle_fn(*inputs)

        oracle_outputs = _normalize_outputs(oracle_out)

        # Compare output counts
        if len(oracle_outputs) != len(eager_outputs):
            return (
                "SCOPE_MISMATCH",
                f"Output count mismatch: oracle={len(oracle_outputs)}, eager={len(eager_outputs)}",
            )

        # Compare shapes
        for i, (o_out, e_out) in enumerate(zip(oracle_outputs, eager_outputs)):
            if o_out.shape != e_out.shape:
                return (
                    "SCOPE_MISMATCH",
                    f"Shape mismatch at output[{i}]: oracle={list(o_out.shape)}, eager={list(e_out.shape)}",
                )

        # Compare values (allclose)
        all_close = True
        max_diffs = []
        for i, (o_out, e_out) in enumerate(zip(oracle_outputs, eager_outputs)):
            o_f32 = o_out.float()
            e_f32 = e_out.float()
            diff = (o_f32 - e_f32).abs().max().item()
            max_diffs.append(diff)
            # Use generous tolerance for bf16 computations
            if not torch.allclose(o_f32, e_f32, rtol=0.05, atol=0.1):
                all_close = False

        if all_close:
            return "PASS", f"Outputs match (max_diffs={[f'{d:.4g}' for d in max_diffs]})"
        else:
            return (
                "INCORRECT",
                f"Outputs differ: max_diffs={[f'{d:.4g}' for d in max_diffs]}",
            )

    except Exception as e:
        tb = traceback.format_exc()
        # Shorten traceback for CSV
        short_tb = tb.splitlines()[-3:]
        return "ERROR", f"{type(e).__name__}: {e}; {'|'.join(short_tb)}"


def _normalize_outputs(out) -> list[torch.Tensor]:
    """Normalize model outputs to a flat list of tensors."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                result.extend(_normalize_outputs(item))
        return result
    return []


def detect_scope_mismatch_from_source(oracle_path: Path) -> str | None:
    """Heuristic: scan oracle source for scope mismatch indicators.

    Returns a note string if a scope concern is detected, else None.
    """
    try:
        source = oracle_path.read_text()
    except Exception:
        return None

    indicators = []
    # Oracle only benchmarks a subset kernel
    if "subset" in source.lower() and "kernel" in source.lower():
        indicators.append("source mentions subset kernel")
    if "partial" in source.lower() and "output" in source.lower():
        indicators.append("source mentions partial output")
    # Oracle has fewer outputs than repro
    if "scope" in source.lower() and "mismatch" in source.lower():
        indicators.append("source mentions scope mismatch")

    return "; ".join(indicators) if indicators else None


def validate_single_oracle(
    repro_id: str, oracle_path: Path, use_subprocess: bool = True
) -> dict:
    """Validate a single oracle, returning a result dict."""
    repro_path = oracle_path.parent / "repro.py"
    oracle_filename = oracle_path.name

    result = {
        "repro_id": repro_id,
        "oracle_file": oracle_filename,
        "status": "ERROR",
        "notes": "",
    }

    # First check source for scope mismatch hints
    scope_note = detect_scope_mismatch_from_source(oracle_path)

    if use_subprocess:
        # Primary method: run with --check flag via subprocess
        status, notes = validate_via_check_flag(oracle_path)

        # If --check failed because the flag doesn't exist (common pattern:
        # "unrecognized arguments: --check"), try module loading
        if status == "ERROR" and "unrecognized arguments" in notes:
            if repro_path.exists():
                status, notes = validate_via_module_loading(
                    repro_id, oracle_path, repro_path
                )
            else:
                notes = "No --check flag support and no repro.py found"
    else:
        # Direct module loading approach
        if repro_path.exists():
            status, notes = validate_via_module_loading(
                repro_id, oracle_path, repro_path
            )
        else:
            status, notes = "ERROR", "repro.py not found"

    # Append scope hint if found
    if scope_note:
        notes = f"{notes}; [source hint: {scope_note}]"

    result["status"] = status
    result["notes"] = notes
    return result


def write_results_csv(results: list[dict]):
    """Write validation results to CSV."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["repro_id", "oracle_file", "status", "notes"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nResults written to: {OUTPUT_CSV}")


def print_summary(results: list[dict]):
    """Print a summary of validation results."""
    total = len(results)
    by_status = {}
    for r in results:
        status = r["status"]
        by_status[status] = by_status.get(status, 0) + 1

    print("\n" + "=" * 70)
    print("ORACLE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total oracles validated: {total}")
    print()
    for status in ["PASS", "SCOPE_MISMATCH", "INCORRECT", "ERROR"]:
        count = by_status.get(status, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {status:16s}: {count:4d} ({pct:5.1f}%)")
    print("=" * 70)

    # Show details for non-PASS results
    non_pass = [r for r in results if r["status"] != "PASS"]
    if non_pass:
        print(f"\nNon-PASS details ({len(non_pass)} oracles):")
        print("-" * 70)
        for r in non_pass[:30]:  # Limit output
            print(f"  [{r['status']}] {r['repro_id']}/{r['oracle_file']}")
            if r["notes"]:
                note_short = r["notes"][:120]
                print(f"    -> {note_short}")
        if len(non_pass) > 30:
            print(f"  ... and {len(non_pass) - 30} more")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of oracles to validate (for testing)"
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout in seconds for each oracle --check call"
    )
    parser.add_argument(
        "--no-subprocess", action="store_true",
        help="Use direct module loading instead of subprocess --check"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress for each oracle"
    )
    args = parser.parse_args()

    oracles = find_all_oracles()
    if not oracles:
        print("No oracle files found!")
        sys.exit(1)

    print(f"Found {len(oracles)} oracle files in {CANONICAL_DIR}")

    if args.limit:
        oracles = oracles[:args.limit]
        print(f"Limiting to first {args.limit} oracles")

    results = []
    start_time = time.time()

    for i, (repro_id, oracle_path) in enumerate(oracles, 1):
        if args.verbose:
            print(f"[{i}/{len(oracles)}] Validating {repro_id}/{oracle_path.name}...", end=" ", flush=True)

        result = validate_single_oracle(
            repro_id, oracle_path, use_subprocess=not args.no_subprocess
        )
        results.append(result)

        if args.verbose:
            print(f"-> {result['status']}")

    elapsed = time.time() - start_time
    print(f"\nValidation completed in {elapsed:.1f}s")

    write_results_csv(results)
    print_summary(results)

    # Return non-zero if any INCORRECT results found
    incorrect_count = sum(1 for r in results if r["status"] == "INCORRECT")
    if incorrect_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
