"""Validate one cuTile oracle: imports, numerics check vs eager Repro.

Usage:
    python scripts/validate_cutile_oracle.py \
        repros_cutile/canonical/<name>

Exits 0 on success, non-zero on failure. Prints a compact JSON summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _validate(canonical_dir: Path) -> dict:
    canonical_dir = canonical_dir.resolve()
    result = {"dir": str(canonical_dir)}
    try:
        # Import the oracle module — proves the file is syntactically valid
        # and that all decorators run (oracle_impl point= resolution loads
        # shapes.json).
        from oracle_harness import (
            _load_oracle_module,
            get_inputs,
            get_repro_instance,
            check_oracle_all_shapes,
        )
        mod, fn, d = _load_oracle_module(canonical_dir)
        result["load"] = "ok"

        # Run numerics check at ALL shape points (mirrors what
        # bench_parallel does before benching).
        checks = check_oracle_all_shapes(
            fn, d, canonical_dir.name, skip_stochastic=True)
        result["checks"] = checks
        # A value can be True/False or 'pass'/'fail'/'skip'.
        bad = [k for k, v in checks.items() if v in ("fail", False)]
        result["numerics_ok"] = not bad
        if bad:
            result["failing_shapes"] = bad
    except NotImplementedError as e:
        # Unsupported ports are OK — mark them and continue.
        result["load"] = "not_implemented"
        result["reason"] = str(e)[:400]
        result["numerics_ok"] = None  # excluded from comparison
    except Exception as e:
        result["load"] = "error"
        result["exception_type"] = type(e).__name__
        result["error"] = str(e)[:1000]
        result["traceback"] = traceback.format_exc()[-2000:]
        result["numerics_ok"] = False
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Canonical repro dir with oracle.py to validate")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_dir() or not (d / "oracle.py").exists():
        print(json.dumps({"dir": str(d), "error": "no oracle.py in dir"}))
        sys.exit(2)
    result = _validate(d)
    print(json.dumps(result, default=str))
    ok = result.get("numerics_ok")
    if ok is False:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
