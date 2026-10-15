"""Validate a batch of cuTile oracle dirs. Runs each in a subprocess so that
CUDA errors don't kill the batch.

Usage:
    python scripts/validate_cutile_batch.py <root> [--jobs N] [--out RESULT.jsonl]

Where <root> is either a single canonical dir or a parent dir like
`repros_cutile/canonical`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_VALIDATOR = _REPO_ROOT / "scripts" / "validate_cutile_oracle.py"


def _validate_one(canonical_dir: str) -> dict:
    d = Path(canonical_dir)
    if not (d / "oracle.py").exists():
        return {"dir": str(d), "load": "missing", "numerics_ok": None}
    try:
        proc = subprocess.run(
            [sys.executable, str(_VALIDATOR), str(d)],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        return {"dir": str(d), "load": "timeout", "numerics_ok": False}
    stdout = (proc.stdout or "").strip().splitlines()
    if not stdout:
        return {
            "dir": str(d),
            "load": "no_output",
            "numerics_ok": False,
            "stderr": (proc.stderr or "")[-500:],
        }
    try:
        return json.loads(stdout[-1])
    except json.JSONDecodeError:
        return {
            "dir": str(d),
            "load": "parse_error",
            "numerics_ok": False,
            "raw": stdout[-1][:500],
        }


def _find_dirs(root: Path) -> list[Path]:
    if (root / "oracle.py").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("oracle.py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None,
                    help="jsonl output; default: print each line to stdout")
    args = ap.parse_args()

    dirs = _find_dirs(args.root)
    if not dirs:
        print("no oracle.py files under", args.root, file=sys.stderr)
        sys.exit(2)

    out_fh = args.out.open("w") if args.out else None

    ok = 0
    bad = 0
    missing = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(_validate_one, str(d)): d for d in dirs}
        for fut in as_completed(futures):
            r = fut.result()
            line = json.dumps(r, default=str)
            if out_fh:
                out_fh.write(line + "\n")
                out_fh.flush()
            else:
                print(line)
            if r.get("numerics_ok") is True:
                ok += 1
            elif r.get("numerics_ok") is False:
                bad += 1
            else:
                missing += 1
    if out_fh:
        out_fh.close()
    print(f"# ok={ok} bad={bad} missing_or_skip={missing} total={len(dirs)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
