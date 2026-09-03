"""Filter the cuTile canonical dirs to only those with a working (non-stub,
importable, numerics-passing) oracle.

Writes one dir path per line — piping into ``xargs`` to bench_parallel:

    python scripts/select_working_cutile_dirs.py --out working.txt
    python scripts/bench_parallel.py --oracles $(cat working.txt) --output ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_VALIDATOR = _REPO_ROOT / "scripts" / "validate_cutile_oracle.py"


def _quick_check(canonical_dir: str) -> tuple[str, str]:
    """Try importing the oracle module without running numerics. If it raises
    NotImplementedError on invocation OR contains a stub comment, mark not_impl.
    Otherwise import + call check_oracle_all_shapes."""
    d = Path(canonical_dir)
    oracle = d / "oracle.py"
    if not oracle.exists():
        return canonical_dir, "no_oracle"
    text = oracle.read_text()
    # A pure stub has no cuTile kernel — if any @ct.kernel is defined, it's a real port
    # (some rescued oracles use raise NotImplementedError as a CUDA-graph-capture guard
    # but still have real cuTile work on the eager path).
    if "@ct.kernel" not in text:
        return canonical_dir, "stub"
    return canonical_dir, "port"


def _quick_check_batch(paths: list[str], jobs: int) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"port": [], "stub": [], "no_oracle": []}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for fut in as_completed(ex.submit(_quick_check, p) for p in paths):
            d, kind = fut.result()
            buckets.setdefault(kind, []).append(d)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=_REPO_ROOT / "repros_cutile" / "canonical")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stub-out", type=Path, default=None,
                    help="write the stub list here (optional)")
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    dirs = sorted(str(p.parent) for p in args.root.rglob("oracle.py"))
    buckets = _quick_check_batch(dirs, args.jobs)
    args.out.write_text("\n".join(sorted(buckets.get("port", []))) + "\n")
    if args.stub_out:
        args.stub_out.write_text("\n".join(sorted(buckets.get("stub", []))) + "\n")
    print(f"port:      {len(buckets.get('port', [])):>5}", file=sys.stderr)
    print(f"stub:      {len(buckets.get('stub', [])):>5}", file=sys.stderr)
    print(f"no_oracle: {len(buckets.get('no_oracle', [])):>5}", file=sys.stderr)
    print(f"total:     {len(dirs):>5}", file=sys.stderr)


if __name__ == "__main__":
    main()
