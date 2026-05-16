#!/usr/bin/env python3
"""Benchmark selected entries from `kernel_issue_queue.json`.

This is the batch runner for the regenerate -> benchmark -> regenerate loop.
It delegates individual measurements to `scripts/bench.py`, which owns the
per-GPU lock.

Examples:
  python scripts/bench_queue.py --bucket missing_perf --limit 5 --update-meta
  python scripts/bench_queue.py --min-score 50 --limit 10 --device-kind H100 --regenerate-after
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "kernel_issue_queue.json"


def load_queue(args: argparse.Namespace) -> list[dict]:
    if args.queue_hardware:
        from build_issue_queue import _load_status, build_queue

        return build_queue(
            args.models_dir,
            args.queue_hardware,
            status_items=_load_status(args.status_file),
        )

    path = args.queue
    if not path.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_issue_queue.py")],
            cwd=ROOT,
            check=True,
        )
    return json.loads(path.read_text())


def select_items(args: argparse.Namespace) -> list[dict]:
    queue = load_queue(args)
    items = []
    for item in queue:
        if args.status and item.get("status") not in args.status:
            continue
        if args.bucket and item.get("bucket") not in args.bucket:
            continue
        if args.min_score is not None and item.get("priority_score", 0) < args.min_score:
            continue
        if args.require_perf_missing and not item.get("missing_perf"):
            continue
        if not Path(item["repro"]).exists():
            continue
        items.append(item)
        if args.limit and len(items) >= args.limit:
            break
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--status-file", type=Path, default=ROOT / "kernel_issue_status.json")
    parser.add_argument(
        "--queue-hardware",
        default=None,
        help="Build an in-memory selection queue for this hardware, e.g. H100 or B200.",
    )
    parser.add_argument("--bucket", action="append", help="Bucket to include; repeatable")
    parser.add_argument(
        "--status",
        action="append",
        default=["to_investigate"],
        help="Status to include; repeatable. Defaults to to_investigate.",
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--device-kind", default=None)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--update-meta", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--require-perf-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--regenerate-after", action="store_true")
    parser.add_argument(
        "--regenerate-canonical",
        action="store_true",
        help="When used with --queue-hardware, overwrite the canonical queue files.",
    )
    parser.add_argument("--regenerate-json-out", type=Path, default=None)
    parser.add_argument("--regenerate-md-out", type=Path, default=None)
    args = parser.parse_args()

    if args.queue_hardware:
        args.device_kind = args.device_kind or args.queue_hardware
        args.hardware = args.hardware or args.queue_hardware

    items = select_items(args)
    if not items:
        print("No queue entries matched.")
        return 0

    failures = 0
    for idx, item in enumerate(items, 1):
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "bench.py"),
        ]
        if args.device_kind:
            cmd.extend(["--device-kind", args.device_kind])
        else:
            cmd.extend(["--gpu", str(args.gpu)])
        if args.hardware:
            cmd.extend(["--hardware", args.hardware])
        if args.update_meta:
            cmd.append("--update-meta")
        if args.no_cuda_graph:
            cmd.append("--no-cuda-graph")
        cmd.append(item["repro"])

        print(
            f"[{idx}/{len(items)}] {item['bucket']} score={item['priority_score']:.2f} "
            f"status={item['status']} {item['repro']}"
        )
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue

        proc = subprocess.run(cmd, cwd=ROOT)
        if proc.returncode != 0:
            failures += 1
            print(f"  FAILED with exit code {proc.returncode}")

    if args.regenerate_after and not args.dry_run:
        cmd = [sys.executable, str(ROOT / "scripts" / "build_issue_queue.py")]
        if args.queue_hardware:
            cmd.extend(["--hardware", args.queue_hardware])
            if not args.regenerate_canonical:
                json_out = args.regenerate_json_out or (
                    ROOT / f"kernel_issue_queue.{args.queue_hardware}.json"
                )
                md_out = args.regenerate_md_out or (
                    ROOT / f"KERNEL_INVESTIGATION_QUEUE.{args.queue_hardware}.md"
                )
                cmd.extend(["--json-out", str(json_out), "--md-out", str(md_out)])
        elif args.regenerate_json_out or args.regenerate_md_out:
            if args.regenerate_json_out:
                cmd.extend(["--json-out", str(args.regenerate_json_out)])
            if args.regenerate_md_out:
                cmd.extend(["--md-out", str(args.regenerate_md_out)])
        subprocess.run(cmd, cwd=ROOT, check=False)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
