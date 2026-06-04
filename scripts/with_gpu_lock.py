#!/usr/bin/env python3
"""Run an arbitrary command while holding a per-GPU benchmark lock.

Examples:
  python scripts/with_gpu_lock.py --gpu 0 -- python path/to/gpu_debug_script.py
  python scripts/with_gpu_lock.py --device-kind H100 -- python scripts/bench.py path/to/repro.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from gpu_lock import gpu_lock, gpu_lock_for_kind


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=None, help="GPU id to lock and expose")
    parser.add_argument("--device-kind", default=None, help="GPU kind to lock, e.g. H100 or B200")
    parser.add_argument("--lock-dir", type=Path, default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--label", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("missing command after --")

    label = args.label or " ".join(command[:4])

    env = os.environ.copy()
    if args.device_kind:
        with gpu_lock_for_kind(
            args.device_kind,
            lock_dir=args.lock_dir,
            timeout_s=args.timeout_s,
            label=label,
        ) as device:
            env["CUDA_VISIBLE_DEVICES"] = device["index"]
            print(
                f"[gpu-lock] acquired {device['lock_path']} "
                f"({device['kind']} {device['name']})",
                flush=True,
            )
            return subprocess.run(command, env=env).returncode

    gpu = 0 if args.gpu is None else args.gpu
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with gpu_lock(gpu, lock_dir=args.lock_dir, timeout_s=args.timeout_s, label=label) as lock_path:
        print(f"[gpu-lock] acquired {lock_path}", flush=True)
        return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
