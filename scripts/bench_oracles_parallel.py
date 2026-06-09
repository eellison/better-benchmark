"""
Parallel oracle benchmark runner.

Finds all oracle_*.py files in the corpus, runs each with --bench in parallel
across available GPUs, and collects JSON results.

Architecture:
  - 2 GPUs, N workers per GPU (default: 4 per GPU = 8 total)
  - Each worker subprocess: CUDA_VISIBLE_DEVICES=<gpu_idx>, INDUCTOR_GPU_BENCH_LOCK=1
  - The GPU lock serializes timing phases; compilation can overlap
  - Subprocess per oracle (avoids CUDA state pollution between oracles)

Usage:
    python scripts/bench_oracles_parallel.py
    python scripts/bench_oracles_parallel.py --workers-per-gpu 4 --output results/all_oracle_timings_b200_v2.json
    python scripts/bench_oracles_parallel.py --resume  # skip already-measured oracles
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from gpu_lock import discover_gpus, matching_gpus


def find_oracles(root: Path) -> list[Path]:
    """Find all oracle_*.py files in the corpus."""
    return sorted(root.rglob("oracle_*.py"))


def _run_single_oracle(oracle_path: str, gpu_idx: str, timeout: int = 120) -> dict:
    """Run a single oracle with --bench and capture JSON output."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    env["INDUCTOR_GPU_BENCH_LOCK"] = "1"

    try:
        result = subprocess.run(
            [sys.executable, oracle_path, "--bench"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(_REPO_ROOT),
        )

        # Parse JSON lines from stdout (bench_oracle prints JSON via json.dumps)
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "repro_id" in data or "oracle_us" in data:
                    return {"status": "ok", "data": data, "path": oracle_path}
            except json.JSONDecodeError:
                continue

        # No JSON found — check if there's an error
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        return {
            "status": "no_output",
            "path": oracle_path,
            "returncode": result.returncode,
            "stderr": stderr_tail,
        }

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "path": oracle_path}
    except Exception as e:
        return {"status": "error", "path": oracle_path, "error": str(e)[:200]}


def _worker(gpu_idx: str, task_queue: mp.Queue, result_queue: mp.Queue,
            timeout: int):
    """Worker process: picks oracles from queue, runs them on assigned GPU."""
    while True:
        try:
            oracle_path = task_queue.get_nowait()
        except Exception:
            break

        t0 = time.time()
        result = _run_single_oracle(oracle_path, gpu_idx, timeout=timeout)
        result["elapsed"] = round(time.time() - t0, 1)
        result["gpu"] = gpu_idx
        result_queue.put(result)


def _repro_id_from_path(oracle_path: Path) -> str:
    """Extract repro_id from oracle path (parent directory name)."""
    return oracle_path.parent.name


def main():
    parser = argparse.ArgumentParser(description="Parallel oracle benchmark runner")
    parser.add_argument("--root", type=Path, default=_REPO_ROOT / "repros" / "canonical",
                        help="Root directory to search for oracle_*.py files")
    parser.add_argument("--workers-per-gpu", type=int, default=4,
                        help="Workers per GPU (default: 4). More overlap compilation.")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout per oracle in seconds (default: 120)")
    parser.add_argument("--output", type=Path,
                        default=_REPO_ROOT / "results" / "all_oracle_timings_b200_v2.json",
                        help="Output JSON file")
    parser.add_argument("--resume", action="store_true",
                        help="Skip oracles whose repro_id is already in the output file")
    parser.add_argument("--device-kind", default="B200",
                        help="GPU kind filter (default: B200)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of oracles to run (for testing)")
    args = parser.parse_args()

    # Discover GPUs
    gpus = matching_gpus(args.device_kind)
    if not gpus:
        gpus = discover_gpus()
    print(f"GPUs: {[g['name'] for g in gpus]} (indices: {[g['index'] for g in gpus]})")

    # Find oracles
    oracles = find_oracles(args.root)
    print(f"Found {len(oracles)} oracle files")

    # Resume: load existing results and skip already-measured
    existing_results = {}
    if args.resume and args.output.exists():
        existing_results = json.loads(args.output.read_text())
        before = len(oracles)
        measured_ids = set(existing_results.keys())
        oracles = [o for o in oracles if _repro_id_from_path(o) not in measured_ids]
        print(f"Resume: {before - len(oracles)} already measured, {len(oracles)} remaining")

    if args.limit:
        oracles = oracles[:args.limit]
        print(f"Limited to {len(oracles)} oracles")

    if not oracles:
        print("Nothing to do.")
        return

    # Set up work queue
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for oracle in oracles:
        task_queue.put(str(oracle))

    # Launch workers
    total_workers = len(gpus) * args.workers_per_gpu
    print(f"Launching {total_workers} workers ({args.workers_per_gpu} per GPU)")
    workers = []
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            p = mp.Process(
                target=_worker,
                args=(gpu["index"], task_queue, result_queue, args.timeout),
            )
            p.start()
            workers.append(p)

    # Collect results with progress
    total = len(oracles)
    collected = 0
    ok_count = 0
    fail_count = 0
    results = dict(existing_results)  # start from existing if resuming
    failures = []

    t_start = time.time()
    while collected < total:
        try:
            result = result_queue.get(timeout=300)
        except Exception:
            # Check if all workers are dead
            alive = [w for w in workers if w.is_alive()]
            if not alive:
                print(f"\nAll workers exited. Collected {collected}/{total}")
                break
            continue

        collected += 1
        oracle_path = result.get("path", "?")
        repro_id = _repro_id_from_path(Path(oracle_path))

        if result["status"] == "ok":
            data = result["data"]
            # Use repro_id from the data if available, else from path
            rid = data.pop("repro_id", repro_id)
            results[rid] = data
            ok_count += 1
            ratio = data.get("ratio", 0)
            status = data.get("status", "?")
            elapsed = result.get("elapsed", 0)
            # Progress line
            if collected % 10 == 0 or collected == total:
                elapsed_total = time.time() - t_start
                rate = collected / elapsed_total if elapsed_total > 0 else 0
                eta = (total - collected) / rate if rate > 0 else 0
                print(
                    f"  [{collected}/{total}] {ok_count} ok, {fail_count} fail | "
                    f"{rate:.1f}/s | ETA {eta/60:.0f}m | "
                    f"last: {rid[:30]} {ratio:.2f}x ({status})",
                    flush=True,
                )
        else:
            fail_count += 1
            failures.append({
                "repro_id": repro_id,
                "status": result["status"],
                "error": result.get("error", result.get("stderr", ""))[:200],
            })
            if collected % 10 == 0:
                elapsed_total = time.time() - t_start
                rate = collected / elapsed_total if elapsed_total > 0 else 0
                eta = (total - collected) / rate if rate > 0 else 0
                print(
                    f"  [{collected}/{total}] {ok_count} ok, {fail_count} fail | "
                    f"{rate:.1f}/s | ETA {eta/60:.0f}m",
                    flush=True,
                )

        # Periodic save every 50 results
        if collected % 50 == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2))

    # Wait for workers
    for w in workers:
        w.join(timeout=10)

    # Final save
    elapsed_total = time.time() - t_start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    print(f"\n{'='*60}")
    print(f"Oracle parallel benchmark complete")
    print(f"  Total oracles: {total}")
    print(f"  OK: {ok_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Results saved: {len(results)} entries -> {args.output}")
    print(f"  Wall time: {elapsed_total:.0f}s ({elapsed_total/60:.1f}m)")
    print(f"  Throughput: {collected/elapsed_total:.1f} oracles/s")

    # Summary stats
    if results:
        good = sum(1 for v in results.values() if v.get("status") == "GOOD")
        bad = sum(1 for v in results.values() if v.get("status") == "BAD_ORACLE")
        floor = sum(1 for v in results.values() if v.get("status") == "AT_FLOOR")
        print(f"\n  Status distribution:")
        print(f"    GOOD (oracle faster): {good}")
        print(f"    AT_FLOOR (within 5%): {floor}")
        print(f"    BAD_ORACLE (slower):  {bad}")

    # Save failures log
    if failures:
        fail_path = args.output.with_suffix(".failures.json")
        fail_path.write_text(json.dumps(failures, indent=2))
        print(f"  Failures log: {fail_path}")


if __name__ == "__main__":
    main()
