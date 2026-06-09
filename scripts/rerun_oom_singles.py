"""
Re-run OOM'd repros from the full corpus sweep, one at a time, to see which
pass when not competing for GPU memory with concurrent workers.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/rerun_oom_singles.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

SWEEP_FILE = Path("results/full_corpus_sweep_b200_2026-06-09.json")
OUTPUT_FILE = Path("results/oom_rerun_singles_b200_2026-06-09.json")
TIMEOUT_SECONDS = 180  # 3 minutes per repro


def get_failed_repros():
    """Get all failed repro paths from the sweep."""
    data = json.load(open(SWEEP_FILE))
    failures = data.get("__failures__", {})
    return sorted(failures.keys())


def run_single_repro(repro_path: str) -> dict:
    """Run a single repro and return timing result or error."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    runner_script = str(_REPO_ROOT / "scripts" / "_single_repro_runner.py")
    cmd = [sys.executable, runner_script, repro_path]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
            env=env, cwd=str(_REPO_ROOT)
        )
        # Parse the last JSON line from stdout
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass
        # If no JSON found, return error
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        return {"status": "failed", "error": f"No JSON output. stderr: {stderr_tail}"}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "timeout (180s)"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def main():
    repros = get_failed_repros()
    print(f"Re-running {len(repros)} failed repros one at a time...")
    print(f"Timeout per repro: {TIMEOUT_SECONDS}s")
    print()

    results = {}
    passed = 0
    failed = 0

    for i, repro_path in enumerate(repros, 1):
        short_name = "/".join(Path(repro_path).parts[-3:])
        print(f"[{i}/{len(repros)}] {short_name} ... ", end="", flush=True)
        t0 = time.time()
        result = run_single_repro(repro_path)
        elapsed = time.time() - t0
        result["elapsed"] = elapsed

        if result["status"] == "ok":
            passed += 1
            print(f"OK ({result['time_us']:.1f} us, compile {result['compile_time_s']:.1f}s)")
        else:
            failed += 1
            err = result.get("error", "unknown")[:80]
            print(f"FAILED ({err})")

        results[repro_path] = result

        # Clear GPU memory between runs
        subprocess.run(
            [sys.executable, "-c", "import torch; torch.cuda.empty_cache()"],
            capture_output=True, timeout=10,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )

    print()
    print(f"={'=' * 60}")
    print(f"Results: {passed}/{len(repros)} passed, {failed}/{len(repros)} failed")
    print(f"={'=' * 60}")

    # Save results
    output = {
        "_metadata": {
            "description": "OOM rerun - single worker, one at a time",
            "source_sweep": str(SWEEP_FILE),
            "total_repros": len(repros),
            "passed": passed,
            "failed": failed,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
        "results": results,
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
