"""
Run CUDA-Graph benchmarks (do_bench + graph.replay) on previously-flagged gap kernels.
"""

import glob
import json
import os
import subprocess
import sys

def run_repro(script_path, gpu_id=0, timeout=180):
    """Run a single repro and capture benchmark results."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    wrapper = f"""
import sys, json, os
sys.path.insert(0, '{os.path.dirname(script_path)}')
exec(open('{script_path}').read())
result = benchmark()
print("BENCHMARK_JSON:" + json.dumps(result, default=str))
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if proc.returncode != 0:
            err = proc.stderr[-200:] if proc.stderr else "unknown"
            return None, err
        for line in proc.stdout.split("\n"):
            if line.startswith("BENCHMARK_JSON:"):
                return json.loads(line[len("BENCHMARK_JSON:"):]), None
        return None, "No JSON output"
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "aten_repros")

    # Load the previously-flagged moderate gap kernels
    gap_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moderate_gap_scripts.json")
    if os.path.exists(gap_file):
        targets = json.load(open(gap_file))
        scripts = [(g, s, l) for g, s, l in targets]
    else:
        # Fall back to all repros
        scripts = []
        for path in sorted(glob.glob(os.path.join(base_dir, "**", "region_*.py"), recursive=True)):
            if not path.endswith("_investigation.txt"):
                scripts.append((0, path, os.path.basename(path)))

    print(f"Running {len(scripts)} previously-flagged kernels with do_bench + CUDA Graph...\n")

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    results = []

    for i, (old_gap, script_path, label) in enumerate(scripts):
        # Resolve relative paths
        if not os.path.isabs(script_path):
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_path)

        if not os.path.exists(script_path):
            print(f"[{i+1}/{len(scripts)}] MISSING: {label}")
            continue

        print(f"[{i+1}/{len(scripts)}] {label[:70]}", end="", flush=True)

        result, err = run_repro(script_path, gpu_id=gpu_id)
        if result:
            gap = result["compiled_us"] / result["memcopy_sol_us"] if result["memcopy_sol_us"] > 0 else 999
            result["gap"] = gap
            result["old_gap"] = old_gap
            result["label"] = label
            result["script"] = script_path
            results.append(result)
            print(f"  was {old_gap:.2f}x -> now {gap:.2f}x  ({result['compiled_us']:.1f}/{result['memcopy_sol_us']:.1f} us)")
        else:
            print(f"  FAIL: {err[:50]}")

    if not results:
        print("\nNo successful benchmarks")
        return

    # Summary
    print(f"\n{'='*110}")
    print(f"RESULTS: do_bench + CUDA Graph (eliminates dispatch + L2 cache artifacts)")
    print(f"{'='*110}")
    print(f"{'Label':<60} {'Old Gap':>8} {'New Gap':>8} {'Kernel':>8} {'SOL':>8}")
    print(f"{'-'*110}")
    for r in sorted(results, key=lambda x: -x["gap"]):
        label = r["label"]
        if len(label) > 59:
            label = "..." + label[-56:]
        print(f"{label:<60} {r['old_gap']:>7.2f}x {r['gap']:>7.2f}x {r['compiled_us']:>7.1f} {r['memcopy_sol_us']:>7.1f}")
    print(f"{'='*110}")

    # Classification
    compute_bound = [r for r in results if r["gap"] < 0.8]
    near_sol = [r for r in results if 0.8 <= r["gap"] < 1.3]
    moderate = [r for r in results if 1.3 <= r["gap"] < 2.0]
    large = [r for r in results if r["gap"] >= 2.0]

    print(f"\nClassification:")
    print(f"  Compute-bound (<0.8x): {len(compute_bound)}")
    print(f"  Near SOL (0.8-1.3x):  {len(near_sol)}")
    print(f"  Moderate (1.3-2.0x):   {len(moderate)}")
    print(f"  Large gap (>2.0x):     {len(large)}")

    # Save
    out_path = os.path.join(base_dir, "cudagraph_dobench_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
