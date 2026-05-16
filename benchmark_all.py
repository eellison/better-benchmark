"""
Batch benchmark runner for extracted reduction kernel repros.

Discovers all reduction_*.py files in the output directory,
runs each one, and produces a summary table + JSON results.
"""

import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def find_repros(base_dir: str) -> list[str]:
    """Find all reduction repro scripts."""
    results = []
    for prefix in ("fused_*", "reduction_*", "region_*"):
        pattern = os.path.join(base_dir, "**", prefix + ".py")
        results.extend(glob.glob(pattern, recursive=True))
    return sorted(set(results))


def run_single_repro(script_path: str, timeout: int = 300) -> dict | None:
    """Run a single repro and capture its JSON result."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "/tmp/pytorch-work"
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    # Modify the script to print JSON results
    wrapper = f"""
import sys, json
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
            print(f"  FAILED: {proc.stderr[-500:]}")
            return None

        # Extract JSON from output
        for line in proc.stdout.split("\n"):
            if line.startswith("BENCHMARK_JSON:"):
                return json.loads(line[len("BENCHMARK_JSON:"):])

        print(f"  No JSON output found")
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    base_dir = os.path.join(os.path.dirname(__file__), "output", "repros")
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]

    repros = find_repros(base_dir)
    if not repros:
        print(f"No repros found in {base_dir}")
        return

    print(f"Found {len(repros)} reduction repros\n")

    results = []
    for path in repros:
        name = os.path.basename(path).replace(".py", "")
        model = os.path.basename(os.path.dirname(path))
        label = f"{model}/{name}"
        print(f"Benchmarking: {label}")

        result = run_single_repro(path)
        if result:
            result["label"] = label
            result["model"] = model
            result["script"] = path
            results.append(result)

    if not results:
        print("\nNo successful benchmarks")
        return

    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Kernel':<65} {'Default':>10} {'CD':>10} {'SOL':>10} {'Gap':>8}")
    print(f"{'':65} {'(us)':>10} {'(us)':>10} {'(us)':>10} {'(D/SOL)':>8}")
    print(f"{'-'*100}")

    for r in results:
        compiled = r["compiled_us"]
        cd = r["coord_descent_us"]
        sol = r["memcopy_sol_us"]
        gap = compiled / sol if sol > 0 else float("inf")
        total_kb = r["total_bytes"] / 1024

        label = r["label"]
        if len(label) > 64:
            label = "..." + label[-61:]

        print(
            f"{label:<65} {compiled:>9.1f} {cd:>9.1f} {sol:>9.1f} {gap:>7.2f}x"
        )

    print(f"{'='*100}")

    # Save results
    out_path = os.path.join(base_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
