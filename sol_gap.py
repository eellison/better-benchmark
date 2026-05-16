#!/usr/bin/env python3
"""
SOL Gap Analysis Tool — Find and diagnose fusion regions that underperform vs memcopy.

Usage:
    # 1. Extract repros from a model
    python sol_gap.py extract --model "meta-llama/Llama-3.2-1B" --mode inference

    # 2. Benchmark all repros (uses do_bench + same-size memcopy)
    python sol_gap.py bench [--dir output/aten_repros/...] [--gpu 0]

    # 3. Show gaps sorted by severity
    python sol_gap.py report [--min-gap 1.3]

    # 4. Investigate a specific kernel (dump generated Triton code)
    python sol_gap.py investigate <repro_path>

    # 5. Run the full pipeline on a model
    python sol_gap.py run --model "meta-llama/Llama-3.2-1B" --mode inference
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output" / "aten_repros"


def cmd_extract(args):
    """Extract fusion region repros from a compiled model."""
    from extract_reductions import ReductionExtractor

    model = args.model
    mode = args.mode
    safe_name = model.replace("/", "_")
    out_dir = str(OUTPUT_DIR / f"vllm_{safe_name}_{mode}")

    print(f"Extracting fusion regions from {model} ({mode})...")
    print(f"Output: {out_dir}")

    # This imports and runs the extraction pipeline
    extractor = ReductionExtractor(out_dir)
    # The extractor hooks into torch.compile — need to compile the model
    # This is model-specific; delegate to extract_vllm.py for vLLM models
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "extract_vllm.py"),
         "--model", model, "--mode", mode, "--output", out_dir],
        check=True
    )
    print(f"Done. Repros saved to {out_dir}/")


def cmd_bench(args):
    """Benchmark repro scripts with do_bench (proper L2 flush + CUDA event timing)."""
    target_dir = args.dir or str(OUTPUT_DIR)
    gpu = args.gpu

    repros = sorted(
        glob.glob(os.path.join(target_dir, "**", "region_*.py"), recursive=True)
        + glob.glob(os.path.join(target_dir, "**", "fused_*.py"), recursive=True)
    )
    repros = [r for r in repros if not r.endswith("_investigation.txt")]

    if not repros:
        print(f"No repro files found in {target_dir}")
        return

    print(f"Benchmarking {len(repros)} repros on GPU {gpu}...")

    results = []
    for i, path in enumerate(repros):
        rel = os.path.relpath(path, str(OUTPUT_DIR))
        label = rel.replace(".py", "")
        print(f"[{i+1}/{len(repros)}] {label[:75]}", end="", flush=True)

        result, err = _run_repro(path, gpu)
        if result:
            gap = result["compiled_us"] / result["memcopy_sol_us"] if result["memcopy_sol_us"] > 0 else 999
            result["gap"] = gap
            result["label"] = label
            result["script"] = path
            results.append(result)
            nk = result.get("n_kernels", "?")
            print(f"  {gap:.2f}x ({result['compiled_us']:.1f}/{result['memcopy_sol_us']:.1f}us) [{nk} kernel(s)]")
        else:
            print(f"  FAIL: {err[:50]}")

    out_path = os.path.join(target_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {out_path}")
    _print_report(results, min_gap=1.3)


def cmd_report(args):
    """Print a summary report from existing benchmark results."""
    results_files = glob.glob(str(OUTPUT_DIR / "**" / "benchmark_results.json"), recursive=True)
    # Also check for the combined file
    combined = OUTPUT_DIR / "cudagraph_dobench_results.json"
    if combined.exists():
        results_files = [str(combined)]

    all_results = []
    for rf in results_files:
        data = json.load(open(rf))
        for r in data:
            if "gap" not in r:
                r["gap"] = r["compiled_us"] / r["memcopy_sol_us"] if r.get("memcopy_sol_us", 0) > 0 else 999
            all_results.append(r)

    if not all_results:
        print("No benchmark results found. Run 'sol_gap.py bench' first.")
        return

    _print_report(all_results, min_gap=args.min_gap)


def cmd_investigate(args):
    """Run a repro with TORCH_LOGS=output_code to dump generated Triton kernels."""
    path = args.repro
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    gpu = args.gpu
    print(f"Investigating {path} on GPU {gpu}...")
    print(f"Capturing generated Triton code...\n")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TORCH_LOGS"] = "output_code"

    # Run just the compilation (not benchmark) to see generated code
    wrapper = f"""
import torch, sys
sys.path.insert(0, '{os.path.dirname(os.path.abspath(path))}')
exec(open('{os.path.abspath(path)}').read())
mod = Repro()
inputs = make_inputs()
compiled = torch.compile(mod)
with torch.no_grad():
    compiled(*inputs)
    torch.cuda.synchronize()
print("\\n=== COMPILATION COMPLETE ===")
"""

    proc = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True, text=True, timeout=300, env=env
    )

    # Save investigation output
    out_path = path.replace(".py", "_investigation.txt")
    with open(out_path, "w") as f:
        f.write(proc.stderr)
        f.write("\n\n=== STDOUT ===\n")
        f.write(proc.stdout)
    print(f"Investigation saved to {out_path}")

    # Print kernel summary
    lines = proc.stderr.split("\n")
    kernel_count = sum(1 for l in lines if "def triton_" in l)
    print(f"\nGenerated {kernel_count} Triton kernel(s)")
    for line in lines:
        if "def triton_" in line:
            print(f"  {line.strip()}")


def cmd_run(args):
    """Full pipeline: extract + bench + report."""
    cmd_extract(args)
    args.dir = str(OUTPUT_DIR / f"vllm_{args.model.replace('/', '_')}_{args.mode}")
    cmd_bench(args)


def _run_repro(script_path, gpu_id=0, timeout=180):
    """Run a single repro and capture benchmark JSON."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    wrapper = f"""
import sys, json, os
sys.path.insert(0, '{os.path.dirname(os.path.abspath(script_path))}')
exec(open('{os.path.abspath(script_path)}').read())
result = benchmark()
print("BENCHMARK_JSON:" + json.dumps(result, default=str))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapper],
            capture_output=True, text=True, timeout=timeout, env=env
        )
        if proc.returncode != 0:
            return None, proc.stderr[-200:] if proc.stderr else "unknown error"
        for line in proc.stdout.split("\n"):
            if line.startswith("BENCHMARK_JSON:"):
                return json.loads(line[len("BENCHMARK_JSON:"):]), None
        return None, "No JSON output"
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    except Exception as e:
        return None, str(e)


def _print_report(results, min_gap=1.3):
    """Print classified summary of benchmark results."""
    compute_bound = [r for r in results if r["gap"] < 0.8]
    near_sol = [r for r in results if 0.8 <= r["gap"] < 1.3]
    moderate = [r for r in results if 1.3 <= r["gap"] < 2.0]
    large = [r for r in results if r["gap"] >= 2.0]

    print(f"\n{'='*90}")
    print(f"SOL Gap Classification ({len(results)} kernels)")
    print(f"{'='*90}")
    print(f"  Compute-bound (<0.8x SOL): {len(compute_bound):>4}")
    print(f"  Near SOL (0.8-1.3x):       {len(near_sol):>4}")
    print(f"  Moderate gap (1.3-2.0x):   {len(moderate):>4}  ← optimization targets")
    print(f"  Large gap (>2.0x):         {len(large):>4}  ← high-priority targets")

    # Fusion analysis: regions that split into multiple kernels
    multi_kernel = [r for r in results if r.get("n_kernels", 1) > 1]
    if multi_kernel:
        print(f"\n  Split regions (>1 kernel):  {len(multi_kernel):>4}  ← missed fusion opportunities")

    targets = [r for r in results if r["gap"] >= min_gap]
    if targets:
        print(f"\n{'Label':<55} {'Kernel':>8} {'SOL':>8} {'Gap':>6} {'#K':>3}")
        print(f"{'-'*90}")
        for r in sorted(targets, key=lambda x: -x["gap"])[:30]:
            label = r.get("label", r.get("script", "?"))
            if len(label) > 54:
                label = "..." + label[-51:]
            nk = r.get("n_kernels", "?")
            print(f"{label:<55} {r['compiled_us']:>7.1f} {r['memcopy_sol_us']:>7.1f} {r['gap']:>5.2f}x {nk:>3}")
    print(f"{'='*90}")


def main():
    parser = argparse.ArgumentParser(description="SOL Gap Analysis Tool")
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract", help="Extract fusion region repros from a model")
    p_extract.add_argument("--model", required=True)
    p_extract.add_argument("--mode", choices=["inference", "training"], default="inference")

    p_bench = sub.add_parser("bench", help="Benchmark repro scripts")
    p_bench.add_argument("--dir", default=None, help="Directory with repro scripts")
    p_bench.add_argument("--gpu", type=int, default=0)

    p_report = sub.add_parser("report", help="Show gap report from existing results")
    p_report.add_argument("--min-gap", type=float, default=1.3)

    p_inv = sub.add_parser("investigate", help="Dump generated Triton code for a repro")
    p_inv.add_argument("repro", help="Path to repro script")
    p_inv.add_argument("--gpu", type=int, default=0)

    p_run = sub.add_parser("run", help="Full pipeline: extract + bench + report")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--mode", choices=["inference", "training"], default="inference")
    p_run.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    cmd_map = {
        "extract": cmd_extract,
        "bench": cmd_bench,
        "report": cmd_report,
        "investigate": cmd_investigate,
        "run": cmd_run,
    }
    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
