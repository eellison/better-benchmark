"""Benchmark 13 oracle repros with fresh cache + all fixes."""
import subprocess
import sys
import os
import tempfile
import json

REPROS = [
    "amax_sum_any_d811783d738f",
    "amax_sum_d112f48ea917",
    "sum_f160d1f03c1f",
    "sum_sum_63e248035ceb",
    "sum_sum_6f518556b3ea",
    "sum_sum_a3ce72227d9d",
    "sum_sum_adb09ad67b46",
    "sum_sum_sum_43d9d6c4646b",
    "sum_sum_sum_73a23287990b",
    "sum_sum_sum_7ebfef460635",
    "sum_sum_sum_995d991d6527",
    "sum_sum_sum_dd4c673f0928",
    "sum_sum_sum_e2d4961f3571",
]

BASE_DIR = "/tmp/scratch_space/better_benchmark/repros/canonical"

BENCH_SCRIPT_TEMPLATE = '''
import torch, torch._dynamo, torch._inductor.config as cfg, importlib.util, math, sys, tempfile, os
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
from triton.testing import do_bench

os.environ['TORCHINDUCTOR_CACHE_DIR'] = tempfile.mkdtemp()
cfg.coordinate_descent_tuning = True
cfg.scatter_reduce_fusion = True
cfg.triton.scalar_reduction_accumulators = True
cfg.linear_reduction_elimination = True
cfg.slice_scatter_elision = True

import importlib.util
repro_path = "{repro_path}"
spec = importlib.util.spec_from_file_location("repro_mod", repro_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Repro().cuda()
inputs = mod.make_inputs()
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

# Compile
torch._dynamo.reset()
compiled = torch.compile(model)
with torch.no_grad():
    for _ in range(3):
        compiled(*inputs)
    torch.cuda.synchronize()

# Benchmark compile
def run_compiled():
    return compiled(*inputs)

compile_us = do_bench(run_compiled, warmup=25, rep=100) * 1000
print(f"COMPILE_US: {{compile_us:.3f}}")
'''

ORACLE_BENCH_TEMPLATE = '''
import torch, torch._dynamo, torch._inductor.config as cfg, importlib.util, math, sys, tempfile, os
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
from triton.testing import do_bench

os.environ['TORCHINDUCTOR_CACHE_DIR'] = tempfile.mkdtemp()

import importlib.util

# Load oracle
oracle_path = "{oracle_path}"
spec = importlib.util.spec_from_file_location("oracle_mod", oracle_path)
oracle_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle_mod)

# Load repro for inputs
repro_path = "{repro_path}"
spec2 = importlib.util.spec_from_file_location("repro_mod", repro_path)
repro_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(repro_mod)

inputs = repro_mod.make_inputs()
inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

# Find oracle function - look for functions starting with 'oracle_'
oracle_fn = None
for name in dir(oracle_mod):
    if name.startswith('oracle_'):
        oracle_fn = getattr(oracle_mod, name)
        break

if oracle_fn is None:
    print("ORACLE_US: NO_ORACLE_FN")
    sys.exit(0)

# Benchmark oracle
try:
    with torch.no_grad():
        # warmup
        for _ in range(3):
            oracle_fn(*inputs)
        torch.cuda.synchronize()

    def run_oracle():
        return oracle_fn(*inputs)

    oracle_us = do_bench(run_oracle, warmup=25, rep=100) * 1000
    print(f"ORACLE_US: {{oracle_us:.3f}}")
except Exception as e:
    print(f"ORACLE_US: ERROR: {{e}}")
'''


def run_bench(repro_id):
    repro_path = os.path.join(BASE_DIR, repro_id, "repro.py")

    # Find oracle file
    oracle_files = [f for f in os.listdir(os.path.join(BASE_DIR, repro_id))
                    if f.startswith("oracle_") and f.endswith(".py")]
    oracle_path = os.path.join(BASE_DIR, repro_id, oracle_files[0]) if oracle_files else None

    results = {"repro_id": repro_id, "oracle_file": oracle_files[0] if oracle_files else None}

    # Run compile benchmark
    script = BENCH_SCRIPT_TEMPLATE.format(repro_path=repro_path)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
        )
        output = proc.stdout + proc.stderr
        for line in output.split("\n"):
            if "COMPILE_US:" in line:
                val = line.split("COMPILE_US:")[1].strip()
                results["compile_us"] = float(val)
                break
        else:
            results["compile_us"] = None
            results["compile_error"] = output[-500:] if len(output) > 500 else output
    except Exception as e:
        results["compile_us"] = None
        results["compile_error"] = str(e)

    # Run oracle benchmark
    if oracle_path:
        script = ORACLE_BENCH_TEMPLATE.format(oracle_path=oracle_path, repro_path=repro_path)
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=120,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            )
            output = proc.stdout + proc.stderr
            for line in output.split("\n"):
                if "ORACLE_US:" in line:
                    val = line.split("ORACLE_US:")[1].strip()
                    if "ERROR" in val or "NO_ORACLE" in val:
                        results["oracle_us"] = None
                        results["oracle_error"] = val
                    else:
                        results["oracle_us"] = float(val)
                    break
            else:
                results["oracle_us"] = None
                results["oracle_error"] = output[-500:] if len(output) > 500 else output
        except Exception as e:
            results["oracle_us"] = None
            results["oracle_error"] = str(e)
    else:
        results["oracle_us"] = None
        results["oracle_error"] = "no oracle file"

    return results


if __name__ == "__main__":
    all_results = []
    for repro_id in REPROS:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {repro_id}")
        print(f"{'='*60}")
        r = run_bench(repro_id)
        all_results.append(r)

        compile_str = f"{r['compile_us']:.1f}us" if r.get('compile_us') else "FAIL"
        oracle_str = f"{r['oracle_us']:.1f}us" if r.get('oracle_us') else r.get('oracle_error', 'N/A')
        ratio = ""
        if r.get('compile_us') and r.get('oracle_us'):
            ratio = f" (ratio: {r['compile_us']/r['oracle_us']:.2f}x)"
        print(f"  compile={compile_str}  oracle={oracle_str}{ratio}")

    print("\n\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Repro ID':<35} {'Compile(us)':>12} {'Oracle(us)':>12} {'Ratio':>8} {'Classification'}")
    print("-"*80)
    for r in all_results:
        c = r.get('compile_us')
        o = r.get('oracle_us')
        c_str = f"{c:.1f}" if c else "FAIL"
        o_str = f"{o:.1f}" if o else "N/A"
        if c and o:
            ratio = c / o
            ratio_str = f"{ratio:.2f}x"
            if ratio <= 1.05:
                classification = "AT_FLOOR"
            else:
                classification = "NEEDS_WORK"
        else:
            ratio_str = "N/A"
            classification = "UNKNOWN"
        print(f"{r['repro_id']:<35} {c_str:>12} {o_str:>12} {ratio_str:>8} {classification}")

    # Save results
    with open("/tmp/scratch_space/better_benchmark/bench_13_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to bench_13_results.json")
