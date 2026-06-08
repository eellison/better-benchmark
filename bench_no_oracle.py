#!/usr/bin/env python3
"""Benchmark repros without oracles - compile and measure performance."""
import sys
import os
import subprocess
import json

sys.path.insert(0, '.')

REPROS = [
    "repros/canonical/pointwise_531d72f1b34a/repro.py",
    "repros/canonical/pointwise_a12dc8b8a059/repro.py",
    "repros/canonical/sum_sum_f4d29f9ee6ad/repro.py",
    "repros/canonical/sum_sum_sum_0147fd1c4296/repro.py",
    "repros/canonical/sum_sum_sum_109f690634a7/repro.py",
    "repros/canonical/sum_sum_sum_34c857ab7db3/repro.py",
    "repros/canonical/sum_sum_sum_3579253dcf89/repro.py",
    "repros/canonical/sum_sum_sum_41857d0f0554/repro.py",
    "repros/canonical/sum_sum_sum_55426f9a4493/repro.py",
    "repros/canonical/sum_sum_sum_5bff1ad7f52a/repro.py",
    "repros/canonical/sum_sum_sum_6b931086c555/repro.py",
    "repros/canonical/sum_sum_sum_6f9b333ed892/repro.py",
    "repros/canonical/sum_sum_sum_7baf7f118798/repro.py",
    "repros/canonical/sum_sum_sum_9e7a546b859f/repro.py",
    "repros/canonical/sum_sum_sum_b35553d96630/repro.py",
    "repros/canonical/sum_sum_sum_bab40cbb0446/repro.py",
    "repros/canonical/sum_sum_sum_c6666009132a/repro.py",
    "repros/canonical/sum_sum_sum_d06bf12e10d0/repro.py",
    "repros/canonical/sum_sum_sum_f5c107db3be9/repro.py",
    "repros/canonical/var_mean_035991ff3d2b/repro.py",
    "repros/canonical/var_mean_0361de9eae81/repro.py",
    "repros/canonical/var_mean_036b334353a4/repro.py",
    "repros/canonical/var_mean_06924cc70cb4/repro.py",
    "repros/canonical/var_mean_0b554bb6615a/repro.py",
    "repros/canonical/var_mean_0c9738dea136/repro.py",
    "repros/canonical/var_mean_0dc1c0150d30/repro.py",
    "repros/canonical/var_mean_0e7d43725d4d/repro.py",
    "repros/canonical/var_mean_1139e33ee710/repro.py",
    "repros/canonical/var_mean_12b6b115a741/repro.py",
    "repros/canonical/var_mean_13d90438bcbd/repro.py",
    "repros/canonical/var_mean_15e821204e82/repro.py",
    "repros/canonical/var_mean_1711c4ddc910/repro.py",
    "repros/canonical/var_mean_1bab7e80cec1/repro.py",
    "repros/canonical/var_mean_1e1a2b2c1b0a/repro.py",
    "repros/canonical/var_mean_219e8e620fdc/repro.py",
]

BENCH_SCRIPT = '''
import sys; sys.path.insert(0, '.')
import importlib.util, torch, math

spec = importlib.util.spec_from_file_location('repro', '{path}')
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device; mod.inf = math.inf; mod.nan = math.nan
spec.loader.exec_module(mod)
instance = mod.Repro()
inputs = mod.make_inputs()
import torch._inductor.config as cfg
cfg.coordinate_descent_tuning = True
cfg.combo_kernels = True
compiled = torch.compile(instance)
with torch.no_grad():
    for _ in range(3): compiled(*inputs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): compiled(*inputs)
    torch.cuda.synchronize()
from triton.testing import do_bench
t = do_bench(lambda: g.replay(), warmup=25, rep=200, return_mode='min') * 1000
print(f'OK: {{t:.1f}}us')
'''

def run_one(path, timeout=90):
    """Run benchmark for a single repro, return (status, time_us_or_error)."""
    script = BENCH_SCRIPT.format(path=path)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    try:
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=timeout,
            cwd='/tmp/scratch_space/better_benchmark',
            env=env
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if result.returncode == 0 and stdout.startswith('OK:'):
            time_us = stdout.split('OK:')[1].strip()
            return ('OK', time_us)
        else:
            # Get last meaningful line of stderr
            err_lines = [l for l in stderr.split('\n') if l.strip()]
            error = err_lines[-1] if err_lines else f"exit code {result.returncode}"
            # Truncate long errors
            if len(error) > 120:
                error = error[:120] + "..."
            return ('CRASH', error)
    except subprocess.TimeoutExpired:
        return ('TIMEOUT', f'exceeded {timeout}s')
    except Exception as e:
        return ('ERROR', str(e)[:120])

def main():
    results = []
    for i, path in enumerate(REPROS):
        repro_id = path.split('/')[-2]
        print(f"[{i+1}/{len(REPROS)}] {repro_id}...", flush=True)
        status, detail = run_one(path)
        results.append((repro_id, status, detail))
        print(f"  -> {status}: {detail}", flush=True)

    # Write results as JSON for later processing
    with open('investigation_results/repros_without_oracles_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    ok_count = sum(1 for _, s, _ in results if s == 'OK')
    crash_count = sum(1 for _, s, _ in results if s == 'CRASH')
    timeout_count = sum(1 for _, s, _ in results if s == 'TIMEOUT')
    print(f"\nSummary: {ok_count} OK, {crash_count} CRASH, {timeout_count} TIMEOUT out of {len(results)} total")

if __name__ == '__main__':
    main()
