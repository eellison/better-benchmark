"""
Re-measure ALL 20 needs_work items with completely fresh caches and all fixes.
Each repro gets its own fresh inductor cache directory and dynamo reset.
"""
import csv
import gc
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import time

import torch
import torch._dynamo
import torch._inductor.config as cfg
from triton.testing import do_bench

sys.path.insert(0, '/tmp/scratch_space/better_benchmark')

# Read needs_work items
with open('/tmp/scratch_space/better_benchmark/investigation_results/oracle_optimization_queue.csv') as f:
    needs = [r for r in csv.DictReader(f) if r.get('impl_status') == 'needs_work']

print(f"Re-measuring {len(needs)} needs_work items on {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print("=" * 120)
print(f"{'repro_id':42s} {'diagnosis':24s} {'compile_us':>12s} {'oracle_us':>12s} {'gap':>8s} {'status':>10s}")
print("-" * 120)

results = []

for row in needs:
    repro_id = row['repro_id']
    diagnosis = row['diagnosis']
    notes = row.get('notes', '')
    path = f'/tmp/scratch_space/better_benchmark/repros/canonical/{repro_id}/repro.py'

    # Fresh cache for each repro
    fresh_cache = tempfile.mkdtemp(prefix=f'inductor_{repro_id}_')
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = fresh_cache

    # Enable all optimization fixes
    cfg.coordinate_descent_tuning = True
    cfg.scatter_reduce_fusion = True
    cfg.triton.scalar_reduction_accumulators = True

    # Reset dynamo completely
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()

    try:
        # Load the repro module
        spec = importlib.util.spec_from_file_location(f'repro_{repro_id}', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Get the Repro class and make inputs
        instance = mod.Repro()

        # Use the module's default make_inputs (parses _shapes_config)
        from repro_harness import parse_shapes_config
        inputs = parse_shapes_config(mod._shapes_config)

        # Compile with fresh cache
        torch._dynamo.reset()
        compiled = torch.compile(instance)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                compiled(*inputs)
            torch.cuda.synchronize()

        # Measure compile time (min over many reps for stable measurement)
        compile_t = do_bench(lambda: compiled(*inputs), warmup=50, rep=200, return_mode='min')
        compile_us = compile_t * 1000  # ms -> us

        # Try to get oracle time from the notes field or oracle module
        oracle_us_str = "N/A"
        gap_str = "N/A"
        oracle_us = None

        # Parse oracle time from notes if available
        if 'oracle=' in notes:
            import re
            m = re.search(r'oracle=(\d+\.?\d*)us', notes)
            if m:
                oracle_us = float(m.group(1))
        elif 'compile=' in notes:
            import re
            m = re.search(r'compile=(\d+\.?\d*)us\s+vs\s+oracle=(\d+\.?\d*)us', notes)
            if m:
                oracle_us = float(m.group(2))

        # Try to load and run the oracle if it exists
        oracle_path = f'/tmp/scratch_space/better_benchmark/repros/canonical/{repro_id}/oracle.py'
        if os.path.exists(oracle_path):
            try:
                oracle_spec = importlib.util.spec_from_file_location(f'oracle_{repro_id}', oracle_path)
                oracle_mod = importlib.util.module_from_spec(oracle_spec)
                oracle_spec.loader.exec_module(oracle_mod)
                if hasattr(oracle_mod, 'oracle_fn'):
                    oracle_inputs = parse_shapes_config(mod._shapes_config)
                    # warmup
                    with torch.no_grad():
                        for _ in range(5):
                            oracle_mod.oracle_fn(*oracle_inputs)
                        torch.cuda.synchronize()
                    oracle_t = do_bench(lambda: oracle_mod.oracle_fn(*oracle_inputs), warmup=50, rep=200, return_mode='min')
                    oracle_us = oracle_t * 1000
            except Exception as oe:
                pass  # Fall back to notes-based oracle

        if oracle_us is not None:
            oracle_us_str = f"{oracle_us:.1f}"
            gap = compile_us / oracle_us
            gap_str = f"{gap:.2f}x"
            if gap <= 1.0:
                status = "CLOSED"
            elif gap <= 1.05:
                status = "~CLOSED"
            else:
                status = "REAL_GAP"
        else:
            status = "NO_ORACLE"

        print(f"  {repro_id:42s} {diagnosis:24s} {compile_us:>10.1f}us {oracle_us_str:>12s} {gap_str:>8s} {status:>10s}")
        results.append({
            'repro_id': repro_id,
            'diagnosis': diagnosis,
            'compile_us': compile_us,
            'oracle_us': oracle_us,
            'gap': compile_us / oracle_us if oracle_us else None,
            'status': status,
            'notes': notes,
        })

    except Exception as e:
        import traceback
        print(f"  {repro_id:42s} {diagnosis:24s} FAILED: {str(e)[:80]}")
        traceback.print_exc()
        results.append({
            'repro_id': repro_id,
            'diagnosis': diagnosis,
            'compile_us': None,
            'oracle_us': None,
            'gap': None,
            'status': 'FAILED',
            'notes': str(e)[:100],
        })

print("\n" + "=" * 120)
print("\nSUMMARY:")
real_gaps = [r for r in results if r['status'] == 'REAL_GAP']
closed = [r for r in results if r['status'] in ('CLOSED', '~CLOSED')]
failed = [r for r in results if r['status'] == 'FAILED']
no_oracle = [r for r in results if r['status'] == 'NO_ORACLE']

print(f"  Real gaps remaining: {len(real_gaps)}")
print(f"  Closed (stale cache): {len(closed)}")
print(f"  Failed to measure:    {len(failed)}")
print(f"  No oracle available:  {len(no_oracle)}")

if real_gaps:
    print(f"\n  Still needs work ({len(real_gaps)}):")
    for r in sorted(real_gaps, key=lambda x: x['gap'] or 0, reverse=True):
        print(f"    {r['repro_id']:42s} {r['diagnosis']:24s} gap={r['gap']:.2f}x  compile={r['compile_us']:.1f}us oracle={r['oracle_us']:.1f}us")

if closed:
    print(f"\n  Now closed ({len(closed)}):")
    for r in closed:
        gap_str = f"{r['gap']:.2f}x" if r['gap'] else "N/A"
        print(f"    {r['repro_id']:42s} {r['diagnosis']:24s} gap={gap_str}  compile={r['compile_us']:.1f}us oracle={r['oracle_us']:.1f}us")
