"""
Test which repros benefit from the pointwise_cat guard relaxation.

The fix (f9493caabcd) removes the is_pointwise_use check that blocked
pointwise_cat when inputs had pointwise consumers. This script compiles
repros with `aten.cat` using the fixed pytorch and compares against the
baseline sweep (measured before the fix).
"""
import sys
sys.path.insert(0, '/tmp/pytorch-work')
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')

import torch
import torch._dynamo
import torch._inductor.config as cfg
import torch._inductor.metrics as metrics
import importlib.util
import math
import tempfile
import os
import json
import traceback
from pathlib import Path
from triton.testing import do_bench

# Load baseline sweep
baseline = json.load(open('/tmp/scratch_space/better_benchmark/sweep_3config_interleaved.json'))

# Find repros with aten.cat
cat_repros = []
canonical_dir = Path('/tmp/scratch_space/better_benchmark/repros/canonical')
for repro_dir in sorted(canonical_dir.iterdir()):
    repro_file = repro_dir / 'repro.py'
    if repro_file.exists():
        content = repro_file.read_text()
        if 'aten.cat' in content:
            cat_repros.append(repro_dir.name)

print(f'Total repros with aten.cat: {len(cat_repros)}')
print('='*80)

helped = []
unchanged = []
regressed = []
errors = []

for i, repro_id in enumerate(cat_repros):
    # Fresh cache for each repro
    cache_dir = tempfile.mkdtemp()
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir

    # Use default config (matching baseline "default" config)
    cfg.coordinate_descent_tuning = False
    cfg.triton.persistent_reductions = True
    cfg.triton.scalar_reduction_accumulators = False

    path = f'/tmp/scratch_space/better_benchmark/repros/canonical/{repro_id}/repro.py'

    try:
        spec = importlib.util.spec_from_file_location(f'repro_{i}', path)
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = math.nan
        spec.loader.exec_module(mod)

        instance = mod.Repro()
        if hasattr(mod, 'make_inputs'):
            inputs = mod.make_inputs()
        elif hasattr(mod, '_default_make_inputs'):
            inputs = mod._default_make_inputs()
        else:
            print(f'  SKIP {repro_id}: no make_inputs')
            continue

        torch._dynamo.reset()
        metrics.reset()
        compiled = torch.compile(instance)
        with torch.no_grad():
            compiled(*inputs)

        # Benchmark
        with torch.no_grad():
            t_us = do_bench(lambda: compiled(*inputs), warmup=25, rep=100, return_mode='min') * 1000

        # Find baseline time
        baseline_key = f'repros/canonical/{repro_id}/repro.py'
        t_base = None
        if baseline_key in baseline:
            cfgs = baseline[baseline_key].get('configs', {})
            # Use the "default" config from baseline since we're testing default config
            if 'default' in cfgs and 'us' in cfgs['default']:
                t_base = cfgs['default']['us']

        if t_base is None:
            print(f'  [{i+1}/{len(cat_repros)}] {repro_id}: {t_us:.1f}us (no baseline)')
            continue

        speedup = t_base / t_us
        delta_pct = (t_base - t_us) / t_base * 100

        if delta_pct > 10:
            helped.append((repro_id, t_base, t_us, speedup))
            print(f'  [{i+1}/{len(cat_repros)}] HELPED: {repro_id} {t_base:.1f} -> {t_us:.1f} us ({speedup:.2f}x, -{delta_pct:.0f}%)')
        elif delta_pct < -10:
            regressed.append((repro_id, t_base, t_us, speedup))
            print(f'  [{i+1}/{len(cat_repros)}] REGRESSED: {repro_id} {t_base:.1f} -> {t_us:.1f} us ({speedup:.2f}x, +{-delta_pct:.0f}%)')
        else:
            unchanged.append((repro_id, t_base, t_us, speedup))
            # Only print a dot for unchanged to avoid spam

    except Exception as e:
        errors.append((repro_id, str(e)))
        # Don't spam with full tracebacks

    # Cleanup
    torch._dynamo.reset()

print('\n' + '='*80)
print(f'\nSUMMARY:')
print(f'  Helped (>10% faster): {len(helped)}')
print(f'  Unchanged: {len(unchanged)}')
print(f'  Regressed (>10% slower): {len(regressed)}')
print(f'  Errors: {len(errors)}')

if helped:
    print(f'\nHELPED repros:')
    helped.sort(key=lambda x: -x[3])  # Sort by speedup
    for repro_id, t_base, t_us, speedup in helped:
        print(f'  {repro_id}: {t_base:.1f} -> {t_us:.1f} us ({speedup:.2f}x)')

if regressed:
    print(f'\nREGRESSED repros:')
    regressed.sort(key=lambda x: x[3])  # Sort by worst first
    for repro_id, t_base, t_us, speedup in regressed:
        print(f'  {repro_id}: {t_base:.1f} -> {t_us:.1f} us ({speedup:.2f}x)')

if errors:
    print(f'\nERRORS ({len(errors)} total):')
    for repro_id, err in errors[:5]:
        print(f'  {repro_id}: {err[:100]}')
