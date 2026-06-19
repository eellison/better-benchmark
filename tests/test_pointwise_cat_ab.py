"""
A/B test for the pointwise_cat guard relaxation.

This script monkey-patches the cat lowering to simulate the old behavior
(with the is_pointwise_use check), then compares against the new behavior
(without it). This isolates just the effect of the pointwise_cat fix.
"""
import sys
sys.path.insert(0, '/tmp/pytorch-work')
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')

import torch
import torch._dynamo
import torch._inductor.config as cfg
import torch._inductor.metrics as metrics
import torch._inductor.lowering as lowering
import importlib.util
import math
import tempfile
import os
import json
import textwrap
import types
from pathlib import Path
from triton.testing import do_bench

# Load baseline sweep for reference
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


def compile_and_bench(repro_id, idx, label=""):
    """Compile a repro and return its runtime in us, or None on error."""
    cache_dir = tempfile.mkdtemp()
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir

    cfg.coordinate_descent_tuning = False
    cfg.triton.persistent_reductions = True
    cfg.triton.scalar_reduction_accumulators = False

    path = f'/tmp/scratch_space/better_benchmark/repros/canonical/{repro_id}/repro.py'

    try:
        spec = importlib.util.spec_from_file_location(f'repro_{label}_{idx}', path)
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
            return None

        torch._dynamo.reset()
        metrics.reset()
        compiled = torch.compile(instance)
        with torch.no_grad():
            compiled(*inputs)

        with torch.no_grad():
            t_us = do_bench(lambda: compiled(*inputs), warmup=25, rep=100, return_mode='min') * 1000

        torch._dynamo.reset()
        return t_us
    except Exception as e:
        torch._dynamo.reset()
        return None


# We need to monkey-patch to simulate old behavior.
# The fix removed these lines from any_input_has_multi_consumers():
#   if any(is_pointwise_use(u) for u in arg.users if u is not current_node):
#       return True
#
# To restore old behavior, we need to inject this check back.
# Strategy: read the source, apply the patch, exec it.

import inspect
import torch._inductor.lowering as _lowering_mod

# Get the source of the cat function
lowering_path = '/tmp/pytorch-work/torch/_inductor/lowering.py'

# Instead of monkey-patching complex internals, let's use force_pointwise_cat config
# and compare kernel counts. Actually, better approach:
#
# Since we can't easily monkey-patch the closure, let's do this:
# Run with current code (fix applied) -> get time A
# Then temporarily revert the file, reimport, run -> get time B
#
# Actually simplest: use subprocess to run each test with/without the patch.

import subprocess

# Create two versions of the lowering file
original_lowering = Path(lowering_path).read_text()

# The fix removed these 4 lines. Let's add them back for the "before" version.
# Looking at the diff:
# The removed code was after "if not hasattr(arg, 'users') or len(arg.users) <= 1: continue"
# and before "# If input is an unrealized Pointwise..."

old_code_to_insert = """\
                # input will be computed multiple times because other consumers
                # (eg. pointwise) will also inline it. So we should realize-in-place via ConcatKernel
                if any(is_pointwise_use(u) for u in arg.users if u is not current_node):
                    return True
"""

# Find the insertion point in the current code
insertion_marker = "                if not hasattr(arg, \"users\") or len(arg.users) <= 1:\n                    continue\n"
after_marker = "                # If input is an unrealized Pointwise"

# Create the "before fix" version
if insertion_marker in original_lowering:
    parts = original_lowering.split(insertion_marker)
    if len(parts) == 2 and after_marker in parts[1]:
        before_version = parts[0] + insertion_marker + old_code_to_insert + parts[1]
        print("Successfully created 'before fix' version of lowering.py")
    else:
        print("ERROR: Could not create before version (marker not unique or after_marker not found)")
        sys.exit(1)
else:
    print("ERROR: Could not find insertion marker")
    sys.exit(1)

# Save the before version
before_path = '/tmp/lowering_before.py'
Path(before_path).write_text(before_version)

# Helper script that will be run as subprocess for each repro
helper_script = '''\
import sys
sys.path.insert(0, '/tmp/pytorch-work')
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')

import torch
import torch._dynamo
import torch._inductor.config as cfg
import importlib.util
import math
import tempfile
import os

from triton.testing import do_bench

# Apply patch if requested
if "--before" in sys.argv:
    # Replace the lowering module's source
    import torch._inductor.lowering
    exec(open('/tmp/lowering_before.py').read(), torch._inductor.lowering.__dict__)

repro_id = sys.argv[1]
cache_dir = tempfile.mkdtemp()
os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir

cfg.coordinate_descent_tuning = False
cfg.triton.persistent_reductions = True
cfg.triton.scalar_reduction_accumulators = False

path = f'/tmp/scratch_space/better_benchmark/repros/canonical/{repro_id}/repro.py'

spec = importlib.util.spec_from_file_location('repro', path)
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
    print("ERROR: no make_inputs")
    sys.exit(1)

torch._dynamo.reset()
compiled = torch.compile(instance)
with torch.no_grad():
    compiled(*inputs)

with torch.no_grad():
    t_us = do_bench(lambda: compiled(*inputs), warmup=25, rep=100, return_mode='min') * 1000

print(f"{t_us:.3f}")
'''

# Actually, the subprocess approach is too slow for 127 repros.
# Let's do it in-process with a simpler approach:
# Just measure "after" (current code), and compare to baseline.
# But we know the baseline includes OTHER changes too.
#
# Better: run each repro twice - once with fix, once without (by patching the file).
# But reloading the module in-process is tricky.
#
# Let's do subprocess but only for the ones that showed changes in the first run.

# From the first run, these repros showed > 10% change:
interesting_repros = [
    # Helped
    'amax_sum_c147ef68d8e3',
    'amax_sum_d112f48ea917',
    'pointwise_2eae25328292',
    'pointwise_79b3932ba286',
    'pointwise_cf3acd87ba9e',
    'pointwise_d011ea1daf34',
    'pointwise_e4cfa8694326',
    'var_mean_4a0c3a2ad6fb',
    'var_mean_525fd66a1241',
    'var_mean_a88cd01397d1',
    'var_mean_mean_1e867cdb78f5',
    'var_mean_var_mean_40a0055bb26e',
    # Regressed (might be from other changes, need to verify)
    'mean_5c93e9826aa8',
    'pointwise_70e5a4aca4b5',
    'var_mean_52ba5f6bf2a0',
    'pointwise_9947dfa4af2d',
    'pointwise_c2e3486ef2f2',
    'var_mean_d3e9f404842a',
    'mean_d0fc206717a8',
]

print(f'\nA/B testing {len(interesting_repros)} interesting repros...')
print('='*80)

# Write helper script
helper_path = '/tmp/bench_helper.py'
Path(helper_path).write_text(helper_script)

results = []
for repro_id in interesting_repros:
    # Run "after" (with fix)
    try:
        after_result = subprocess.run(
            [sys.executable, helper_path, repro_id],
            capture_output=True, text=True, timeout=120
        )
        t_after = float(after_result.stdout.strip()) if after_result.returncode == 0 else None
    except:
        t_after = None

    # Run "before" (without fix - old behavior restored)
    try:
        before_result = subprocess.run(
            [sys.executable, helper_path, repro_id, '--before'],
            capture_output=True, text=True, timeout=120
        )
        t_before = float(before_result.stdout.strip()) if before_result.returncode == 0 else None
    except:
        t_before = None

    if t_after is not None and t_before is not None:
        speedup = t_before / t_after
        delta_pct = (t_before - t_after) / t_before * 100
        status = 'HELPED' if delta_pct > 5 else ('REGRESSED' if delta_pct < -5 else 'NEUTRAL')
        results.append((repro_id, t_before, t_after, speedup, status))
        print(f'  {status:10s} {repro_id}: before={t_before:.1f}us after={t_after:.1f}us ({speedup:.2f}x)')
    else:
        err_msg = ""
        if t_after is None:
            err_msg += f" after_err={after_result.stderr[-200:] if after_result else 'timeout'}"
        if t_before is None:
            err_msg += f" before_err={before_result.stderr[-200:] if before_result else 'timeout'}"
        print(f'  ERROR     {repro_id}:{err_msg[:150]}')

print('\n' + '='*80)
print('FINAL A/B RESULTS (isolated pointwise_cat fix effect):')
print('='*80)

helped_ab = [(r, b, a, s) for r, b, a, s, st in results if st == 'HELPED']
regressed_ab = [(r, b, a, s) for r, b, a, s, st in results if st == 'REGRESSED']
neutral_ab = [(r, b, a, s) for r, b, a, s, st in results if st == 'NEUTRAL']

print(f'\n  Helped (>5% faster with fix): {len(helped_ab)}')
if helped_ab:
    helped_ab.sort(key=lambda x: -x[3])
    for r, b, a, s in helped_ab:
        print(f'    {r}: {b:.1f} -> {a:.1f} us ({s:.2f}x)')

print(f'\n  Regressed (>5% slower with fix): {len(regressed_ab)}')
if regressed_ab:
    regressed_ab.sort(key=lambda x: x[3])
    for r, b, a, s in regressed_ab:
        print(f'    {r}: {b:.1f} -> {a:.1f} us ({s:.2f}x)')

print(f'\n  Neutral: {len(neutral_ab)}')
