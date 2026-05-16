"""Test what happens with ConcatKernel path (disable pointwise_cat)."""
import sys
sys.path.insert(0, '/tmp/pytorch-work')
import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import fresh_inductor_cache
from torch._inductor.codecache import cache_dir
import glob
import os

sys.path.insert(0, '/tmp/scratch_space/better_benchmark/repros/canonical/pointwise_22b61ae3e8fe')
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
from repro import Repro, _default_make_inputs

# Force ConcatKernel path by setting max_pointwise_cat_inputs to 0
inductor_config.max_pointwise_cat_inputs = 0

mod = Repro()
inputs = _default_make_inputs()

torch._dynamo.reset()
with fresh_inductor_cache():
    compiled = torch.compile(mod)
    with torch.no_grad():
        compiled(*inputs)
        torch.cuda.synchronize()
    
    cd = cache_dir()
    py_files = sorted(glob.glob(os.path.join(cd, "**", "*.py"), recursive=True), key=os.path.getmtime)
    
    for f in py_files:
        with open(f) as fh:
            content = fh.read()
        if 'def call(' in content and '.run(' in content:
            print(f"MAIN WRAPPER: {f}")
            # Count .run() calls
            runs = [l.strip() for l in content.split('\n') if '.run(' in l and not l.strip().startswith('#')]
            print(f"\nNumber of kernel launches: {len(runs)}")
            for r in runs:
                print(f"  {r[:120]}")
            print("\n--- Full wrapper ---")
            # Print just the call function
            in_call = False
            for line in content.split('\n'):
                if 'def call(' in line:
                    in_call = True
                if in_call:
                    print(line)
                    if line.strip().startswith('return'):
                        break
            break
