import sys
sys.path.insert(0, '/tmp/pytorch-work')
import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import fresh_inductor_cache
from torch._inductor.codecache import cache_dir
import glob
import os

# Import the repro module
sys.path.insert(0, '/tmp/scratch_space/better_benchmark/repros/canonical/pointwise_22b61ae3e8fe')
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
from repro import Repro, _default_make_inputs

mod = Repro()
inputs = _default_make_inputs()

torch._dynamo.reset()
with fresh_inductor_cache():
    compiled = torch.compile(mod)
    with torch.no_grad():
        compiled(*inputs)
        torch.cuda.synchronize()
    
    cd = cache_dir()
    print(f"Cache dir: {cd}")
    
    # Find all python files
    py_files = sorted(glob.glob(os.path.join(cd, "**", "*.py"), recursive=True), key=os.path.getmtime)
    
    # Look for the main wrapper and kernel files
    for f in py_files:
        with open(f) as fh:
            content = fh.read()
        if 'def call(' in content and '.run(' in content:
            print(f"\n{'='*80}")
            print(f"MAIN WRAPPER: {f}")
            print(f"{'='*80}")
            print(content)
            break
    
    # Also find triton kernel files
    for f in py_files:
        with open(f) as fh:
            content = fh.read()
        if '@triton.jit' in content or '@triton_heuristics' in content or 'triton_poi' in content or 'triton_' in content.split('\n')[0] if content else False:
            basename = os.path.basename(f)
            if basename.startswith('triton_'):
                print(f"\n{'='*80}")
                print(f"TRITON KERNEL: {f}")
                print(f"{'='*80}")
                print(content)
