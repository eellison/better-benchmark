"""Test with ConcatKernel path by patching."""
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

# Patch the lowering to force ConcatKernel
import torch._inductor.lowering as lowering
import torch._inductor.ir as ir
from torch._inductor.ir import TensorBox

_orig_pointwise_cat = lowering.pointwise_cat
def disabled_pointwise_cat(inputs, dim=0):
    raise RuntimeError("pointwise_cat disabled for testing")
lowering.pointwise_cat = disabled_pointwise_cat

# Also patch the cat function's internal reference
import torch._inductor.lowering
orig_module_dict = torch._inductor.lowering.__dict__

# Simpler: just set force_pointwise_cat to False and MAX to 0
# Actually the cleanest way is to patch the inner condition
# Let me just override the registered function directly
from torch._inductor.lowering import lowerings
def force_concat_cat(inputs, dim=0):
    if len(inputs) == 1:
        return lowering.clone(inputs[0])
    dim = lowering._validate_dim(inputs[0], dim, 0)
    dtype = lowering.get_promoted_dtype(
        *inputs, type_promotion_kind=lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    inputs = [lowering.to_dtype(inp, dtype) for inp in inputs]
    return TensorBox(ir.ConcatKernel.create(inputs, dim))

lowerings[torch.ops.aten.cat.default] = force_concat_cat

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
            runs = [l.strip() for l in content.split('\n') if '.run(' in l and not l.strip().startswith('#')]
            print(f"\nNumber of kernel launches: {len(runs)}")
            for i, r in enumerate(runs):
                print(f"  [{i}] {r[:150]}")
            print("\n--- Kernel names ---")
            for r in runs:
                name = r.split('.run(')[0]
                print(f"  {name}")
            break
