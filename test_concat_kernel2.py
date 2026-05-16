"""Test with ConcatKernel path by patching the cat lowering."""
import sys
sys.path.insert(0, '/tmp/pytorch-work')
import torch
import torch._inductor.config as inductor_config
from torch._inductor.utils import fresh_inductor_cache
from torch._inductor.codecache import cache_dir
import glob
import os

# Monkey-patch to force ConcatKernel
import torch._inductor.lowering as lowering
original_cat = lowering.cat

def patched_cat(inputs, dim=0):
    from torch._inductor import ir
    from torch._inductor.ir import TensorBox
    dim = lowering._validate_dim(inputs[0], dim, 0)
    dtype = lowering.get_promoted_dtype(
        *inputs, type_promotion_kind=lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    inputs = [lowering.to_dtype(inp, dtype) for inp in inputs]
    return TensorBox(ir.ConcatKernel.create(inputs, dim))

lowering.cat.__wrapped__ = patched_cat  # won't work easily, let's just use force

# Actually, let me use a different approach - just look at what registers the lowering produces
sys.path.insert(0, '/tmp/scratch_space/better_benchmark/repros/canonical/pointwise_22b61ae3e8fe')
sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
from repro import Repro, _default_make_inputs

# Override the registered lowering
from torch._inductor.lowering import register_lowering
from torch import ops as torch_ops

@register_lowering(torch.ops.aten.cat, override=True)
def force_concat_kernel_cat(inputs, dim=0):
    from torch._inductor import ir
    from torch._inductor.ir import TensorBox
    dim = lowering._validate_dim(inputs[0], dim, 0)
    dtype = lowering.get_promoted_dtype(
        *inputs, type_promotion_kind=lowering.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    )
    inputs = [lowering.to_dtype(inp, dtype) for inp in inputs]
    return TensorBox(ir.ConcatKernel.create(inputs, dim))

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
            for r in runs:
                print(f"  {r[:150]}")
            break
