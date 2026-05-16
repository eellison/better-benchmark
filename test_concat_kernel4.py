"""Check the full wrapper code for the ConcatKernel path."""
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

import torch._inductor.lowering as lowering
import torch._inductor.ir as ir
from torch._inductor.ir import TensorBox
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
            # Print the buffer allocation and kernel call section
            lines = content.split('\n')
            in_call = False
            for line in lines:
                if 'def call(' in line:
                    in_call = True
                if in_call:
                    if 'buf' in line or '.run(' in line or 'reinterpret' in line or 'return' in line or 'empty_strided' in line:
                        print(line)
                    if line.strip().startswith('return'):
                        break
            break
