"""Debug: which ops pass the profitability check in Qwen3-0.6B?"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
from torch.utils._sympy.functions import Identity
import sympy, copy as _copy, importlib.util, os
from torch.utils._sympy.value_ranges import ValueRanges
from functools import reduce
import operator
cfg.force_disable_caches = True

def _is_profitable_to_split(self_graph, op):
    buf_name = op.get_name()
    op_numel = reduce(operator.mul, (int(x) for x in op.data.ranges), 1)
    print(f"    Checking profitability for {buf_name}, numel={op_numel}")
    for other_op in self_graph.operations:
        if other_op is op:
            continue
        if not isinstance(other_op, ir.ComputedBuffer):
            continue
        if not isinstance(other_op.data, (ir.Pointwise, ir.Reduction)):
            continue
        read_names = other_op.get_read_names()
        if buf_name not in read_names:
            continue
        other_name = other_op.get_name()
        if isinstance(other_op.data, ir.Pointwise):
            other_numel = reduce(operator.mul, (int(x) for x in other_op.data.ranges), 1)
            print(f"      Consumer {other_name}: Pointwise numel={other_numel} {'BLOCKS' if other_numel == op_numel else 'ok'}")
            if other_numel == op_numel:
                return False
        elif isinstance(other_op.data, ir.Reduction):
            other_numel = reduce(operator.mul, (int(x) for x in other_op.data.ranges), 1)
            print(f"      Consumer {other_name}: Reduction numel={other_numel}")
    return True


original_update = GraphLowering._update_scheduler
def debug_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    print("  All operations:")
    for i, op in enumerate(self.operations):
        if isinstance(op, ir.ComputedBuffer):
            kind = type(op.data).__name__
            ranges = list(op.data.ranges) if hasattr(op.data, 'ranges') else '?'
            numel = reduce(operator.mul, (int(x) for x in op.data.ranges), 1) if hasattr(op.data, 'ranges') else '?'
            reads = list(op.get_read_names())
            print(f"    [{i}] {op.get_name()}: {kind} ranges={ranges} numel={numel} reads={reads}")
    
    # Check which ops would be split candidates
    print("\n  Split candidates (Pointwise with 2 masked subblocks):")
    for op in list(self.operations):
        if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
            pw = op.data
            ranges = pw.ranges
            ndim = len(ranges)
            var_ranges = {}
            iter_vars = []
            for j, r in enumerate(ranges):
                sym = sympy.Symbol(f"d{j}", integer=True, nonneg=True)
                var_ranges[sym] = r
                iter_vars.append(sym)
            try:
                body = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, [])
            except Exception:
                continue
            masked = [k for k in body.subblocks if k.startswith('masked_subblock')]
            if len(masked) == 2:
                print(f"    {op.get_name()}: ranges={list(ranges)} numel={reduce(operator.mul, (int(x) for x in ranges), 1)}")
                _is_profitable_to_split(self, op)
    
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

repro_path = "repo/repros/multi_kernel_inference/k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"
spec = importlib.util.spec_from_file_location("repro_mod", repro_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.make_inputs()

torch._dynamo.reset()
GraphLowering._update_scheduler = debug_update_scheduler
compiled = torch.compile(mod.Repro())
with torch.no_grad():
    compiled(*inputs)
GraphLowering._update_scheduler = original_update
