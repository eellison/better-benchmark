"""Debug: What does the Qwen3 pointwise_cat look like?"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.virtualized import V, ops
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody, LoopBodyBlock
from torch._inductor.sizevars import statically_known_true
from torch.utils._sympy.functions import Identity
import sympy, copy as _copy, importlib.util
from torch.utils._sympy.value_ranges import ValueRanges
cfg.force_disable_caches = True

repro_path = "repo/repros/multi_kernel_inference/k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"
spec = importlib.util.spec_from_file_location("repro_mod", repro_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.make_inputs()

original_update = GraphLowering._update_scheduler
def debug_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    for i, op in enumerate(self.operations):
        if not isinstance(op, ir.ComputedBuffer) or not isinstance(op.data, ir.Pointwise):
            continue
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
        if len(masked) != 2:
            continue
        print(f"\n{'='*60}")
        print(f"Op {i}: {op.get_name()}")
        print(f"  ranges: {ranges}")
        print(f"  ndim: {ndim}")
        print(f"  masked_subblocks: {masked}")
        print(f"  indexing_exprs: {body.indexing_exprs}")
        print(f"\n  Root graph:")
        body.root_block.graph.print_tabular()
        print(f"\n  Subblocks:")
        for name, block in body.subblocks.items():
            print(f"    {name}:")
            block.graph.print_tabular()

        # Find the split condition
        for node in body.root_block.graph.nodes:
            if node.op == "call_method" and node.target == "lt":
                rhs = node.args[2]
                if rhs.op == "call_method" and rhs.target == "constant":
                    split_point = int(rhs.args[1])
                    print(f"\n  Split point: {split_point}")
                    print(f"  Split dim (last): {ndim-1}, total={int(ranges[ndim-1])}")
                    break

    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)


GraphLowering._update_scheduler = debug_update_scheduler
torch._dynamo.reset()
compiled = torch.compile(mod.Repro())
with torch.no_grad():
    out = compiled(*inputs)

GraphLowering._update_scheduler = original_update
