"""Print operations list for Qwen3-0.6B to understand ordering."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
import sympy, importlib.util
cfg.force_disable_caches = True

repro_path = "repo/repros/multi_kernel_inference/k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"
spec = importlib.util.spec_from_file_location("repro_mod", repro_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.make_inputs()

original_update = GraphLowering._update_scheduler
def debug_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    print("Operations list:")
    for i, op in enumerate(self.operations):
        name = op.get_name() if hasattr(op, 'get_name') else '?'
        tp = type(op).__name__
        if isinstance(op, ir.ComputedBuffer):
            data_type = type(op.data).__name__
            if isinstance(op.data, ir.Pointwise):
                ranges = op.data.ranges
                # Check for masked subblocks
                ndim = len(ranges)
                var_ranges = {}
                iter_vars = []
                for j, r in enumerate(ranges):
                    sym = sympy.Symbol(f"d{j}", integer=True, nonneg=True)
                    var_ranges[sym] = r
                    iter_vars.append(sym)
                try:
                    body = LoopBody(op.data.inner_fn, (iter_vars,), var_ranges, iter_vars, [])
                    masked = [k for k in body.subblocks if k.startswith('masked_subblock')]
                    has_cat = len(masked) == 2
                except:
                    has_cat = False
                cat_marker = " *** POINTWISE_CAT ***" if has_cat else ""
                print(f"  [{i}] {name}: {data_type} ranges={ranges}{cat_marker}")
            else:
                print(f"  [{i}] {name}: {data_type}")
                if hasattr(op.data, 'ranges') and hasattr(op.data, 'reduction_ranges'):
                    print(f"       ranges={op.data.ranges} red_ranges={op.data.reduction_ranges}")
        else:
            print(f"  [{i}] {name}: {tp}")

        # Show reads
        if hasattr(op, 'get_read_names'):
            reads = op.get_read_names()
            if reads:
                print(f"       reads: {sorted(reads)}")

    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

GraphLowering._update_scheduler = debug_update_scheduler
torch._dynamo.reset()
compiled = torch.compile(mod.Repro())
with torch.no_grad():
    out = compiled(*inputs)
GraphLowering._update_scheduler = original_update
