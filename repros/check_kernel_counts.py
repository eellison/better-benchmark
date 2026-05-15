"""Compare kernel counts between baseline and split for Qwen3."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
from torch.utils._sympy.functions import Identity
from torch._inductor.sizevars import statically_known_true
import sympy, copy as _copy, importlib.util, os
from torch.utils._sympy.value_ranges import ValueRanges
from functools import reduce
import operator
cfg.force_disable_caches = True

# Import the full pass from v3
sys.path.insert(0, ".")
exec(open("repo/repros/test_all_models_v3.py").read().split("# === Benchmark helper ===")[0])

original_update = GraphLowering._update_scheduler

def counting_update_baseline(self):
    from torch._inductor.scheduler import Scheduler
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
    nodes = self.scheduler.nodes
    fused = [n for n in nodes if hasattr(n, 'snodes') and len(n.snodes) > 1]
    unfused = [n for n in nodes if not hasattr(n, 'snodes') or len(n.snodes) <= 1]
    print(f"  Baseline: {len(nodes)} scheduler nodes ({len(fused)} fused groups, {len(unfused)} single)")
    for n in nodes:
        if hasattr(n, 'snodes'):
            names = [s.node.get_name() for s in n.snodes if hasattr(s, 'node')]
            print(f"    Fused: {names}")
        else:
            name = n.node.get_name() if hasattr(n, 'node') else '?'
            print(f"    Single: {name}")

def counting_update_split(self):
    from torch._inductor.scheduler import Scheduler
    # Run the split pass WITHOUT profitability check
    n_split = 0
    split_any = True
    while split_any:
        split_any = False
        for op in list(self.operations):
            if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
                # Skip profitability — force split everything
                pw = op.data
                ranges = pw.ranges
                ndim = len(ranges)
                var_ranges = {}
                iter_vars = []
                for j, r in enumerate(ranges):
                    sym2 = sympy.Symbol(f"d{j}", integer=True, nonneg=True)
                    var_ranges[sym2] = r
                    iter_vars.append(sym2)
                try:
                    body = LoopBody(pw.inner_fn, (iter_vars,), var_ranges, iter_vars, [])
                except Exception:
                    continue
                masked = [k for k in body.subblocks if k.startswith('masked_subblock')]
                if len(masked) != 2:
                    continue
                if split_pointwise_cat(self, op):
                    split_any = True
                    n_split += 1
                    break
    print(f"  Split {n_split} ops")
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
    nodes = self.scheduler.nodes
    fused = [n for n in nodes if hasattr(n, 'snodes') and len(n.snodes) > 1]
    unfused = [n for n in nodes if not hasattr(n, 'snodes') or len(n.snodes) <= 1]
    print(f"  After split: {len(nodes)} scheduler nodes ({len(fused)} fused groups, {len(unfused)} single)")
    for n in nodes:
        if hasattr(n, 'snodes'):
            names = [s.node.get_name() for s in n.snodes if hasattr(s, 'node')]
            print(f"    Fused: {names}")
        else:
            name = n.node.get_name() if hasattr(n, 'node') else '?'
            print(f"    Single: {name}")

repro_path = "repo/repros/multi_kernel_inference/k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"
spec = importlib.util.spec_from_file_location("repro_mod", repro_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.make_inputs()

print("=== Qwen3-0.6B ===")
torch._dynamo.reset()
GraphLowering._update_scheduler = counting_update_baseline
compiled = torch.compile(mod.Repro())
with torch.no_grad():
    compiled(*inputs)

torch._dynamo.reset()
GraphLowering._update_scheduler = counting_update_split
compiled2 = torch.compile(mod.Repro())
with torch.no_grad():
    compiled2(*inputs)

GraphLowering._update_scheduler = original_update
