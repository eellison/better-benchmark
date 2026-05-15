"""Check the original buf1 layout in Qwen3."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
cfg.force_disable_caches = True
import importlib.util

repro_path = "repo/repros/multi_kernel_inference/k03_Qwen3-0.6B_inference__region_005_mean_c0e8d7954fcb_fcbe8ae0.py"
spec = importlib.util.spec_from_file_location("repro_mod", repro_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
inputs = mod.make_inputs()

original_update = GraphLowering._update_scheduler
def debug_update_scheduler(self):
    from torch._inductor.scheduler import Scheduler
    for op in self.operations:
        if isinstance(op, ir.ComputedBuffer) and isinstance(op.data, ir.Pointwise):
            name = op.get_name()
            layout = op.layout
            print(f"  {name}: layout={layout}")
            if hasattr(layout, 'size') and hasattr(layout, 'stride'):
                print(f"    size={layout.size}, stride={layout.stride}")
            elif hasattr(layout, 'target_size'):
                print(f"    target_size={layout.target_size}")
            else:
                print(f"    type={type(layout).__name__}")
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

GraphLowering._update_scheduler = debug_update_scheduler
torch._dynamo.reset()
compiled = torch.compile(mod.Repro())
with torch.no_grad():
    out = compiled(*inputs)
GraphLowering._update_scheduler = original_update
