"""Explore the LoopBody FX graph for pointwise_cat (RoPE) to understand
what branch elimination needs to do.

Goal: After splitting into half_lo (d3 ∈ [0,64)), the root graph has:
  - A `call_module masked_subblock0` node that always takes the True branch
  - We want to inline the subblock's computation, eliminating ops.masked/ops.where
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.virtualized import V, ops
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
import sympy
cfg.force_disable_caches = True

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2


class LlamaRoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x2 = x[..., D_HALF:]
        x1 = x[..., :D_HALF]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_e + rotated * sin_e


x_t = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos_t = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')
sin_t = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')

# Capture the graph to get the Pointwise op
from torch._inductor import scheduler as sched

found_body = [None]
found_pw = [None]

original_update = GraphLowering._update_scheduler

def patched_update_scheduler(self):
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
        masked_subblocks = [k for k in body.subblocks if k.startswith('masked_subblock')]
        if len(masked_subblocks) != 2:
            continue

        found_body[0] = body
        found_pw[0] = pw

        print("="*60)
        print("FULL RANGE LoopBody (d3 ∈ [0, 128))")
        print("="*60)
        print("\n--- Root block graph ---")
        print(body.root_block.debug_str("root"))
        print("\n--- Indexing exprs ---")
        for name, expr in body.indexing_exprs.items():
            print(f"  {name}: {expr}")
        print("\n--- Subblocks ---")
        for name, block in body.subblocks.items():
            print(f"\n  {name}:")
            print(f"  {block.debug_str(name)}")

        # Now trace the SAME inner_fn but with restricted range d3 ∈ [0, 64)
        print("\n\n" + "="*60)
        print("HALF RANGE LoopBody (d3 ∈ [0, 64))")
        print("="*60)
        var_ranges_half = dict(var_ranges)
        d3 = iter_vars[3]
        var_ranges_half[d3] = 64  # Restrict to [0, 64)

        body_half = LoopBody(pw.inner_fn, (iter_vars,), var_ranges_half, iter_vars, [])
        print("\n--- Root block graph ---")
        print(body_half.root_block.debug_str("root"))
        print("\n--- Indexing exprs ---")
        for name, expr in body_half.indexing_exprs.items():
            print(f"  {name}: {expr}")
        print("\n--- Subblocks ---")
        for name, block in body_half.subblocks.items():
            print(f"\n  {name}:")
            print(f"  {block.debug_str(name)}")

        break

    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)


GraphLowering._update_scheduler = patched_update_scheduler
torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x_t, cos_t, sin_t)

GraphLowering._update_scheduler = original_update
