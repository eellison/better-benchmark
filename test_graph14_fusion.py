"""Analyze graph14: cat + FP8 quantize — what's the fusion structure?"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
from torch._inductor.split_cat_pointwise import split_pointwise_cat_pass
import sympy
cfg.force_disable_caches = True

original_update = GraphLowering._update_scheduler

def detail_reporter(self):
    from torch._inductor.scheduler import Scheduler
    print("  Operations:")
    for i, op in enumerate(self.operations):
        if isinstance(op, ir.ComputedBuffer):
            kind = type(op.data).__name__
            ranges = list(op.data.ranges) if hasattr(op.data, 'ranges') else '?'
            reads = list(op.get_read_names())
            print(f"    [{i}] {op.get_name()}: {kind} ranges={ranges} reads={reads}")
        else:
            print(f"    [{i}] {op.get_name()}: {type(op).__name__}")

    n = split_pointwise_cat_pass(self)
    if n:
        print(f"  Split {n} ops")
        print("  Operations after split:")
        for i, op in enumerate(self.operations):
            if isinstance(op, ir.ComputedBuffer):
                kind = type(op.data).__name__
                ranges = list(op.data.ranges) if hasattr(op.data, 'ranges') else '?'
                reads = list(op.get_read_names())
                print(f"    [{i}] {op.get_name()}: {kind} ranges={ranges} reads={reads}")
            else:
                print(f"    [{i}] {op.get_name()}: {type(op).__name__}")

    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
    nodes = self.scheduler.nodes
    print(f"  Fusion result: {len(nodes)} nodes")
    for nd in nodes:
        if hasattr(nd, 'snodes') and len(nd.snodes) > 1:
            names = [s.node.get_name() for s in nd.snodes if hasattr(s, 'node')]
            print(f"    Fused: {names}")
        else:
            name = nd.node.get_name() if hasattr(nd, 'node') else '?'
            print(f"    Single: {name} ({type(nd).__name__})")


# graph14 pattern: cat([nope_512, pe_64], -1) -> reshape -> f32 -> mul(reciprocal) -> clamp -> fp8
print("=" * 70)
print("graph14: cat([bf16[B,32,512], bf16[B,32,64]], -1) + FP8 quant")
print("=" * 70)
def graph14(nope, pe, scale):
    cat = torch.cat([nope, pe], dim=-1)   # [B, 32, 576]
    flat = cat.reshape(cat.shape[0], -1)  # [B, 18432]
    x = flat.float()
    x = x * scale.reciprocal()
    x = x.clamp(-448.0, 448.0)
    return x.to(torch.float8_e4m3fn)

torch._dynamo.reset()
GraphLowering._update_scheduler = detail_reporter
nope = torch.randn(4, 32, 512, device="cuda", dtype=torch.bfloat16)
pe = torch.randn(4, 32, 64, device="cuda", dtype=torch.bfloat16)
scale = torch.tensor(0.1, device="cuda")
compiled = torch.compile(graph14, dynamic=True)
with torch.no_grad():
    out = compiled(nope, pe, scale)

GraphLowering._update_scheduler = original_update
