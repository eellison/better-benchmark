"""Analyze the full DeepSeek graph1 RoPE pipeline fusion."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.split_cat_pointwise import split_pointwise_cat_pass
cfg.force_disable_caches = True

original_update = GraphLowering._update_scheduler

def detail_reporter(self):
    from torch._inductor.scheduler import Scheduler
    print("  Operations (pre-split):")
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
        print(f"\n  Split {n} ops. Operations (post-split):")
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
    print(f"\n  Fusion: {len(nodes)} nodes")
    for nd in nodes:
        if hasattr(nd, 'snodes') and len(nd.snodes) > 1:
            names = [s.node.get_name() for s in nd.snodes if hasattr(s, 'node')]
            print(f"    Fused({len(names)}): {names}")
        else:
            name = nd.node.get_name() if hasattr(nd, 'node') else '?'
            print(f"    Single: {name} ({type(nd).__name__})")


# Full RoPE: cos/sin lookup -> rotate_neox -> apply rope -> cat(q_rope, q_nope) -> output
# This mimics graph1's cat_4: cat([rope_applied_q, q_nope], -1)
print("=" * 70)
print("Graph1 essence: RoPE -> cat to output (no consumer)")
print("  cos/sin lookup -> rotate -> rope_apply -> cat([roped, nope]) -> out")
print("=" * 70)
def graph1_rope_cat(q_nope, q_pe, cos_cache, sin_cache, positions):
    # cos/sin lookup
    cos = cos_cache[positions]  # [B, 32]
    sin = sin_cache[positions]  # [B, 32]
    # reshape pe for rotate
    q_r = q_pe.reshape(q_pe.shape[0], -1, 2)  # [B, 32, 2]
    # rotate_neox
    x1 = q_r[..., :1]
    x2 = q_r[..., 1:]
    rotated = torch.cat([-x2, x1], dim=-1)  # [B, 32, 2]
    # apply RoPE
    roped = (q_r * cos.unsqueeze(-1) + rotated * sin.unsqueeze(-1)).reshape(q_pe.shape[0], -1)
    # final cat to output
    return torch.cat([roped, q_nope], dim=-1)

torch._dynamo.reset()
GraphLowering._update_scheduler = detail_reporter
B = 128
q_nope = torch.randn(B, 64, device="cuda", dtype=torch.bfloat16)
q_pe = torch.randn(B, 64, device="cuda", dtype=torch.bfloat16)
cos_cache = torch.randn(8192, 32, device="cuda", dtype=torch.bfloat16)
sin_cache = torch.randn(8192, 32, device="cuda", dtype=torch.bfloat16)
positions = torch.randint(0, 8192, (B,), device="cuda")
compiled = torch.compile(graph1_rope_cat, dynamic=True)
with torch.no_grad():
    out = compiled(q_nope, q_pe, cos_cache, sin_cache, positions)
ref = graph1_rope_cat(q_nope, q_pe, cos_cache, sin_cache, positions)
print(f"\n  Correctness: max_diff={( ref - out).abs().max().item():.6f}")

GraphLowering._update_scheduler = original_update
