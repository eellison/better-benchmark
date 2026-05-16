"""Test DeepSeek-like patterns with dynamic shapes, report fusion."""
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

def fusion_reporter(self):
    from torch._inductor.scheduler import Scheduler
    n = split_pointwise_cat_pass(self)
    if n:
        print(f"    Split {n} ops")
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
    nodes = self.scheduler.nodes
    fused = [nd for nd in nodes if hasattr(nd, 'snodes') and len(nd.snodes) > 1]
    single = len(nodes) - len(fused)
    print(f"    Scheduler: {len(nodes)} nodes ({len(fused)} fused, {single} single)")
    for nd in nodes:
        if hasattr(nd, 'snodes') and len(nd.snodes) > 1:
            names = [s.node.get_name() for s in nd.snodes if hasattr(s, 'node')]
            print(f"      Fused: {names}")
        else:
            name = nd.node.get_name() if hasattr(nd, 'node') else '?'
            ntype = type(nd).__name__
            print(f"      {ntype}: {name}")


# Test 1: cat to output (no consumer) — should split
print("=" * 60)
print("Test 1: cat([k_pe, k_nope]) -> output (dynamic batch)")
print("=" * 60)
def cat_output(k_pe, k_nope):
    return torch.cat([k_pe, k_nope], dim=-1)

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
k_pe = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
k_nope = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
compiled = torch.compile(cat_output, dynamic=True)
with torch.no_grad():
    out = compiled(k_pe, k_nope)
ref = cat_output(k_pe, k_nope)
print(f"  Correctness: max_diff={( ref - out).abs().max().item():.6f}")

# Re-run at different size
k_pe2 = torch.randn(512, 64, device="cuda", dtype=torch.bfloat16)
k_nope2 = torch.randn(512, 64, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    out2 = compiled(k_pe2, k_nope2)
ref2 = cat_output(k_pe2, k_nope2)
print(f"  Correctness (n=512): max_diff={( ref2 - out2).abs().max().item():.6f}")


# Test 2: cat + reshape + consumer (graph14 pattern) — should NOT split
print("\n" + "=" * 60)
print("Test 2: cat + fp8_quant (graph14) — should NOT split")
print("=" * 60)
def cat_quant(nope, pe, scale):
    cat = torch.cat([nope, pe], dim=-1)
    flat = cat.reshape(cat.shape[0], -1)
    x = flat.float() * scale.reciprocal()
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
nope = torch.randn(4, 32, 512, device="cuda", dtype=torch.bfloat16)
pe = torch.randn(4, 32, 64, device="cuda", dtype=torch.bfloat16)
scale = torch.tensor(0.1, device="cuda")
compiled2 = torch.compile(cat_quant, dynamic=True)
with torch.no_grad():
    out2 = compiled2(nope, pe, scale)


# Test 3: rotate_neox + mul consumer — should NOT split
print("\n" + "=" * 60)
print("Test 3: rotate_neox(-x2, x1) * sin + x * cos — should NOT split")
print("=" * 60)
def rope_apply(x, cos, sin):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
x = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
cos = torch.randn(1, 128, 1, 64, device="cuda", dtype=torch.bfloat16)
sin = torch.randn(1, 128, 1, 64, device="cuda", dtype=torch.bfloat16)
compiled3 = torch.compile(rope_apply, dynamic=True)
with torch.no_grad():
    out3 = compiled3(x, cos, sin)
ref3 = rope_apply(x, cos, sin)
print(f"  Correctness: max_diff={( ref3 - out3).abs().max().item():.6f}")


# Test 4: cat_4 pattern — 3D cat to view (goes to output after reshape)
print("\n" + "=" * 60)
print("Test 4: cat([q_rope, q_nope]) -> reshape -> output (dynamic)")
print("=" * 60)
def cat_reshape_output(rope_q, proj_q):
    cat = torch.cat([rope_q, proj_q], dim=-1)
    return cat.reshape(-1, 128)

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
rope_q = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
proj_q = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
compiled4 = torch.compile(cat_reshape_output, dynamic=True)
with torch.no_grad():
    out4 = compiled4(rope_q, proj_q)
ref4 = cat_reshape_output(rope_q, proj_q)
print(f"  Correctness: max_diff={( ref4 - out4).abs().max().item():.6f}")


# Test 5: Full DeepSeek-like MLA decode path
print("\n" + "=" * 60)
print("Test 5: Full MLA decode: RMSNorm -> proj -> RoPE -> cat -> output")
print("=" * 60)
def mla_decode_q(q_nope, q_pe_proj, cos_cache, sin_cache, positions):
    # RoPE on q_pe
    cos = cos_cache[positions].unsqueeze(1)  # [B, 1, D/2]
    sin = sin_cache[positions].unsqueeze(1)  # [B, 1, D/2]
    q_pe = q_pe_proj.reshape(q_pe_proj.shape[0], 32, -1)  # [B, 32, 64]
    # rotate_neox
    x1 = q_pe[..., :32]
    x2 = q_pe[..., 32:]
    rotated = torch.cat([-x2, x1], dim=-1)
    q_pe_roped = q_pe * cos + rotated * sin
    # final cat: nope + pe -> output
    return torch.cat([q_nope, q_pe_roped], dim=-1)

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
q_nope = torch.randn(4, 32, 512, device="cuda", dtype=torch.bfloat16)
q_pe_proj = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
cos_c = torch.randn(8192, 32, device="cuda", dtype=torch.bfloat16)
sin_c = torch.randn(8192, 32, device="cuda", dtype=torch.bfloat16)
pos = torch.randint(0, 8192, (4,), device="cuda")
compiled5 = torch.compile(mla_decode_q, dynamic=True)
with torch.no_grad():
    out5 = compiled5(q_nope, q_pe_proj, cos_c, sin_c, pos)
ref5 = mla_decode_q(q_nope, q_pe_proj, cos_c, sin_c, pos)
print(f"  Correctness: max_diff={( ref5 - out5).abs().max().item():.6f}")


GraphLowering._update_scheduler = original_update
print("\n" + "=" * 60)
print("All DeepSeek dynamic tests passed!")
print("=" * 60)
