"""
Run DeepSeek post-grad graphs through inductor and report fusion decisions.

We parse the graph files, substitute concrete sizes for symbolic dims,
skip custom vllm ops by using simple graphs, and feed them through inductor
to see how the scheduler fuses them.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.loop_body import LoopBody
import sympy
from functools import reduce
import operator

cfg.force_disable_caches = True

# We can't easily run the full graphs with custom ops,
# but we CAN create equivalent torch functions that reproduce
# the cat patterns and their consumers, then compile them.

# From the graph analysis, the key patterns in DeepSeek are:
#
# 1. graph14: cat([nope_512, pe_64], dim=-1) -> reshape -> cast -> mul -> clamp -> cast
#    Shape: [batch, 32, 576] where 576 = 512 + 64
#    Consumer: reshape (view) -- which fuses as pointwise
#
# 2. graph1-9 cat_5: cat([k_pe_squeezed, k_nope], dim=-1) -> output
#    Shape: [tokens, 128] where 128 = 64 + 64
#    No consumer (goes to output)
#
# 3. graph1-9 cat_4: cat([rope_q, proj_q], dim=-1) -> view -> (output/matmul)
#    Shape: [tokens, 64, 128] where 128 = 64 + 64
#    Consumer: view/reshape
#
# 4. graph1-9 cat: cat([cos_part, sin_part], dim=-1)
#    Shape: [tokens, 32, 32, 2] -- interleaved cos/sin for RoPE
#    Many consumers (all the RoPE muls)
#
# 5. graph1-9 cat_2: cat([-x2, x1], dim=-1) -- rotate_neox
#    Shape: [tokens, 64, 64] where 64 = 32 + 32
#    Consumer: mul (RoPE application)

# Let's create concrete versions of these patterns and compile them

original_update = GraphLowering._update_scheduler

def fusion_reporter(self):
    from torch._inductor.scheduler import Scheduler

    # Report pointwise_cat candidates
    candidates = []
    for op in self.operations:
        if not isinstance(op, ir.ComputedBuffer):
            continue
        if not isinstance(op.data, ir.Pointwise):
            continue
        pw = op.data
        ranges = pw.ranges
        if any(not isinstance(r, (int, sympy.Integer)) for r in ranges):
            continue
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
        masked = [k for k in body.subblocks if k.startswith("masked_subblock")]
        if len(masked) == 2:
            numel = reduce(operator.mul, (int(x) for x in ranges), 1)
            buf_name = op.get_name()
            reads = list(op.get_read_names())
            candidates.append((buf_name, list(ranges), numel, reads))

    if candidates:
        print(f"  pointwise_cat candidates:")
        for name, ranges, numel, reads in candidates:
            print(f"    {name}: ranges={ranges} numel={numel} reads={reads}")

    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)

    # Report fusion groups
    nodes = self.scheduler.nodes
    fused = [n for n in nodes if hasattr(n, 'snodes') and len(n.snodes) > 1]
    print(f"  Scheduler: {len(nodes)} nodes ({len(fused)} fused)")
    for n in nodes:
        if hasattr(n, 'snodes') and len(n.snodes) > 1:
            names = [s.node.get_name() for s in n.snodes if hasattr(s, 'node')]
            print(f"    Fused: {names}")


# === Pattern 1: graph14 — MLA decode_q cat + FP8 quantize ===
print("=" * 70)
print("Pattern 1: graph14 — cat([nope_512, pe_64], dim=-1) + FP8 quant")
print("  DeepSeek MLA: decode_q0 = cat(decode_ql_nope, decode_q_pe)")
print("=" * 70)

def graph14_pattern(nope, pe, scale):
    cat = torch.cat([nope, pe], dim=-1)
    flat = cat.reshape(cat.shape[0], -1)
    x = flat.to(torch.float32)
    x = x * scale.reciprocal()
    x = x.clamp(-448.0, 448.0)
    x = x.to(torch.float8_e4m3fn)
    return x.reshape(cat.shape[0], 32, 576)

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
nope = torch.randn(4, 32, 512, device="cuda", dtype=torch.bfloat16)
pe = torch.randn(4, 32, 64, device="cuda", dtype=torch.bfloat16)
scale = torch.tensor(0.1, device="cuda", dtype=torch.float32)
compiled14 = torch.compile(graph14_pattern)
with torch.no_grad():
    compiled14(nope, pe, scale)


# === Pattern 2: graph1-9 cat_5 — k = cat([k_pe, k_nope]) to output ===
print("\n" + "=" * 70)
print("Pattern 2: cat_5 — cat([k_pe, k_nope], dim=-1) -> output")
print("  DeepSeek MLA: k = cat(k_pe.squeeze(-2), k_nope)")
print("=" * 70)

def cat5_pattern(k_pe, k_nope):
    return torch.cat([k_pe, k_nope], dim=-1)

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
k_pe = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
k_nope = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
compiled5 = torch.compile(cat5_pattern)
with torch.no_grad():
    compiled5(k_pe, k_nope)


# === Pattern 3: graph1-9 cat_4 — cat([rope_q, proj_q]) -> view -> output ===
print("\n" + "=" * 70)
print("Pattern 3: cat_4 — cat([rope_q, proj_q], dim=-1) -> view")
print("  DeepSeek MLA: q = cat(q_rope, q_nope) for attention")
print("=" * 70)

def cat4_pattern(rope_q, proj_q):
    cat = torch.cat([rope_q, proj_q], dim=-1)
    return cat.reshape(-1, 128)

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
rope_q = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
proj_q = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
compiled4 = torch.compile(cat4_pattern)
with torch.no_grad():
    compiled4(rope_q, proj_q)


# === Pattern 4: rotate_neox cat + mul consumer ===
print("\n" + "=" * 70)
print("Pattern 4: cat_2/cat_3 — rotate_neox + RoPE mul")
print("  cat([-x2, x1], dim=-1) * sin + x * cos")
print("=" * 70)

def rope_pattern(x, cos, sin):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
x = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
cos = torch.randn(1, 128, 1, 64, device="cuda", dtype=torch.bfloat16)
sin = torch.randn(1, 128, 1, 64, device="cuda", dtype=torch.bfloat16)
compiled_rope = torch.compile(rope_pattern)
with torch.no_grad():
    compiled_rope(x, cos, sin)


# === Pattern 5: cos/sin interleave cat (many consumers) ===
print("\n" + "=" * 70)
print("Pattern 5: cat — cos/sin interleave [tokens, 32, 32, 2]")
print("  cat([cos.unsqueeze, sin.unsqueeze], dim=-1) with many consumers")
print("=" * 70)

def cos_sin_interleave(cos_half, sin_half, q, k):
    # Interleave cos and sin
    cos_sin = torch.cat([cos_half.unsqueeze(-1), sin_half.unsqueeze(-1)], dim=-1)
    # cos_sin shape: [tokens, 32, 32, 2] -> view to [tokens, 32, 64]
    cs = cos_sin.reshape(cos_sin.shape[0], cos_sin.shape[1], -1)
    # Apply to both q and k (multiple consumers of cat)
    q_rot = q * cs
    k_rot = k * cs
    return q_rot, k_rot

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
cos_half = torch.randn(128, 32, 32, device="cuda", dtype=torch.bfloat16)
sin_half = torch.randn(128, 32, 32, device="cuda", dtype=torch.bfloat16)
q = torch.randn(128, 32, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn(128, 32, 64, device="cuda", dtype=torch.bfloat16)
compiled_cs = torch.compile(cos_sin_interleave)
with torch.no_grad():
    compiled_cs(cos_half, sin_half, q, k)


# === Full RoPE + cat pattern (graph1 essence) ===
print("\n" + "=" * 70)
print("Pattern 6: Full RoPE pipeline — RMSNorm + RoPE + cat to output")
print("  Like Mistral/Qwen patterns from earlier benchmarks")
print("=" * 70)

def full_rope_cat(x, w_rope, cos_cache, sin_cache, positions):
    # Simulate: proj -> rope -> cat
    # x: [tokens, hidden]
    # Project to get q_rope_part and q_nope_part
    q_proj = x @ w_rope.t()  # [tokens, 128] -> split into 64+64
    q_rope = q_proj[:, :64]
    q_nope = q_proj[:, 64:]

    # Apply RoPE
    cos = cos_cache[positions]  # [tokens, 32]
    sin = sin_cache[positions]  # [tokens, 32]

    # rotate_neox on q_rope (viewed as [tokens, 32, 2] -> interleave)
    qr = q_rope.reshape(-1, 32, 2)
    x1 = qr[..., 0:1]
    x2 = qr[..., 1:2]
    rotated = torch.cat([-x2, x1], dim=-1)
    qr_rotated = qr * cos.unsqueeze(-1) + rotated * sin.unsqueeze(-1)
    q_rope_out = qr_rotated.reshape(-1, 64)

    # Final cat to output (no consumer)
    return torch.cat([q_rope_out, q_nope], dim=-1)

torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
x6 = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
w_rope6 = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
cos_cache6 = torch.randn(4096, 32, device="cuda", dtype=torch.bfloat16)
sin_cache6 = torch.randn(4096, 32, device="cuda", dtype=torch.bfloat16)
positions6 = torch.randint(0, 4096, (128,), device="cuda")
compiled6 = torch.compile(full_rope_cat)
with torch.no_grad():
    compiled6(x6, w_rope6, cos_cache6, sin_cache6, positions6)


GraphLowering._update_scheduler = original_update
