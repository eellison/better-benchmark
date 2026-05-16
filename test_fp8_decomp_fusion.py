"""
Test: decompose per_token_group_fp8_quant and check what fuses.

The DeepSeek pattern is:
  cat_4([q_pe, q_nope], dim=-1) → view(-1, 128) → per_token_group_fp8_quant

With the C op, fusion stops at the barrier. With the decomposition,
inductor should fuse cat+reshape+quant into one kernel.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
from torch._inductor.split_cat_pointwise import split_pointwise_cat_pass
import sympy

cfg.force_disable_caches = True

original_update = GraphLowering._update_scheduler


def fusion_reporter(self):
    from torch._inductor.scheduler import Scheduler
    # NO split_cat_pointwise pass — vanilla inductor
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
    nodes = self.scheduler.nodes
    fused = [nd for nd in nodes if hasattr(nd, 'snodes') and len(nd.snodes) > 1]
    single = len(nodes) - len(fused)
    print(f"    Scheduler: {len(nodes)} nodes ({len(fused)} fused, {single} single)")
    for nd in nodes:
        if hasattr(nd, 'snodes') and len(nd.snodes) > 1:
            names = [s.node.get_name() for s in nd.snodes if hasattr(s, 'node')]
            print(f"      Fused({len(nd.snodes)}): {names[:6]}{'...' if len(names) > 6 else ''}")
        else:
            name = nd.node.get_name() if hasattr(nd, 'node') else '?'
            ntype = type(nd).__name__
            print(f"      {ntype}: {name}")


# ============================================================
# per_token_group_fp8_quant decomposition (group_size=128)
# ============================================================
def per_token_group_fp8_quant_decomp(x, group_size=128, eps=1e-10):
    """Decomposed per_token_group_fp8_quant in pure aten ops."""
    orig_shape = x.shape
    # Reshape to [..., num_groups, group_size]
    assert x.shape[-1] % group_size == 0
    x_grouped = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    x_f32 = x_grouped.float()
    # Per-group absmax scale
    amax = x_f32.abs().amax(dim=-1, keepdim=True)  # [..., num_groups, 1]
    scale = amax / 448.0
    scale = scale.clamp(min=eps)
    # Quantize
    x_scaled = x_f32 / scale
    x_clamped = x_scaled.clamp(-448.0, 448.0)
    x_fp8 = x_clamped.to(torch.float8_e4m3fn)
    # Reshape back
    x_fp8 = x_fp8.reshape(orig_shape)
    scale = scale.reshape(*orig_shape[:-1], orig_shape[-1] // group_size)
    return x_fp8, scale


# ============================================================
# Test 1: Standalone decomposed quant
# ============================================================
print("=" * 70)
print("Test 1: Standalone per_token_group_fp8_quant decomposition")
print("=" * 70)


def standalone_quant(x):
    x_fp8, scale = per_token_group_fp8_quant_decomp(x, group_size=128)
    return x_fp8, scale


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
compiled = torch.compile(standalone_quant, dynamic=True)
with torch.no_grad():
    out_fp8, out_scale = compiled(x)

# Correctness check
ref_fp8, ref_scale = per_token_group_fp8_quant_decomp(x, group_size=128)
print(f"  Correctness: fp8 match={torch.equal(out_fp8, ref_fp8)}, "
      f"scale max_diff={( ref_scale - out_scale).abs().max().item():.8f}")


# ============================================================
# Test 2: cat + reshape + decomposed quant (THE KEY PATTERN)
# ============================================================
print("\n" + "=" * 70)
print("Test 2: cat([q_pe, q_nope]) → view(-1, 128) → fp8_quant (DeepSeek MLA)")
print("  This is the graph1-9 pattern that was blocked by the opaque C op")
print("=" * 70)


def cat_reshape_quant(q_pe, q_nope):
    # cat_4: [tokens, 64, 128] = cat([q_pe(64), q_nope(64)], dim=-1)
    cat = torch.cat([q_pe, q_nope], dim=-1)
    # view_11: [64*tokens, 128] = reshape(cat, [-1, 128])
    flat = cat.reshape(-1, 128)
    # per_token_group_fp8_quant decomposed
    x_fp8, scale = per_token_group_fp8_quant_decomp(flat, group_size=128)
    return x_fp8, scale


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
q_pe = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
q_nope = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
compiled2 = torch.compile(cat_reshape_quant, dynamic=True)
with torch.no_grad():
    out2_fp8, out2_scale = compiled2(q_pe, q_nope)

ref2_fp8, ref2_scale = cat_reshape_quant(q_pe, q_nope)
print(f"  Correctness: fp8 match={torch.equal(out2_fp8, ref2_fp8)}, "
      f"scale max_diff={( ref2_scale - out2_scale).abs().max().item():.8f}")


# ============================================================
# Test 3: Full MLA decode path: RoPE → cat → view → quant
# ============================================================
print("\n" + "=" * 70)
print("Test 3: Full MLA q path: RoPE(q_pe) → cat(q_pe, q_nope) → view → quant")
print("  Full pipeline from graph1 line 130-176")
print("=" * 70)


def mla_q_path(q_pe_raw, q_nope, cos, sin):
    # rotate_neox on q_pe: [tokens, 64, 64]
    x1 = q_pe_raw[..., :32]
    x2 = q_pe_raw[..., 32:]
    rotated = torch.cat([-x2, x1], dim=-1)
    q_pe_roped = q_pe_raw * cos + rotated * sin

    # cat q = [q_pe_roped, q_nope]: [tokens, 64, 128]
    q = torch.cat([q_pe_roped, q_nope], dim=-1)

    # view to flat: [64*tokens, 128]
    q_flat = q.reshape(-1, 128)

    # decomposed fp8 quant
    x_fp8, scale = per_token_group_fp8_quant_decomp(q_flat, group_size=128)
    return x_fp8, scale


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
q_pe_raw = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
q_nope3 = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
cos3 = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
sin3 = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
compiled3 = torch.compile(mla_q_path, dynamic=True)
with torch.no_grad():
    out3_fp8, out3_scale = compiled3(q_pe_raw, q_nope3, cos3, sin3)

ref3_fp8, ref3_scale = mla_q_path(q_pe_raw, q_nope3, cos3, sin3)
print(f"  Correctness: fp8 match={torch.equal(out3_fp8, ref3_fp8)}, "
      f"scale max_diff={( ref3_scale - out3_scale).abs().max().item():.8f}")


# ============================================================
# Test 4: graph14 pattern — cat + reshape + quant (576 = 512+64)
# ============================================================
print("\n" + "=" * 70)
print("Test 4: graph14 MLA decode — cat([nope_512, pe_64]) → flat → quant")
print("  This pattern uses forward_native already, but let's confirm fusion")
print("=" * 70)


def graph14_decomposed(nope, pe):
    # cat: [batch, 32, 576] = cat([nope(512), pe(64)], dim=-1)
    cat = torch.cat([nope, pe], dim=-1)
    # reshape to [batch, 18432]
    flat = cat.reshape(cat.shape[0], -1)
    # per_token_group_fp8_quant with group_size=128
    # 18432 / 128 = 144 groups per row
    x_fp8, scale = per_token_group_fp8_quant_decomp(flat, group_size=128)
    # reshape back
    x_fp8 = x_fp8.reshape(cat.shape[0], 32, 576)
    return x_fp8, scale


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
nope4 = torch.randn(4, 32, 512, device="cuda", dtype=torch.bfloat16)
pe4 = torch.randn(4, 32, 64, device="cuda", dtype=torch.bfloat16)
compiled4 = torch.compile(graph14_decomposed, dynamic=True)
with torch.no_grad():
    out4_fp8, out4_scale = compiled4(nope4, pe4)

ref4_fp8, ref4_scale = graph14_decomposed(nope4, pe4)
print(f"  Correctness: fp8 match={torch.equal(out4_fp8, ref4_fp8)}, "
      f"scale max_diff={( ref4_scale - out4_scale).abs().max().item():.8f}")


# ============================================================
# Test 5: Compare opaque vs decomposed (simulated)
# ============================================================
print("\n" + "=" * 70)
print("Test 5: What the ORIGINAL graph looks like (cat → output, no quant fuse)")
print("  cat goes to output since opaque quant blocks fusion")
print("=" * 70)


def cat_to_output_only(q_pe, q_nope):
    """What inductor sees when per_token_group_fp8_quant is opaque:
    cat is a standalone kernel writing to memory."""
    cat = torch.cat([q_pe, q_nope], dim=-1)
    return cat.reshape(-1, 128)


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
compiled5 = torch.compile(cat_to_output_only, dynamic=True)
with torch.no_grad():
    out5 = compiled5(q_pe, q_nope)


GraphLowering._update_scheduler = original_update
print("\n" + "=" * 70)
print("DONE — Summary")
print("=" * 70)
print("""
Key finding:
  - Test 2: cat + reshape + fp8_quant → should fuse into 1-2 kernels
  - Test 3: RoPE + cat + reshape + fp8_quant → should fuse end-to-end
  - Test 5: cat + reshape alone (opaque barrier) → separate kernel

The decomposition eliminates the fusion barrier and lets inductor
merge the cat's pointwise_cat logic directly into the quantization kernel.
""")
