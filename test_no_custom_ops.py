"""
Test: What if vllm had NO custom ops at all?
Everything decomposed to standard aten. What fusions does inductor find?

This simulates the "pure torch.compile" world where:
  - flashinfer_trtllm_fused_allreduce_norm → all_reduce + add + RMSNorm
  - min_latency_fused_qkv_a_proj → mm
  - per_token_group_fp8_quant → absmax + div + clamp + cast
  - silu_and_mul_nvfp4_quant → silu_and_mul + scaled_fp4_quant
  - scaled_fp4_quant → absmax + scale + cast (pretend fp4 = fp8 for fusion analysis)
  - flashinfer_mm_fp4 → mm (pretend it's a normal matmul)
  - moe_forward_shared → keep opaque (too complex)
  - all_reduce → keep opaque (communication)

Question: which fusions would we MISS if vllm didn't hand-fuse these?
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering

cfg.force_disable_caches = True

original_update = GraphLowering._update_scheduler


def fusion_reporter(self):
    from torch._inductor.scheduler import Scheduler
    with cfg.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
    nodes = self.scheduler.nodes
    fused = [nd for nd in nodes if hasattr(nd, 'snodes') and len(nd.snodes) > 1]
    single = len(nodes) - len(fused)
    print(f"    Scheduler: {len(nodes)} nodes ({len(fused)} fused, {single} single)")
    for nd in nodes:
        if hasattr(nd, 'snodes') and len(nd.snodes) > 1:
            names = [s.node.get_name() for s in nd.snodes if hasattr(s, 'node')]
            print(f"      Fused({len(nd.snodes)}): {names[:8]}{'...' if len(names) > 8 else ''}")
        else:
            name = nd.node.get_name() if hasattr(nd, 'node') else '?'
            ntype = type(nd).__name__
            print(f"      {ntype}: {name}")


def rms_norm(x, weight, eps=1e-6):
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (x_normed.to(x.dtype) * weight)


def per_token_group_fp8_quant(x, group_size=128, eps=1e-10):
    orig_shape = x.shape
    x_grouped = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    x_f32 = x_grouped.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 448.0).clamp(min=eps)
    x_scaled = x_f32 / scale
    x_clamped = x_scaled.clamp(-448.0, 448.0)
    x_fp8 = x_clamped.to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(orig_shape)
    scale = scale.reshape(*orig_shape[:-1], orig_shape[-1] // group_size)
    return x_fp8, scale


def per_tensor_fp8_quant(x, scale):
    """Simulates scaled_fp4_quant as fp8 for fusion analysis purposes."""
    x_f32 = x.float()
    x_scaled = x_f32 * scale.reciprocal()
    x_clamped = x_scaled.clamp(-448.0, 448.0)
    return x_clamped.to(torch.float8_e4m3fn)


def silu_and_mul(x):
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.nn.functional.silu(x1) * x2


# ============================================================
# GRAPH 1 pattern: Embedding layer + first attention
# allreduce_norm → qkv_a_proj → split → RMSNorm → q_proj
#   → split → RoPE → cat → quant
# Also: kw_proj → LayerNorm → split → RoPE(k_pe) → cat(k)
# ============================================================
print("=" * 70)
print("GRAPH 1: Full MLA attention (first layer)")
print("  allreduce+norm → mm → norm → mm → RoPE → cat → fp8_quant")
print("=" * 70)


def graph1_fully_decomposed(
    hidden,         # [tokens, 7168] - from embedding
    residual,       # [tokens, 7168] - zero for first layer
    gamma_attn,     # [7168]
    w_qkv_a,       # [2112, 7168]
    gamma_q,        # [1536]
    w_q_proj,       # [8192, 1536]
    w_kw,           # [192, 7168]
    gamma_kw,       # [128] (LayerNorm weight)
    beta_kw,        # [128] (LayerNorm bias)
    cos_cache,      # [max_pos, 32]
    sin_cache,      # [max_pos, 32]
    positions,      # [tokens]
):
    # === Decomposed flashinfer_trtllm_fused_allreduce_norm (pattern=1) ===
    # all_reduce(hidden) — opaque on multi-GPU, identity here
    residual_out = hidden + residual
    norm_out = rms_norm(residual_out, gamma_attn)

    # === Decomposed min_latency_fused_qkv_a_proj ===
    qkv_lora = norm_out @ w_qkv_a.t()  # [tokens, 2112]

    # Split: q_c(1536) + kv_lora(576)
    q_c = qkv_lora[:, :1536]
    kv_lora = qkv_lora[:, 1536:]
    # Split kv_lora: kv_c(512) + k_pe(64)
    kv_c = kv_lora[:, :512]
    k_pe_raw = kv_lora[:, 512:]

    # === RMSNorm on q_c ===
    q_c_normed = rms_norm(q_c, gamma_q)

    # === Second matmul: q_proj ===
    q_proj = q_c_normed @ w_q_proj.t()  # [tokens, 8192]

    # === q path: split → RoPE → cat → quant ===
    q = q_proj.reshape(-1, 64, 128)
    q_pe = q[..., :64]
    q_nope = q[..., 64:]

    # RoPE on q_pe (rotate_neox): q_pe is [tokens, 64, 64]
    x1 = q_pe[..., :32]
    x2 = q_pe[..., 32:]
    rotated_q = torch.cat([-x2, x1], dim=-1)
    # cos/sin: [tokens, 64] → unsqueeze to [tokens, 1, 64]
    cos = cos_cache[positions]  # [tokens, 32]
    sin = sin_cache[positions]
    cos_full = cos.repeat(1, 2).unsqueeze(1)  # [tokens, 1, 64]
    sin_full = sin.repeat(1, 2).unsqueeze(1)
    q_pe_roped = q_pe * cos_full + rotated_q * sin_full

    # cat + view + fp8_quant
    q_cat = torch.cat([q_pe_roped, q_nope], dim=-1)  # [tokens, 64, 128]
    q_flat = q_cat.reshape(-1, 128)
    q_fp8, q_scale = per_token_group_fp8_quant(q_flat, group_size=128)

    # === k path: kw_proj → LayerNorm → split → RoPE → cat ===
    kw = norm_out @ w_kw.t()  # [tokens, 192]
    k_head = kw[:, :128]
    # LayerNorm on k_head
    k_f32 = k_head.float()
    mean = k_f32.mean(dim=-1, keepdim=True)
    var = k_f32.var(dim=-1, keepdim=True, correction=0)
    k_normed = ((k_f32 - mean) / torch.sqrt(var + 1e-6)).to(k_head.dtype)
    k_normed = k_normed * gamma_kw + beta_kw
    # split k
    k_pe_from_kw = k_normed[:, :64]
    k_nope = k_normed[:, 64:]

    # RoPE on k_pe_raw (from kv_lora split)
    k_pe_unsq = k_pe_raw.unsqueeze(1)  # [tokens, 1, 64]
    k1 = k_pe_unsq[..., :32]
    k2 = k_pe_unsq[..., 32:]
    rotated_k = torch.cat([-k2, k1], dim=-1)
    cos_k = cos_full  # [tokens, 1, 64]
    sin_k = sin_full
    k_pe_roped = k_pe_unsq * cos_k + rotated_k * sin_k
    k_pe_roped = k_pe_roped.squeeze(1)  # [tokens, 64]

    # Final k = cat([k_pe_roped, k_nope])
    k = torch.cat([k_pe_roped, k_nope], dim=-1)

    return q_fp8, q_scale, k, kv_c, residual_out


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
tokens = 4
hidden = torch.randn(tokens, 7168, device="cuda", dtype=torch.bfloat16)
residual = torch.zeros(tokens, 7168, device="cuda", dtype=torch.bfloat16)
gamma_attn = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
w_qkv_a = torch.randn(2112, 7168, device="cuda", dtype=torch.bfloat16)
gamma_q = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
w_q_proj = torch.randn(8192, 1536, device="cuda", dtype=torch.bfloat16)
w_kw = torch.randn(192, 7168, device="cuda", dtype=torch.bfloat16)
gamma_kw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
beta_kw = torch.randn(128, device="cuda", dtype=torch.bfloat16)
cos_cache = torch.randn(8192, 32, device="cuda", dtype=torch.bfloat16)
sin_cache = torch.randn(8192, 32, device="cuda", dtype=torch.bfloat16)
positions = torch.randint(0, 8192, (tokens,), device="cuda")
compiled_g1 = torch.compile(graph1_fully_decomposed, dynamic=True)
with torch.no_grad():
    compiled_g1(hidden, residual, gamma_attn, w_qkv_a, gamma_q, w_q_proj,
                w_kw, gamma_kw, beta_kw, cos_cache, sin_cache, positions)


# ============================================================
# GRAPH 2 pattern: FFN layer (allreduce+norm+fp4quant → mm → silu_mul → mm)
# ============================================================
print("\n" + "=" * 70)
print("GRAPH 2/3: FFN path (MLP layer)")
print("  allreduce+norm+quant → mm → silu_and_mul+quant → mm")
print("  (simulating fp4 as fp8 for fusion analysis)")
print("=" * 70)


def graph2_ffn_decomposed(
    hidden,       # [tokens, 7168]
    residual,     # [tokens, 7168]
    gamma_ffn,    # [7168]
    scale_ffn,    # scalar
    w_gate_up,    # [9216, 7168]  (gate + up fused)
    scale_down,   # scalar
    w_down,       # [7168, 4608]
):
    # === allreduce + add + RMSNorm + quant (pattern_code=3) ===
    residual_out = hidden + residual
    norm_out = rms_norm(residual_out, gamma_ffn)
    # "FP4 quant" → simulating as fp8 quant for fusion analysis
    norm_quant = per_tensor_fp8_quant(norm_out, scale_ffn)

    # === matmul (gate_up) ===
    # In real code this is flashinfer_mm_fp4, here simulated
    gate_up_out = norm_out @ w_gate_up.t()  # [tokens, 9216]

    # === silu_and_mul (decomposed from silu_and_mul_nvfp4_quant) ===
    mlp_out = silu_and_mul(gate_up_out)  # [tokens, 4608]
    # + quant
    mlp_quant = per_tensor_fp8_quant(mlp_out, scale_down)

    # === matmul (down_proj) ===
    down_out = mlp_out @ w_down.t()  # [tokens, 7168]

    return down_out, residual_out


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
hidden2 = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
res2 = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
gamma_ffn = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
scale_ffn = torch.tensor(0.1, device="cuda", dtype=torch.float32)
w_gate_up = torch.randn(9216, 7168, device="cuda", dtype=torch.bfloat16)
scale_down = torch.tensor(0.1, device="cuda", dtype=torch.float32)
w_down = torch.randn(7168, 4608, device="cuda", dtype=torch.bfloat16)
compiled_g2 = torch.compile(graph2_ffn_decomposed, dynamic=True)
with torch.no_grad():
    compiled_g2(hidden2, res2, gamma_ffn, scale_ffn, w_gate_up, scale_down, w_down)


# ============================================================
# GRAPH 5/10 pattern: MoE layer
# allreduce+norm → MoE(opaque) → scale + add → allreduce+norm
# ============================================================
print("\n" + "=" * 70)
print("GRAPH 5/10: MoE layer")
print("  norm → quant → mm → silu_mul+quant → mm → scale+add → norm")
print("  (MoE routing is opaque, simulating the pointwise parts)")
print("=" * 70)


def graph5_moe_decomposed(
    hidden,         # [tokens, 7168]
    residual,       # [tokens, 7168]
    gamma_moe,      # [7168]
    moe_output,     # [tokens, 7168] (pretend MoE already ran)
    shared_output,  # [tokens, 7168]
    gamma_next,     # [7168]
):
    # === allreduce + add + RMSNorm (first, feeds into MoE) ===
    residual_out = hidden + residual
    norm_out = rms_norm(residual_out, gamma_moe)

    # === MoE is opaque, but the post-MoE math is: ===
    # final = moe_output * 2.5 + shared_output
    final = moe_output * 2.5 + shared_output

    # === Second allreduce + add + RMSNorm (feeds into next layer) ===
    residual_out2 = final + residual_out
    norm_out2 = rms_norm(residual_out2, gamma_next)

    return norm_out, norm_out2, residual_out2


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
hidden5 = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
res5 = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
gamma_moe = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
moe_out = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
shared_out = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
gamma_next = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
compiled_g5 = torch.compile(graph5_moe_decomposed, dynamic=True)
with torch.no_grad():
    compiled_g5(hidden5, res5, gamma_moe, moe_out, shared_out, gamma_next)


# ============================================================
# GRAPH 13/14: MLA decode (already native — cat + fp8 quant)
# ============================================================
print("\n" + "=" * 70)
print("GRAPH 13/14: MLA decode q (already uses forward_native)")
print("  cat([nope_512, pe_64]) → reshape → fp8_quant (per-tensor)")
print("=" * 70)


def graph14_native(nope, pe, scale):
    cat = torch.cat([nope, pe], dim=-1)
    flat = cat.reshape(cat.shape[0], -1)
    x = flat.float() * scale.reciprocal()
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
nope14 = torch.randn(4, 32, 512, device="cuda", dtype=torch.bfloat16)
pe14 = torch.randn(4, 32, 64, device="cuda", dtype=torch.bfloat16)
scale14 = torch.tensor(0.1, device="cuda", dtype=torch.float32)
compiled_g14 = torch.compile(graph14_native, dynamic=True)
with torch.no_grad():
    compiled_g14(nope14, pe14, scale14)


GraphLowering._update_scheduler = original_update
print("\n" + "=" * 70)
print("SUMMARY: What fusions does inductor find with NO custom ops?")
print("=" * 70)
print("""
The natural fusion boundaries are MATMULS (cuBLAS extern kernels).
Between matmuls, everything pointwise collapses.

With all ops decomposed, the kernel count per layer should be:

  ATTENTION (graph1):
    Kernel 1: add + RMSNorm (fused)
    Kernel 2: mm (qkv_a_proj) — cuBLAS
    Kernel 3: RMSNorm(q_c) (fused)
    Kernel 4: mm (q_proj) — cuBLAS
    Kernel 5: RoPE + cat + fp8_quant (ALL FUSED — the big win)
    Kernel 6: mm (kw_proj) — cuBLAS
    Kernel 7: LayerNorm + split + RoPE + cat (fused)

  FFN (graph2):
    Kernel 1: add + RMSNorm + quant (fused)
    Kernel 2: mm (gate_up) — cuBLAS
    Kernel 3: silu_and_mul + quant (fused)
    Kernel 4: mm (down_proj) — cuBLAS

  MoE (graph5):
    Kernel 1: add + RMSNorm (fused)
    [MoE opaque]
    Kernel 2: scale + add (fused)
    Kernel 3: add + RMSNorm (fused)

What the CUSTOM OPS currently provide:
  - flashinfer_trtllm_fused_allreduce_norm: already optimal (same as Kernel 1)
  - silu_and_mul_nvfp4_quant: already optimal (same as Kernel 3 in FFN)
  - per_token_group_fp8_quant: BLOCKS fusion (Kernel 5 can't form)
    → This is the ONLY one where decomposition unlocks new fusion

MISSED FUSION (with custom ops):
  - cat + reshape + fp8_quant can't fuse → extra kernel + memory round-trip
  - This happens on EVERY attention layer (graphs 1-9)
""")
