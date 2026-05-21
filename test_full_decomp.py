"""
Test: decompose ALL the custom ops in the graph1 attention path and check fusion.

The full chain in graph1 is:
  flashinfer_trtllm_fused_allreduce_norm (pattern=1)
    → min_latency_fused_qkv_a_proj (matmul)
      → split → RMSNorm(q_c) → mm (big matmul)
        → split(q_pe, q_nope)
          → RoPE(q_pe) → cat([q_pe_roped, q_nope])
            → view → per_token_group_fp8_quant

We decompose:
  - flashinfer_trtllm_fused_allreduce_norm → all_reduce (opaque) + add + RMSNorm
  - min_latency_fused_qkv_a_proj → mm
  - per_token_group_fp8_quant → absmax + div + clamp + cast

Question: does the RMSNorm between the two matmuls fuse with anything?
And does the full post-second-matmul chain fuse end-to-end?
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
import torch._inductor.ir as ir
from torch._inductor.graph import GraphLowering
import sympy

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


def per_token_group_fp8_quant_decomp(x, group_size=128, eps=1e-10):
    """Decomposed per_token_group_fp8_quant."""
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


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm decomposed."""
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    return (x_normed.to(x.dtype) * weight)


# ============================================================
# Test 1: RMSNorm → matmul chain (the q_c path)
# Does RMSNorm fuse standalone?
# ============================================================
print("=" * 70)
print("Test 1: RMSNorm alone — does it fuse into 1 kernel?")
print("=" * 70)


def rmsnorm_only(x, gamma):
    return rms_norm(x, gamma)


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
x1 = torch.randn(128, 1536, device="cuda", dtype=torch.bfloat16)
gamma1 = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
compiled1 = torch.compile(rmsnorm_only, dynamic=True)
with torch.no_grad():
    compiled1(x1, gamma1)


# ============================================================
# Test 2: allreduce_norm decomposed (pattern_code=1)
# all_reduce + residual_add + RMSNorm
# ============================================================
print("\n" + "=" * 70)
print("Test 2: all_reduce + add + RMSNorm (flashinfer_trtllm decomposed, pattern=1)")
print("  In reality, all_reduce is opaque. Simulating with identity.")
print("=" * 70)


def allreduce_norm_decomp(x, residual, gamma):
    # all_reduce would be here (opaque, returns same shape)
    # For testing, just use x directly (simulates single-GPU)
    reduced = x  # torch.ops.vllm.all_reduce(x) in real code
    # residual add
    residual_out = reduced + residual
    # RMSNorm
    norm_out = rms_norm(residual_out, gamma)
    return norm_out, residual_out


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
x2 = torch.randn(128, 7168, device="cuda", dtype=torch.bfloat16)
res2 = torch.randn(128, 7168, device="cuda", dtype=torch.bfloat16)
gamma2 = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
compiled2 = torch.compile(allreduce_norm_decomp, dynamic=True)
with torch.no_grad():
    compiled2(x2, res2, gamma2)


# ============================================================
# Test 3: allreduce_norm → matmul → RMSNorm → matmul
# The first half of the MLA path
# ============================================================
print("\n" + "=" * 70)
print("Test 3: norm → mm(qkv_a_proj) → split → RMSNorm(q_c) → mm(q_proj)")
print("  Two matmuls with RMSNorm between them")
print("=" * 70)


def mla_first_half(hidden, residual, gamma_attn, w_qkv_a, gamma_q, w_q_proj):
    # Decomposed flashinfer_trtllm_fused_allreduce_norm
    residual_out = hidden + residual
    norm_out = rms_norm(residual_out, gamma_attn)

    # Decomposed min_latency_fused_qkv_a_proj = mm
    qkv_lora = norm_out @ w_qkv_a.t()  # [tokens, 2112]

    # split into q_c and kv_lora
    q_c = qkv_lora[:, :1536]
    kv_lora = qkv_lora[:, 1536:]

    # RMSNorm on q_c
    q_c_normed = rms_norm(q_c, gamma_q)

    # Second matmul: q_proj
    q = q_c_normed @ w_q_proj.t()  # [tokens, 8192]

    return q, kv_lora, residual_out


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
hidden3 = torch.randn(128, 7168, device="cuda", dtype=torch.bfloat16)
res3 = torch.randn(128, 7168, device="cuda", dtype=torch.bfloat16)
gamma_attn3 = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
w_qkv_a3 = torch.randn(2112, 7168, device="cuda", dtype=torch.bfloat16)
gamma_q3 = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
w_q_proj3 = torch.randn(8192, 1536, device="cuda", dtype=torch.bfloat16)
compiled3 = torch.compile(mla_first_half, dynamic=True)
with torch.no_grad():
    compiled3(hidden3, res3, gamma_attn3, w_qkv_a3, gamma_q3, w_q_proj3)


# ============================================================
# Test 4: The full q-path after second matmul
# mm → split(q_pe, q_nope) → RoPE → cat → view → fp8_quant
# ============================================================
print("\n" + "=" * 70)
print("Test 4: mm_output → split → RoPE → cat → view → fp8_quant")
print("  Everything after the second matmul")
print("=" * 70)


def mla_q_after_mm(q_proj_out, cos, sin):
    # q_proj_out: [tokens, 8192] → view as [tokens, 64, 128]
    q = q_proj_out.reshape(-1, 64, 128)

    # split into q_pe(64) and q_nope(64)
    q_pe = q[..., :64]
    q_nope = q[..., 64:]

    # RoPE on q_pe (rotate_neox)
    x1 = q_pe[..., :32]
    x2 = q_pe[..., 32:]
    rotated = torch.cat([-x2, x1], dim=-1)
    q_pe_roped = q_pe * cos + rotated * sin

    # cat: [tokens, 64, 128] = cat([q_pe_roped, q_nope])
    q_cat = torch.cat([q_pe_roped, q_nope], dim=-1)

    # view to [64*tokens, 128]
    q_flat = q_cat.reshape(-1, 128)

    # Decomposed per_token_group_fp8_quant
    q_fp8, q_scale = per_token_group_fp8_quant_decomp(q_flat, group_size=128)

    return q_fp8, q_scale


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
q_proj_out4 = torch.randn(4, 8192, device="cuda", dtype=torch.bfloat16)
cos4 = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
sin4 = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
compiled4 = torch.compile(mla_q_after_mm, dynamic=True)
with torch.no_grad():
    compiled4(q_proj_out4, cos4, sin4)


# ============================================================
# Test 5: The FULL path end-to-end (norm → mm → norm → mm → rope → cat → quant)
# ============================================================
print("\n" + "=" * 70)
print("Test 5: FULL MLA q-path end-to-end")
print("  norm → mm → split → norm → mm → split → RoPE → cat → quant")
print("=" * 70)


def full_mla_q_path(hidden, residual, gamma_attn, w_qkv_a, gamma_q, w_q_proj, cos, sin):
    # 1. Decomposed allreduce_norm (simulating single GPU)
    residual_out = hidden + residual
    norm_out = rms_norm(residual_out, gamma_attn)

    # 2. First matmul (qkv_a_proj)
    qkv_lora = norm_out @ w_qkv_a.t()

    # 3. Split
    q_c = qkv_lora[:, :1536]

    # 4. RMSNorm on q_c
    q_c_normed = rms_norm(q_c, gamma_q)

    # 5. Second matmul (q_proj)
    q_proj = q_c_normed @ w_q_proj.t()  # [tokens, 8192]

    # 6. Reshape + split into q_pe/q_nope
    q = q_proj.reshape(-1, 64, 128)
    q_pe = q[..., :64]
    q_nope = q[..., 64:]

    # 7. RoPE on q_pe
    x1 = q_pe[..., :32]
    x2 = q_pe[..., 32:]
    rotated = torch.cat([-x2, x1], dim=-1)
    q_pe_roped = q_pe * cos + rotated * sin

    # 8. cat + view + fp8_quant
    q_cat = torch.cat([q_pe_roped, q_nope], dim=-1)
    q_flat = q_cat.reshape(-1, 128)
    q_fp8, q_scale = per_token_group_fp8_quant_decomp(q_flat, group_size=128)

    return q_fp8, q_scale, residual_out


torch._dynamo.reset()
GraphLowering._update_scheduler = fusion_reporter
hidden5 = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
res5 = torch.randn(4, 7168, device="cuda", dtype=torch.bfloat16)
gamma_attn5 = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
w_qkv_a5 = torch.randn(2112, 7168, device="cuda", dtype=torch.bfloat16)
gamma_q5 = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
w_q_proj5 = torch.randn(8192, 1536, device="cuda", dtype=torch.bfloat16)
cos5 = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
sin5 = torch.randn(4, 64, 64, device="cuda", dtype=torch.bfloat16)
compiled5 = torch.compile(full_mla_q_path, dynamic=True)
with torch.no_grad():
    compiled5(hidden5, res5, gamma_attn5, w_qkv_a5, gamma_q5, w_q_proj5, cos5, sin5)


GraphLowering._update_scheduler = original_update
print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print("""
Expected fusion pattern:
  - RMSNorm: fuses into 1 kernel (reduction + pointwise)
  - add+RMSNorm: fuses into 1 kernel
  - After matmul: split+RoPE+cat+quant → 1 kernel (all pointwise)
  - Matmuls: standalone (extern_kernels)

The matmuls are natural fusion barriers (cuBLAS calls).
Everything between matmuls should collapse into single kernels.
""")
