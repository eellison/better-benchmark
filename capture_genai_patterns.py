"""
Capture kernels from common GenAI/LLM patterns via torch.compile + capture_hook.

Patterns captured:
1. Flash attention (inductor-compiled matmul + softmax variant)
2. SwiGLU activation (LLaMA/Mistral FFN)
3. Grouped Query Attention (GQA) with repeat_kv
4. RMSNorm + residual (full pattern)

Shapes: LLM inference typical — batch=4, seq=2048, hidden=4096, heads=32, head_dim=128, kv_heads=8
"""
import math
import os
import sys

sys.path.insert(0, "/tmp/scratch_space/better_benchmark")

import torch
import torch.nn.functional as F

from capture_hook import install_capture_hook, uninstall_capture_hook

CAPTURE_DIR = "/tmp/captures/genai_patterns"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Install capture hook
install_capture_hook(output_dir=CAPTURE_DIR, label="genai_patterns")

# ============================================================
# Shapes (LLM inference typical)
# ============================================================
BATCH = 4
SEQ = 2048
HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
KV_HEADS = 8
NUM_GROUPS = HEADS // KV_HEADS  # 4

DEVICE = "cuda"
DTYPE = torch.bfloat16


# ============================================================
# Pattern 1: Flash attention pattern (inductor-compiled)
# Q, K, V: [batch, heads, seq, head_dim]
# scores = Q @ K^T / sqrt(head_dim), softmax, then @ V
# ============================================================
def flash_attention_pattern(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    scores = torch.softmax(scores, dim=-1)
    out = torch.matmul(scores, V)
    return out


print("\n=== Pattern 1: Flash Attention (compiled) ===")
Q = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)
K = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)
V = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)

compiled_attn = torch.compile(flash_attention_pattern, fullgraph=True)
with torch.no_grad():
    out = compiled_attn(Q, K, V)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del Q, K, V, out
torch.cuda.empty_cache()

# ============================================================
# Pattern 2: SwiGLU activation (LLaMA/Mistral FFN)
# x1, x2 = x.chunk(2, dim=-1)
# return x1 * silu(x2)
# ============================================================
def swiglu_pattern(x):
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * F.silu(x2)


print("\n=== Pattern 2: SwiGLU Activation ===")
# In LLaMA, the FFN intermediate is typically 4 * hidden * 2/3 * 2 (for gate+up)
# Simplified: use 2*hidden for the gate+up combined
x_ffn = torch.randn(BATCH, SEQ, 2 * HIDDEN, device=DEVICE, dtype=DTYPE)

compiled_swiglu = torch.compile(swiglu_pattern, fullgraph=True)
with torch.no_grad():
    out = compiled_swiglu(x_ffn)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del x_ffn, out
torch.cuda.empty_cache()


# ============================================================
# Pattern 2b: GeGLU activation variant (used in some models)
# x1, x2 = x.chunk(2, dim=-1)
# return x1 * gelu(x2)
# ============================================================
def geglu_pattern(x):
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * F.gelu(x2)


print("\n=== Pattern 2b: GeGLU Activation ===")
x_ffn = torch.randn(BATCH, SEQ, 2 * HIDDEN, device=DEVICE, dtype=DTYPE)

compiled_geglu = torch.compile(geglu_pattern, fullgraph=True)
with torch.no_grad():
    out = compiled_geglu(x_ffn)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del x_ffn, out
torch.cuda.empty_cache()


# ============================================================
# Pattern 3: Grouped Query Attention (GQA) with repeat_kv
# KV: [batch, kv_heads, seq, head_dim] -> [batch, q_heads, seq, head_dim]
# Then perform attention
# ============================================================
def repeat_kv(hidden_states, n_rep):
    """Repeat KV heads to match Q heads (from transformers library)."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def gqa_attention_pattern(Q, K, V):
    # Expand KV to match Q heads
    K_expanded = repeat_kv(K, NUM_GROUPS)
    V_expanded = repeat_kv(V, NUM_GROUPS)
    # Attention
    scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    scores = torch.softmax(scores, dim=-1)
    out = torch.matmul(scores, V_expanded)
    return out


print("\n=== Pattern 3: Grouped Query Attention (GQA) ===")
Q = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)
K_gqa = torch.randn(BATCH, KV_HEADS, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)
V_gqa = torch.randn(BATCH, KV_HEADS, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)

compiled_gqa = torch.compile(gqa_attention_pattern, fullgraph=True)
with torch.no_grad():
    out = compiled_gqa(Q, K_gqa, V_gqa)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del Q, K_gqa, V_gqa, out
torch.cuda.empty_cache()


# ============================================================
# Pattern 4: RMSNorm + residual (full pattern)
# residual = hidden + input
# normed = rmsnorm(residual)
# ============================================================
def rmsnorm_residual_pattern(hidden_states, residual, weight, eps=1e-6):
    """RMSNorm with residual connection (common in LLaMA decoder layers)."""
    # Add residual
    hidden_states = hidden_states + residual
    # RMSNorm
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(hidden_states.dtype)


print("\n=== Pattern 4: RMSNorm + Residual ===")
hidden = torch.randn(BATCH, SEQ, HIDDEN, device=DEVICE, dtype=DTYPE)
residual = torch.randn(BATCH, SEQ, HIDDEN, device=DEVICE, dtype=DTYPE)
weight = torch.randn(HIDDEN, device=DEVICE, dtype=DTYPE)

compiled_rmsnorm = torch.compile(rmsnorm_residual_pattern, fullgraph=True)
with torch.no_grad():
    out = compiled_rmsnorm(hidden, residual, weight)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del hidden, residual, weight, out
torch.cuda.empty_cache()


# ============================================================
# Pattern 4b: RMSNorm without residual (standalone)
# ============================================================
def rmsnorm_pattern(hidden_states, weight, eps=1e-6):
    """Standalone RMSNorm."""
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (weight * hidden_states).to(hidden_states.dtype)


print("\n=== Pattern 4b: Standalone RMSNorm ===")
hidden = torch.randn(BATCH, SEQ, HIDDEN, device=DEVICE, dtype=DTYPE)
weight = torch.randn(HIDDEN, device=DEVICE, dtype=DTYPE)

compiled_rmsnorm2 = torch.compile(rmsnorm_pattern, fullgraph=True)
with torch.no_grad():
    out = compiled_rmsnorm2(hidden, weight)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del hidden, weight, out
torch.cuda.empty_cache()


# ============================================================
# Pattern 5: Rotary Position Embedding (RoPE) — common in all modern LLMs
# ============================================================
def rotary_embedding_pattern(x, cos, sin):
    """Apply rotary position embedding."""
    # x: [batch, heads, seq, head_dim]
    # cos, sin: [1, 1, seq, head_dim]
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # Rotate
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


print("\n=== Pattern 5: Rotary Position Embedding (RoPE) ===")
x_rope = torch.randn(BATCH, HEADS, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)
cos = torch.randn(1, 1, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)
sin = torch.randn(1, 1, SEQ, HEAD_DIM, device=DEVICE, dtype=DTYPE)

compiled_rope = torch.compile(rotary_embedding_pattern, fullgraph=True)
with torch.no_grad():
    out = compiled_rope(x_rope, cos, sin)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del x_rope, cos, sin, out
torch.cuda.empty_cache()


# ============================================================
# Pattern 6: Causal attention mask + softmax (fused pattern)
# ============================================================
def causal_attention_with_mask(Q, K, V):
    """Attention with causal mask (upper triangle = -inf)."""
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
    # Apply causal mask
    seq_len = scores.shape[-1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    scores = torch.softmax(scores, dim=-1)
    out = torch.matmul(scores, V)
    return out


print("\n=== Pattern 6: Causal Attention with Mask ===")
# Use smaller seq for memory — causal mask is O(seq^2)
SEQ_SHORT = 512
Q = torch.randn(BATCH, HEADS, SEQ_SHORT, HEAD_DIM, device=DEVICE, dtype=DTYPE)
K = torch.randn(BATCH, HEADS, SEQ_SHORT, HEAD_DIM, device=DEVICE, dtype=DTYPE)
V = torch.randn(BATCH, HEADS, SEQ_SHORT, HEAD_DIM, device=DEVICE, dtype=DTYPE)

compiled_causal = torch.compile(causal_attention_with_mask, fullgraph=True)
with torch.no_grad():
    out = compiled_causal(Q, K, V)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del Q, K, V, out
torch.cuda.empty_cache()


# ============================================================
# Pattern 7: Linear + SwiGLU + Linear (full MLP block)
# ============================================================
def llama_mlp_pattern(x, gate_weight, up_weight, down_weight):
    """LLaMA MLP: gate_proj + up_proj with SwiGLU, then down_proj."""
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    hidden = F.silu(gate) * up
    out = F.linear(hidden, down_weight)
    return out


print("\n=== Pattern 7: LLaMA MLP (Linear + SwiGLU + Linear) ===")
# LLaMA intermediate_size is typically 11008 for 4096 hidden
INTERMEDIATE = 11008
x_mlp = torch.randn(BATCH, SEQ, HIDDEN, device=DEVICE, dtype=DTYPE)
gate_w = torch.randn(INTERMEDIATE, HIDDEN, device=DEVICE, dtype=DTYPE)
up_w = torch.randn(INTERMEDIATE, HIDDEN, device=DEVICE, dtype=DTYPE)
down_w = torch.randn(HIDDEN, INTERMEDIATE, device=DEVICE, dtype=DTYPE)

compiled_mlp = torch.compile(llama_mlp_pattern, fullgraph=True)
with torch.no_grad():
    out = compiled_mlp(x_mlp, gate_w, up_w, down_w)
    torch.cuda.synchronize()
print(f"  Output shape: {out.shape}")

del x_mlp, gate_w, up_w, down_w, out
torch.cuda.empty_cache()


# Finalize capture
uninstall_capture_hook()
print(f"\n=== Capture complete. Files in {CAPTURE_DIR} ===")
