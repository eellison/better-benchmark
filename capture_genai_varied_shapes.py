"""
Capture GenAI/LLM kernel patterns at varied shapes reflecting real workloads.

Shape configs:
1. Decode (single token): batch=32, seq=1, hidden=4096, heads=32, head_dim=128
2. Short prefill: batch=4, seq=128, hidden=4096
3. Long context: batch=1, seq=8192, hidden=4096
4. Small model (7B-like): batch=8, seq=512, hidden=4096
5. Large model (70B-like): batch=1, seq=2048, hidden=8192, heads=64, head_dim=128, kv_heads=8

Patterns: SwiGLU, RMSNorm, RoPE, Causal Attention, GQA
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, "/tmp/scratch_space/better_benchmark")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
from pathlib import Path

from capture_hook import install_capture_hook, uninstall_capture_hook
from merge_captures import merge_one_capture
from torch._inductor.utils import fresh_inductor_cache

REPRO_DIR = Path("/tmp/scratch_space/better_benchmark/repros")
DEVICE = "cuda"
DTYPE = torch.bfloat16

# Shape configurations
SHAPE_CONFIGS = {
    "decode": {
        "batch": 32, "seq": 1, "hidden": 4096,
        "heads": 32, "head_dim": 128, "kv_heads": 8,
    },
    "short_prefill": {
        "batch": 4, "seq": 128, "hidden": 4096,
        "heads": 32, "head_dim": 128, "kv_heads": 8,
    },
    "long_context": {
        "batch": 1, "seq": 8192, "hidden": 4096,
        "heads": 32, "head_dim": 128, "kv_heads": 8,
    },
    "small_model": {
        "batch": 8, "seq": 512, "hidden": 4096,
        "heads": 32, "head_dim": 128, "kv_heads": 8,
    },
    "large_model": {
        "batch": 1, "seq": 2048, "hidden": 8192,
        "heads": 64, "head_dim": 128, "kv_heads": 8,
    },
}


def capture_one(pattern_fn, label, make_inputs_fn):
    """Capture a single pattern+shape combination."""
    torch._dynamo.reset()
    tmpdir = tempfile.mkdtemp()
    install_capture_hook(tmpdir, label=label)
    try:
        with fresh_inductor_cache():
            compiled = torch.compile(pattern_fn, fullgraph=True)
            inputs = make_inputs_fn()
            with torch.no_grad():
                compiled(*inputs)
                torch.cuda.synchronize()
    except Exception as e:
        print(f"  ERROR capturing {label}: {e}")
        uninstall_capture_hook()
        return 0
    uninstall_capture_hook()
    n = merge_one_capture(Path(tmpdir), REPRO_DIR, label)
    # Clean up GPU memory
    del inputs
    torch.cuda.empty_cache()
    return n


# ============================================================
# Pattern: SwiGLU
# x1, x2 = x.chunk(2, -1); return x1 * F.silu(x2)
# ============================================================
def swiglu_pattern(x):
    x1, x2 = x.chunk(2, dim=-1)
    return x1 * F.silu(x2)


# ============================================================
# Pattern: RMSNorm
# x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
# ============================================================
class RMSNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=DTYPE, device=DEVICE))

    def forward(self, x):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-6)
        return (self.weight * x).to(x.dtype)


# ============================================================
# Pattern: RoPE
# x * cos + rotate_half(x) * sin
# rotate_half: cat(-x2, x1)
# ============================================================
def rope_pattern(x, cos, sin):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


# ============================================================
# Pattern: Causal Attention
# softmax(QK^T / sqrt(d) + causal_mask) @ V
# ============================================================
def causal_attention_pattern(Q, K, V, head_dim):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    seq_len = scores.shape[-1]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    scores = torch.softmax(scores, dim=-1)
    out = torch.matmul(scores, V)
    return out


# ============================================================
# Pattern: GQA (Grouped Query Attention)
# expand kv_heads to q_heads then attention
# ============================================================
def gqa_pattern(Q, K, V, n_rep, head_dim):
    # Expand KV heads
    batch, num_kv_heads, slen, hd = K.shape
    K_expanded = K[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, hd)
    K_expanded = K_expanded.reshape(batch, num_kv_heads * n_rep, slen, hd)
    V_expanded = V[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, hd)
    V_expanded = V_expanded.reshape(batch, num_kv_heads * n_rep, slen, hd)
    # Attention
    scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = torch.softmax(scores, dim=-1)
    out = torch.matmul(scores, V_expanded)
    return out


# ============================================================
# Run captures for all shape configs
# ============================================================
total_merged = 0

for config_name, cfg in SHAPE_CONFIGS.items():
    batch = cfg["batch"]
    seq = cfg["seq"]
    hidden = cfg["hidden"]
    heads = cfg["heads"]
    head_dim = cfg["head_dim"]
    kv_heads = cfg["kv_heads"]
    n_rep = heads // kv_heads

    print(f"\n{'='*60}")
    print(f"Shape config: {config_name} (batch={batch}, seq={seq}, hidden={hidden}, heads={heads})")
    print(f"{'='*60}")

    # --- SwiGLU ---
    print(f"\n  [SwiGLU] {config_name}")
    label = f"genai_swiglu_{config_name}"
    n = capture_one(
        swiglu_pattern,
        label,
        lambda b=batch, s=seq, h=hidden: [
            torch.randn(b, s, 2 * h, device=DEVICE, dtype=DTYPE)
        ],
    )
    total_merged += n
    print(f"    Merged {n} regions")

    # --- RMSNorm ---
    print(f"\n  [RMSNorm] {config_name}")
    label = f"genai_rmsnorm_{config_name}"
    model = RMSNorm(hidden).cuda()
    n = capture_one(
        model,
        label,
        lambda b=batch, s=seq, h=hidden: [
            torch.randn(b, s, h, device=DEVICE, dtype=DTYPE)
        ],
    )
    total_merged += n
    print(f"    Merged {n} regions")
    del model

    # --- RoPE ---
    print(f"\n  [RoPE] {config_name}")
    label = f"genai_rope_{config_name}"
    n = capture_one(
        rope_pattern,
        label,
        lambda b=batch, s=seq, hds=heads, hd=head_dim: [
            torch.randn(b, hds, s, hd, device=DEVICE, dtype=DTYPE),
            torch.randn(1, 1, s, hd, device=DEVICE, dtype=DTYPE),
            torch.randn(1, 1, s, hd, device=DEVICE, dtype=DTYPE),
        ],
    )
    total_merged += n
    print(f"    Merged {n} regions")

    # --- Causal Attention ---
    # For long sequences, causal attention is O(seq^2) in memory - limit seq
    attn_seq = min(seq, 2048)  # cap at 2048 to avoid OOM on seq=8192
    print(f"\n  [Causal Attention] {config_name} (attn_seq={attn_seq})")
    label = f"genai_causal_attn_{config_name}"

    # Use functools.partial equivalent via closure for head_dim
    _hd = head_dim
    def _causal_attn_fn(Q, K, V, _head_dim=_hd):
        return causal_attention_pattern(Q, K, V, _head_dim)

    n = capture_one(
        _causal_attn_fn,
        label,
        lambda b=batch, s=attn_seq, hds=heads, hd=head_dim: [
            torch.randn(b, hds, s, hd, device=DEVICE, dtype=DTYPE),
            torch.randn(b, hds, s, hd, device=DEVICE, dtype=DTYPE),
            torch.randn(b, hds, s, hd, device=DEVICE, dtype=DTYPE),
        ],
    )
    total_merged += n
    print(f"    Merged {n} regions")

    # --- GQA ---
    # Also cap GQA seq length
    gqa_seq = min(seq, 2048)
    print(f"\n  [GQA] {config_name} (gqa_seq={gqa_seq})")
    label = f"genai_gqa_{config_name}"

    _n_rep = n_rep
    _hd2 = head_dim
    def _gqa_fn(Q, K, V, _nr=_n_rep, _hd=_hd2):
        return gqa_pattern(Q, K, V, _nr, _hd)

    n = capture_one(
        _gqa_fn,
        label,
        lambda b=batch, s=gqa_seq, hds=heads, kvh=kv_heads, hd=head_dim: [
            torch.randn(b, hds, s, hd, device=DEVICE, dtype=DTYPE),
            torch.randn(b, kvh, s, hd, device=DEVICE, dtype=DTYPE),
            torch.randn(b, kvh, s, hd, device=DEVICE, dtype=DTYPE),
        ],
    )
    total_merged += n
    print(f"    Merged {n} regions")

    torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"TOTAL MERGED: {total_merged} regions across all shape configs")
print(f"{'='*60}")
