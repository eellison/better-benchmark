"""
Capture kernel regions from tritonbench operator patterns using torch.compile.

For each operator pattern in tritonbench, we:
1. Define the same computation in PyTorch
2. Compile it with torch.compile
3. Capture the kernel regions via capture_hook

This gives us the inductor-generated kernels for these standard patterns,
which we can compare against the hand-written Triton reference implementations.
"""
import sys
import os
import torch
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, '/tmp/scratch_space/better_benchmark')
from capture_hook import install_capture_hook, uninstall_capture_hook
from pathlib import Path

CAPTURE_BASE = Path("/tmp/captures_tritonbench")
CAPTURE_BASE.mkdir(parents=True, exist_ok=True)

torch._dynamo.config.automatic_dynamic_shapes = False


def capture_operator(name, fn, inputs, shapes_desc=""):
    """Capture a single operator pattern."""
    capture_dir = CAPTURE_BASE / name
    capture_dir.mkdir(parents=True, exist_ok=True)

    install_capture_hook(str(capture_dir), label=f"tritonbench_{name}")

    try:
        compiled_fn = torch.compile(fn, mode="max-autotune-no-cudagraphs")
        with torch.no_grad():
            out = compiled_fn(*inputs)
            torch.cuda.synchronize()
        print(f"  [OK] {name} ({shapes_desc})")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
    finally:
        uninstall_capture_hook()


# ============================================================================
# 1. SOFTMAX - various shapes
# ============================================================================
print("\n=== SOFTMAX ===")

def softmax_fn(x):
    return F.softmax(x, dim=-1)

# Typical attention shapes
for M, N in [(4096, 4096), (4096, 8192), (8192, 1024), (32768, 256), (2048, 50257)]:
    x = torch.randn(M, N, dtype=torch.float16, device='cuda')
    capture_operator(f"softmax_{M}x{N}", softmax_fn, (x,), f"{M}x{N}")

# ============================================================================
# 2. FUSED SOFTMAX (permute + bias + softmax) - attention pattern
# ============================================================================
print("\n=== FUSED SOFTMAX ===")

def fused_softmax(x, bias):
    """Fused softmax: permute (B, S, H, S) -> (B, H, S, S), add bias, softmax."""
    return F.softmax(x.permute(0, 2, 1, 3) + bias, dim=-1)

for B, S, H in [(8, 512, 16), (4, 1024, 16), (2, 2048, 32)]:
    x = torch.randn(B, S, H, S, dtype=torch.bfloat16, device='cuda')
    bias = torch.randn(1, 1, 1, S, dtype=torch.bfloat16, device='cuda')
    capture_operator(f"fused_softmax_B{B}_S{S}_H{H}", fused_softmax, (x, bias), f"B={B},S={S},H={H}")

# ============================================================================
# 3. LAYER NORM
# ============================================================================
print("\n=== LAYER NORM ===")

def layer_norm_fn(x, weight, bias):
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps=1e-5)

for M, N in [(4096, 1024), (4096, 4096), (8192, 2048), (4096, 8192)]:
    x = torch.randn(M, N, dtype=torch.float16, device='cuda')
    w = torch.randn(N, dtype=torch.float16, device='cuda')
    b = torch.randn(N, dtype=torch.float16, device='cuda')
    capture_operator(f"layer_norm_{M}x{N}", layer_norm_fn, (x, w, b), f"{M}x{N}")

# ============================================================================
# 4. RMS NORM (LLaMA-style)
# ============================================================================
print("\n=== RMS NORM ===")

def rms_norm_fn(x, weight, eps=1e-6):
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return (weight * x_normed).to(input_dtype)

for M, H in [(2048, 4096), (2048, 8192), (4096, 4096), (8192, 4096)]:
    x = torch.randn(M, H, dtype=torch.bfloat16, device='cuda')
    w = torch.ones(H, dtype=torch.bfloat16, device='cuda')
    capture_operator(f"rms_norm_{M}x{H}", rms_norm_fn, (x, w), f"{M}x{H}")

# ============================================================================
# 5. RESIDUAL + RMS NORM (common transformer block pattern)
# ============================================================================
print("\n=== RESIDUAL RMS NORM ===")

def residual_rmsnorm(x, residual, weight, eps=1e-6):
    hidden = x + residual
    variance = hidden.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_normed = hidden * torch.rsqrt(variance + eps)
    return weight * hidden_normed

for M, H in [(8192, 4096), (4096, 8192), (2048, 4096)]:
    x = torch.randn(M, H, dtype=torch.bfloat16, device='cuda')
    residual = torch.randn(M, H, dtype=torch.bfloat16, device='cuda')
    w = torch.ones(H, dtype=torch.bfloat16, device='cuda')
    capture_operator(f"residual_rmsnorm_{M}x{H}", residual_rmsnorm, (x, residual, w), f"{M}x{H}")

# ============================================================================
# 6. CROSS ENTROPY LOSS
# ============================================================================
print("\n=== CROSS ENTROPY ===")

def cross_entropy_fn(logits, targets):
    return F.cross_entropy(logits, targets)

for BT, V in [(16384, 32768), (4096, 50257), (8192, 131072)]:
    logits = torch.randn(BT, V, dtype=torch.float32, device='cuda')
    targets = torch.randint(0, V, (BT,), device='cuda')
    capture_operator(f"cross_entropy_{BT}x{V}", cross_entropy_fn, (logits, targets), f"BT={BT},V={V}")

# ============================================================================
# 7. KL DIVERGENCE
# ============================================================================
print("\n=== KL DIVERGENCE ===")

def kl_div_fn(log_probs, target_probs):
    return F.kl_div(log_probs, target_probs, reduction='batchmean')

for BT, V in [(4096, 4096), (4096, 32768)]:
    log_probs = torch.randn(BT, V, dtype=torch.float32, device='cuda').log_softmax(dim=-1)
    target_probs = torch.randn(BT, V, dtype=torch.float32, device='cuda').softmax(dim=-1)
    capture_operator(f"kl_div_{BT}x{V}", kl_div_fn, (log_probs, target_probs), f"BT={BT},V={V}")

# ============================================================================
# 8. ROPE (Rotary Position Embedding)
# ============================================================================
print("\n=== ROPE ===")

def rope_fn(q, cos, sin):
    """Apply rotary position embedding."""
    # Standard RoPE: split into halves, rotate
    q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
    return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)

for B, S, H, D in [(1, 2048, 32, 128), (1, 4096, 32, 128), (4, 2048, 32, 128)]:
    q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(1, 1, S, D//2, dtype=torch.bfloat16, device='cuda')
    sin = torch.randn(1, 1, S, D//2, dtype=torch.bfloat16, device='cuda')
    capture_operator(f"rope_B{B}_S{S}_H{H}_D{D}", rope_fn, (q, cos, sin), f"B={B},S={S},H={H},D={D}")

# ============================================================================
# 9. SWIGLU / GEGLU (MLP activations)
# ============================================================================
print("\n=== SWIGLU ===")

def swiglu_fn(gate, up, x):
    """SwiGLU: silu(gate(x)) * up(x) - as the pointwise fusion part."""
    return F.silu(gate) * up

for B, T, H in [(4, 2048, 11008), (8, 1024, 11008), (4, 4096, 11008)]:
    gate = torch.randn(B, T, H, dtype=torch.bfloat16, device='cuda')
    up = torch.randn(B, T, H, dtype=torch.bfloat16, device='cuda')
    # Capture just the pointwise fusion (silu * up), not the matmuls
    capture_operator(f"swiglu_B{B}_T{T}_H{H}", swiglu_fn, (gate, up, None), f"B={B},T={T},H={H}")

print("\n=== GEGLU ===")

def geglu_fn(gate, up):
    """GeGLU: gelu(gate) * up."""
    return F.gelu(gate, approximate='tanh') * up

for B, T, H in [(8, 1024, 11008), (4, 2048, 11008)]:
    gate = torch.randn(B, T, H, dtype=torch.bfloat16, device='cuda')
    up = torch.randn(B, T, H, dtype=torch.bfloat16, device='cuda')
    capture_operator(f"geglu_B{B}_T{T}_H{H}", geglu_fn, (gate, up), f"B={B},T={T},H={H}")

# ============================================================================
# 10. DROPOUT (pointwise with random)
# ============================================================================
print("\n=== DROPOUT ===")

def dropout_fn(x):
    return F.dropout(x, p=0.1, training=True)

for shape in [(4096, 4096), (8192, 8192)]:
    x = torch.randn(*shape, dtype=torch.float16, device='cuda')
    # Note: dropout in no_grad mode won't generate the random pattern
    # We need training=True with grad
    try:
        install_capture_hook(str(CAPTURE_BASE / f"dropout_{'x'.join(map(str, shape))}"),
                           label=f"tritonbench_dropout")
        compiled = torch.compile(dropout_fn, mode="max-autotune-no-cudagraphs")
        # Need to run with grad enabled for dropout to actually do something
        out = compiled(x)
        torch.cuda.synchronize()
        print(f"  [OK] dropout ({'x'.join(map(str, shape))})")
    except Exception as e:
        print(f"  [FAIL] dropout: {e}")
    finally:
        uninstall_capture_hook()

# ============================================================================
# 11. EMBEDDING (gather pattern)
# ============================================================================
print("\n=== EMBEDDING ===")

def embedding_fn(weight, indices):
    return F.embedding(weight, indices)

# Note: embedding is typically not fused by inductor into a triton kernel
# in a meaningful way - it's a gather. Skip unless we see it fuse.

# ============================================================================
# 12. SUM/MEAN REDUCTIONS (standalone)
# ============================================================================
print("\n=== STANDALONE REDUCTIONS ===")

def sum_reduction(x):
    return x.sum(dim=-1)

def mean_reduction(x):
    return x.mean(dim=-1)

def sum_all(x):
    return x.sum()

for M, N in [(4096, 4096), (8192, 8192), (32768, 1024)]:
    x = torch.randn(M, N, dtype=torch.float32, device='cuda')
    capture_operator(f"sum_dim1_{M}x{N}", sum_reduction, (x,), f"{M}x{N}")
    capture_operator(f"mean_dim1_{M}x{N}", mean_reduction, (x,), f"{M}x{N}")

for N in [1048576, 16777216]:
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    capture_operator(f"sum_all_{N}", sum_all, (x,), f"N={N}")

# ============================================================================
# 13. VECTOR OPS (elementwise baselines)
# ============================================================================
print("\n=== VECTOR OPS ===")

def vector_add(x, y):
    return x + y

def vector_exp(x):
    return torch.exp(x)

for N in [1048576, 16777216]:
    x = torch.randn(N, dtype=torch.float32, device='cuda')
    y = torch.randn(N, dtype=torch.float32, device='cuda')
    capture_operator(f"vector_add_{N}", vector_add, (x, y), f"N={N}")
    capture_operator(f"vector_exp_{N}", vector_exp, (x,), f"N={N}")

# ============================================================================
# 14. WELFORD (online variance computation for layer norm)
# ============================================================================
print("\n=== WELFORD (layer norm via variance) ===")

def welford_layer_norm(x, weight, bias, eps=1e-5):
    """Layer norm expressed as explicit welford-style mean + variance."""
    x_fp32 = x.to(torch.float32)
    mean = x_fp32.mean(dim=-1, keepdim=True)
    var = ((x_fp32 - mean) ** 2).mean(dim=-1, keepdim=True)
    x_hat = (x_fp32 - mean) * torch.rsqrt(var + eps)
    return (weight * x_hat + bias).to(x.dtype)

for M, N in [(262144, 1024), (262144, 4096), (65536, 8192)]:
    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    w = torch.randn(N, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(N, dtype=torch.bfloat16, device='cuda')
    capture_operator(f"welford_ln_{M}x{N}", welford_layer_norm, (x, w, b), f"{M}x{N}")

# ============================================================================
# 15. INDUCTOR FUSED LINEAR + GELU (common MLP pattern)
# ============================================================================
print("\n=== FUSED LINEAR GELU (pointwise part only) ===")

def linear_gelu_fused(x, bias):
    """The pointwise fusion after matmul: bias_add + gelu."""
    return F.gelu(x + bias, approximate='tanh')

for M, N in [(8192, 4096), (4096, 11008), (2048, 8192)]:
    x = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
    capture_operator(f"bias_gelu_{M}x{N}", linear_gelu_fused, (x, bias), f"{M}x{N}")


print("\n\n=== CAPTURE COMPLETE ===")
print(f"All captures in: {CAPTURE_BASE}")

# List what was captured
import glob
all_indexes = sorted(glob.glob(str(CAPTURE_BASE / "*/index.json")))
total_regions = 0
for idx_file in all_indexes:
    import json
    with open(idx_file) as f:
        data = json.load(f)
    if data:
        total_regions += len(data)
        op_name = Path(idx_file).parent.name
        kinds = {}
        for entry in data:
            k = entry.get("kind", "unknown")
            kinds[k] = kinds.get(k, 0) + 1
        print(f"  {op_name}: {len(data)} regions ({kinds})")

print(f"\nTotal regions captured: {total_regions}")
