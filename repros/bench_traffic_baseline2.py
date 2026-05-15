"""Refined benchmark: isolate ALU vs memory overhead."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch, time
import torch._inductor.config as cfg
import torch._inductor.metrics as m

cfg.force_disable_caches = True

B, S, H_q, H_kv, D = 4, 512, 32, 8, 128
N_rep = H_q // H_kv  # 4

def bench(fn, inputs, name, warmup=100, iters=1000):
    for _ in range(warmup):
        with torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()

    # With CUDA graphs only (eliminate launch overhead)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        fn(*inputs)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        out = fn(*inputs)
    torch.cuda.synchronize()

    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    us_graph = (time.perf_counter() - t0) / iters * 1e6
    return us_graph

# --- 1. Full RoPE fused ---
class RoPEFused(torch.nn.Module):
    def forward(self, q_proj, cos, sin, k_proj):
        q = q_proj.view(B, S, H_q, D).permute(0, 2, 1, 3)
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        q1, q2 = q[..., :D//2], q[..., D//2:]
        q_rotated = torch.cat((-q2, q1), dim=-1)
        q_embed = q * cos_q + q_rotated * sin_q
        k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)
        k1, k2 = k[..., :D//2], k[..., D//2:]
        k_rotated = torch.cat((-k2, k1), dim=-1)
        k_embed = k * cos_q + k_rotated * sin_q
        k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D).clone().reshape(B, H_q, S, D)
        return q_embed, k_expanded

# --- 2. RoPE without cat (just mul+add, no index tricks) ---
class RoPENoCat(torch.nn.Module):
    def forward(self, q_proj, cos, sin, k_proj):
        q = q_proj.view(B, S, H_q, D).permute(0, 2, 1, 3)
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        # Simplified: just q*cos + q*sin (no cat/slice)
        q_embed = q * cos_q + q * sin_q
        k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)
        k_embed = k * cos_q + k * sin_q
        k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D).clone().reshape(B, H_q, S, D)
        return q_embed, k_expanded

# --- 3. Just expand (k -> expanded, no RoPE) + identity for q ---
class JustExpand(torch.nn.Module):
    def forward(self, q_proj, cos, sin, k_proj):
        q = q_proj.view(B, S, H_q, D).permute(0, 2, 1, 3).contiguous()
        k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)
        k_expanded = k.unsqueeze(2).expand(B, H_kv, N_rep, S, D).clone().reshape(B, H_q, S, D)
        return q, k_expanded

# --- 4. Contiguous layout (no permute) full RoPE ---
class RoPEContiguous(torch.nn.Module):
    def forward(self, q, cos, sin, k):
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        q1, q2 = q[..., :D//2], q[..., D//2:]
        q_rotated = torch.cat((-q2, q1), dim=-1)
        q_embed = q * cos_q + q_rotated * sin_q
        k1, k2 = k[..., :D//2], k[..., D//2:]
        k_rotated = torch.cat((-k2, k1), dim=-1)
        k_embed = k * cos_q + k_rotated * sin_q
        k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D).clone().reshape(B, H_q, S, D)
        return q_embed, k_expanded

# --- 5. All contiguous, no permute, no cat ---
class MinimalALU(torch.nn.Module):
    """Same traffic as RoPE but just a*b+c*d per element."""
    def forward(self, q, cos, sin, k):
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        q_embed = q * cos_q + q * sin_q
        k_embed = k * cos_q + k * sin_q
        k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D).clone().reshape(B, H_q, S, D)
        return q_embed, k_expanded

rope_inputs = [
    torch.randn(B*S, H_q*D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(B*S, H_kv*D, dtype=torch.bfloat16, device='cuda'),
]

contig_inputs = [
    torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device='cuda'),
]

print(f"Config: B={B}, S={S}, H_q={H_q}, H_kv={H_kv}, D={D}, N_rep={N_rep}")
total_bytes = (B*H_q*S*D + B*H_kv*S*D + 2*S*D + 2*B*H_q*S*D) * 2
print(f"Total traffic: {total_bytes/1e6:.1f} MB | SOL @ 7.8 TB/s: {total_bytes/7.8e12*1e6:.1f} us")
print()

models = [
    ("Full RoPE (permute+cat+expand)", RoPEFused(), rope_inputs),
    ("RoPE no cat (permute+expand)",   RoPENoCat(), rope_inputs),
    ("Just copy+expand (permute)",     JustExpand(), rope_inputs),
    ("Full RoPE contiguous",           RoPEContiguous(), contig_inputs),
    ("Minimal ALU contiguous",         MinimalALU(), contig_inputs),
]

print(f"{'Variant':40s} {'Kernels':>7s} {'Time(us)':>8s} {'vs SOL':>7s}")
print("-" * 70)
for name, model, inputs in models:
    m.reset()
    compiled = torch.compile(model.cuda())
    with torch.no_grad():
        compiled(*inputs)
    nk = m.generated_kernel_count
    us = bench(compiled, inputs, name)
    sol = total_bytes / 7.8e12 * 1e6
    print(f"  {name:38s} {nk:7d} {us:8.1f} {us/sol:7.2f}x")
