"""Benchmark: isolate memory traffic overhead vs RoPE compute overhead.

Compares:
1. Fused RoPE kernel (10 loads, 2 stores, complex indexing)
2. Simple kernel with same memory traffic (10 loads, 2 stores, just adds)
3. Minimal kernel (2 loads, 1 store) to establish bandwidth baseline
4. SOL estimate from tensor sizes
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch, time
import torch._inductor.config as cfg

cfg.force_disable_caches = True

B, S, H_q, H_kv, D = 4, 512, 32, 8, 128
N_rep = H_q // H_kv  # 4

# --- Workload 1: Actual fused RoPE ---
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

# --- Workload 2: Same traffic pattern, minimal compute ---
# Q path: reads q(8M), cos(64K), sin(64K) -> writes q_out(8M)
# K path: reads k(2M), cos(64K), sin(64K) -> writes k_out(8M) via expand
# Total unique reads: q(8M) + k(2M) + cos(64K) + sin(64K) = ~10.1M bf16 elements
# Total writes: q_out(8M) + k_out(8M) = 16M bf16 elements
# But due to expand, k data is re-read 4x from registers, not HBM
class SimpleTraffic(torch.nn.Module):
    """Same shapes as RoPE but just a+b+c per element."""
    def forward(self, q, cos_broad, sin_broad, k, k_expand_target):
        # Q: 3 reads (q, cos, sin) -> 1 write
        q_out = q + cos_broad + sin_broad
        # K: read k [B,H_kv,S,D], expand to [B,H_q,S,D], add cos+sin -> write
        k_exp = k.unsqueeze(2).expand(B, H_kv, N_rep, S, D).reshape(B, H_q, S, D)
        k_out = k_exp + cos_broad + sin_broad
        return q_out, k_out

# --- Workload 3: Pure bandwidth test (same total bytes) ---
# Total bytes moved by RoPE:
#   Reads: q_proj(8M) + k_proj(2M) + cos(64K) + sin(64K) = 10.25M elements = 20.5 MB
#   Writes: q_embed(8M) + k_expanded(8M) = 16M elements = 32 MB
#   Total: ~52 MB
class BandwidthBaseline(torch.nn.Module):
    """Just copy: total ~52MB of traffic."""
    def forward(self, src1, src2):
        return src1.clone(), src2.clone()

def bench(fn, inputs, name, warmup=100, iters=500):
    for _ in range(warmup):
        with torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()

    # Without CUDA graphs
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    us_no_graph = (time.perf_counter() - t0) / iters * 1e6

    # With CUDA graphs
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

    print(f"  {name:30s}  no-graph: {us_no_graph:6.1f} us  |  graph: {us_graph:5.1f} us")
    return us_no_graph, us_graph

print(f"Config: B={B}, S={S}, H_q={H_q}, H_kv={H_kv}, D={D}")
print(f"Q elements: {B*H_q*S*D/1e6:.1f}M, K elements: {B*H_kv*S*D/1e6:.1f}M")

# Total bytes
q_bytes = B * H_q * S * D * 2  # bf16
k_bytes = B * H_kv * S * D * 2
cos_bytes = 1 * S * D * 2  # broadcast
sin_bytes = cos_bytes
out_q_bytes = q_bytes
out_k_bytes = q_bytes  # after expand
total_bytes = q_bytes + k_bytes + cos_bytes + sin_bytes + out_q_bytes + out_k_bytes
print(f"Total traffic (reads+writes): {total_bytes/1e6:.1f} MB")
print(f"SOL @ 7.8 TB/s: {total_bytes / 7.8e12 * 1e6:.1f} us")
print()

# Compile all
rope_inputs = [
    torch.randn(B*S, H_q*D, dtype=torch.bfloat16, device='cuda'),  # q_proj
    torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda'),      # cos
    torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda'),      # sin
    torch.randn(B*S, H_kv*D, dtype=torch.bfloat16, device='cuda'),  # k_proj
]

simple_inputs = [
    torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device='cuda'),    # q
    torch.randn(B, 1, S, D, dtype=torch.bfloat16, device='cuda'),      # cos broadcast
    torch.randn(B, 1, S, D, dtype=torch.bfloat16, device='cuda'),      # sin broadcast
    torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device='cuda'),   # k
    torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device='cuda'),    # expand target (unused)
]

# For bandwidth baseline: 2 inputs of ~26MB each -> clone both
bw_inputs = [
    torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device='cuda'),    # ~16MB
    torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device='cuda'),    # ~16MB
]

print("Compiling...")
rope_compiled = torch.compile(RoPEFused().cuda())
simple_compiled = torch.compile(SimpleTraffic().cuda())
bw_compiled = torch.compile(BandwidthBaseline().cuda())

# Warmup compile
with torch.no_grad():
    rope_compiled(*rope_inputs)
    simple_compiled(*simple_inputs)
    bw_compiled(*bw_inputs)

print("\nResults:")
bench(rope_compiled, rope_inputs, "Fused RoPE (10 ld, 2 st)")
bench(simple_compiled, simple_inputs, "Simple traffic (expand+add)")
bench(bw_compiled, bw_inputs, "Bandwidth (2x clone 16MB)")

# Also test: what about a single large memcpy?
single_input = [torch.randn(total_bytes // 2, dtype=torch.bfloat16, device='cuda')]
single_compiled = torch.compile(lambda x: x.clone())
with torch.no_grad(): single_compiled(*single_input)
bench(single_compiled, single_input, f"Single clone ({total_bytes//1e6:.0f}MB)")
