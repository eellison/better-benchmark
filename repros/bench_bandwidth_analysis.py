"""Analyze achieved bandwidth of the fused RoPE kernel.

The kernel has:
- 10 loads per output element (but many are redundant/from cache)
- 2 stores per output element
- Output elements: 8,388,608

True HBM traffic depends on cache behavior. Let's estimate.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch, time
import torch._inductor.config as cfg

cfg.force_disable_caches = True

B, S, H_q, H_kv, D = 4, 512, 32, 8, 128
N_rep = H_q // H_kv  # 4

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

inputs = [
    torch.randn(B*S, H_q*D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda'),
    torch.randn(B*S, H_kv*D, dtype=torch.bfloat16, device='cuda'),
]

compiled = torch.compile(RoPEFused().cuda())
with torch.no_grad():
    compiled(*inputs)

# CUDA graph timing
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream), torch.no_grad():
    compiled(*inputs)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g), torch.no_grad():
    out = compiled(*inputs)
torch.cuda.synchronize()
for _ in range(200):
    g.replay()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(2000):
    g.replay()
torch.cuda.synchronize()
us = (time.perf_counter() - t0) / 2000 * 1e6

print(f"Fused RoPE kernel time: {us:.2f} us")
print()

# Calculate traffic under different cache assumptions
print("=== HBM Traffic Analysis ===")
elem_bf16 = 2  # bytes

# Unique data accessed:
q_proj_bytes = B * S * H_q * D * elem_bf16  # 16 MB - Q projection (2048 x 4096)
k_proj_bytes = B * S * H_kv * D * elem_bf16  # 4 MB - K projection (2048 x 1024)
cos_bytes = S * D * elem_bf16  # 128 KB
sin_bytes = S * D * elem_bf16  # 128 KB
q_out_bytes = B * H_q * S * D * elem_bf16  # 16 MB
k_out_bytes = B * H_q * S * D * elem_bf16  # 16 MB (after expand)

print(f"Input unique data:")
print(f"  q_proj: {q_proj_bytes/1e6:.1f} MB")
print(f"  k_proj: {k_proj_bytes/1e6:.1f} MB")
print(f"  cos:    {cos_bytes/1e6:.3f} MB")
print(f"  sin:    {sin_bytes/1e6:.3f} MB")
print(f"Output data:")
print(f"  q_out:  {q_out_bytes/1e6:.1f} MB")
print(f"  k_out:  {k_out_bytes/1e6:.1f} MB")
print()

# Scenario 1: Perfect caching (only unique data touches HBM once)
perfect_cache = q_proj_bytes + k_proj_bytes + cos_bytes + sin_bytes + q_out_bytes + k_out_bytes
print(f"Scenario 1: Perfect L2 cache (unique data only)")
print(f"  Total: {perfect_cache/1e6:.1f} MB")
print(f"  Achieved BW: {perfect_cache / (us * 1e-6) / 1e12:.2f} TB/s")
print(f"  vs SOL time: {perfect_cache / 7.8e12 * 1e6:.2f} us (ratio: {us / (perfect_cache / 7.8e12 * 1e6):.2f}x)")
print()

# Scenario 2: No caching on K expand (K read 4x from HBM)
no_k_cache = q_proj_bytes + 4*k_proj_bytes + cos_bytes + sin_bytes + q_out_bytes + k_out_bytes
print(f"Scenario 2: K data read 4x (no cache on expand)")
print(f"  Total: {no_k_cache/1e6:.1f} MB")
print(f"  Achieved BW: {no_k_cache / (us * 1e-6) / 1e12:.2f} TB/s")
print(f"  vs SOL time: {no_k_cache / 7.8e12 * 1e6:.2f} us (ratio: {us / (no_k_cache / 7.8e12 * 1e6):.2f}x)")
print()

# Scenario 3: Q also read 3x (for q, q[+64], q[-64] in cat pattern)
# Q is read at 3 different offsets per element. If cache line serves all 3, no penalty.
# But the access pattern is: q[i], q[i+64*stride], q[i-64*stride]
# With 128 elements per head and contiguous innermost, q[i] and q[i+64] are in same 128B cache line
# Actually for bf16, a 128B cache line holds 64 elements. D=128 means 2 cache lines per head per (B,S) position.
# So q[0:64] and q[64:128] are in different cache lines. The cat pattern reads both for each output element.
# This means Q is read from 2 cache lines per output position regardless = Q is effectively read once.
# Same analysis: K is read from 2 cache lines per output position.

# The real question: does the expand pattern cause 4x K HBM reads?
# K iteration order in kernel: x0=D(128), x4=S(512), x6=H_kv(8), x3=B(4)
# But the kernel iterates [B, H_q, S, D] where H_q=32 = H_kv*N_rep = 8*4
# Consecutive elements in H_q that map to same H_kv head are spaced by 262144 (=4*S*D) apart in output
# So 4 threads reading same K data are far apart in the grid - L2 may not help

# Scenario 4: All loads from HBM (worst case)
# 10 loads per element, but what are they?
# From kernel code:
# tmp0 = in_ptr0[x7]           = q_proj (contiguous)
# tmp1 = in_ptr1[x0+128*x2]   = cos (broadcast over B,H)
# tmp17 = in_ptr2[x0+128*x2]  = sin (broadcast over B,H)
# tmp20 = in_ptr3[...]         = k_proj (strided)
# tmp21 = in_ptr1[x10]         = cos again (different index pattern for K path)
# tmp29 = in_ptr2[x10]         = sin again
# tmp8 = in_ptr0[64+128*x8+x0] = q_proj at +64 offset (for cat)
# tmp15 = in_ptr0[128*x8+(x0-64)] = q_proj at -64 offset (for cat)
# tmp23 = in_ptr3[64+...]      = k_proj at +64
# tmp27 = in_ptr3[...(x0-64)]  = k_proj at -64

# Q: loaded 3x but at offsets within same 256B (2 cache lines of bf16) - cached in L1
# K: loaded 3x per H_kv head, then that whole pattern repeated 4x for expand
# cos/sin: loaded 2x (once for Q path, once for K path) but tiny -> L2

# Most likely scenario: Q and cos/sin are cached, K depends on L2 capacity
# L2 cache on B200 is 96MB. K_proj is only 4MB -> should fit easily!
# But access pattern matters more than size.

print(f"Scenario 4: Assume Q cached, K 4x from L2 (likely)")
q_effective = q_proj_bytes  # Q read once from HBM, cat offsets from L1
k_effective = k_proj_bytes  # K read once from HBM into L2, then served 4x from L2
cos_sin_eff = cos_bytes + sin_bytes  # tiny, definitely cached
writes = q_out_bytes + k_out_bytes
realistic = q_effective + k_effective + cos_sin_eff + writes
print(f"  Total HBM: {realistic/1e6:.1f} MB")
print(f"  Achieved BW: {realistic / (us * 1e-6) / 1e12:.2f} TB/s")
print(f"  vs SOL time: {realistic / 7.8e12 * 1e6:.2f} us (ratio: {us / (realistic / 7.8e12 * 1e6):.2f}x)")
print()

# Let's also measure what bandwidth a simple copy achieves at same size
print("=== Reference: copy bandwidth at similar sizes ===")
for size_mb in [16, 32, 52]:
    n = int(size_mb * 1e6 / 2)  # bf16 elements
    x = torch.randn(n, dtype=torch.bfloat16, device='cuda')
    fn = torch.compile(lambda x: x.clone())
    with torch.no_grad(): fn(x)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2), torch.no_grad():
        y = fn(x)
    torch.cuda.synchronize()
    for _ in range(200): g2.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2000): g2.replay()
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 2000 * 1e6
    bw = (2 * size_mb * 1e6) / (t * 1e-6) / 1e12  # read + write
    print(f"  {size_mb:3d} MB copy: {t:.2f} us, BW: {bw:.2f} TB/s")
