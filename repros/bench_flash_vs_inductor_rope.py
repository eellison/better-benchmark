"""Compare Flash-Attention Triton RoPE vs Inductor-generated kernel.

Tests both GQA case (investigation 02) and DeepSeek interleaved case (investigation 06).
Flash-Attention's approach:
  - Non-interleaved: Two explicit loads at offset 0 and ROTARY_DIM_HALF, direct rotation
  - Interleaved: tl.split(tl.reshape(x, [..., D//2, 2])) for deinterleave, tl.join to recombine
  - 3D grid (heads, seqlen_blocks, batch) vs Inductor's flat 1D iteration
"""
import sys, importlib.util
sys.path.insert(0, "/tmp/pytorch-work")
import torch, time
import torch._inductor.config as cfg
import torch._inductor.metrics as m
cfg.force_disable_caches = True

# Load the triton rotary module directly to avoid flash_attn CUDA dep
spec = importlib.util.spec_from_file_location(
    "rotary", "/tmp/pytorch-work/third_party/flash-attention/flash_attn/ops/triton/rotary.py"
)
rotary_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotary_mod)
apply_rotary = rotary_mod.apply_rotary

def bench_graph(fn, inputs, warmup=200, iters=1000):
    for _ in range(warmup):
        with torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    for _ in range(warmup): g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

def bench_eager(fn, inputs, warmup=200, iters=1000):
    for _ in range(warmup):
        with torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad(): fn(*inputs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6

print("=" * 80)
print("CASE 1: GQA Q RoPE (Investigation 02)")
print("  Shape: [B=4, S=512, H=32, D=128], rotary_dim=128")
print("=" * 80)

B, S, H, D = 4, 512, 32, 128
rotary_dim = D

# Flash-Attention expects [B, S, H, D] layout
x_bshd = torch.randn(B, S, H, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, rotary_dim // 2, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, rotary_dim // 2, dtype=torch.bfloat16, device='cuda')

# Flash-Attention Triton kernel
def flash_rope_gqa(x, cos, sin):
    return apply_rotary(x, cos, sin, interleaved=False)

t_flash = bench_eager(flash_rope_gqa, [x_bshd, cos, sin])
print(f"  Flash-Attention Triton:  {t_flash:.1f} us")

# Inductor (torch.compile) - same operation
class RoPEModule(torch.nn.Module):
    def forward(self, x, cos, sin):
        # Standard non-interleaved RoPE
        cos_expanded = cos.unsqueeze(1)  # [S, 1, D//2]
        sin_expanded = sin.unsqueeze(1)  # [S, 1, D//2]
        x1 = x[..., :D//2]
        x2 = x[..., D//2:]
        o1 = x1 * cos_expanded - x2 * sin_expanded
        o2 = x1 * sin_expanded + x2 * cos_expanded
        return torch.cat([o1, o2], dim=-1)

# Inductor needs [B, S, H, D] with cos [S, D//2]
x_for_inductor = x_bshd.clone()  # [B, S, H, D]
m.reset()
rope_compiled = torch.compile(RoPEModule().cuda())
with torch.no_grad(): rope_compiled(x_for_inductor, cos, sin)
nk = m.generated_kernel_count
t_inductor = bench_graph(rope_compiled, [x_for_inductor, cos, sin])
print(f"  Inductor (compile):      {t_inductor:.1f} us  ({nk} kernels)")

# Now the ACTUAL GQA case: [B, H, S, D] with permute
class RoPEWithPermute(torch.nn.Module):
    def forward(self, x_bhsd, cos, sin):
        # x is [B,H,S,D], cos is [S, D//2]
        # Typical pattern: cos needs broadcasting over B and H
        cos_expanded = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D//2]
        sin_expanded = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, S, D//2]
        x1 = x_bhsd[..., :D//2]
        x2 = x_bhsd[..., D//2:]
        o1 = x1 * cos_expanded - x2 * sin_expanded
        o2 = x1 * sin_expanded + x2 * cos_expanded
        return torch.cat([o1, o2], dim=-1)

x_bhsd = x_bshd.permute(0, 2, 1, 3).contiguous()  # [B, H, S, D]
m.reset()
rope_bhsd_compiled = torch.compile(RoPEWithPermute().cuda())
with torch.no_grad(): rope_bhsd_compiled(x_bhsd, cos, sin)
nk2 = m.generated_kernel_count
t_inductor_bhsd = bench_graph(rope_bhsd_compiled, [x_bhsd, cos, sin])
print(f"  Inductor (BHSD layout):  {t_inductor_bhsd:.1f} us  ({nk2} kernels)")

# Traffic analysis
total_bytes = B * S * H * D * 2 * 2 + S * D * 2  # input + output + cos/sin
print(f"\n  Traffic: {total_bytes/1e6:.1f} MB")
print(f"  SOL at 7.2 TB/s: {total_bytes/7.2e12*1e6:.1f} us")

print()
print("=" * 80)
print("CASE 2: DeepSeek-V3 Interleaved RoPE (Investigation 06)")
print("  Shape: [B=4, H=128, S=512, D_rope=64], interleaved pattern")
print("=" * 80)

B2, H2, S2, D_rope = 4, 128, 512, 64

# Flash-Attention expects [B, S, H, D]
x_ds = torch.randn(B2, S2, H2, D_rope, dtype=torch.bfloat16, device='cuda')
cos_ds = torch.randn(S2, D_rope // 2, dtype=torch.bfloat16, device='cuda')
sin_ds = torch.randn(S2, D_rope // 2, dtype=torch.bfloat16, device='cuda')

# Flash-Attention interleaved
def flash_rope_interleaved(x, cos, sin):
    return apply_rotary(x, cos, sin, interleaved=True)

t_flash_int = bench_eager(flash_rope_interleaved, [x_ds, cos_ds, sin_ds])
print(f"  Flash-Attention Triton (interleaved): {t_flash_int:.1f} us")

# Inductor interleaved (the pattern from DeepSeek)
class InterleavedRoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        # DeepSeek interleave pattern: reshape + permute + clone
        B, S, H, D = x.shape
        x_int = x.view(B, S, H, D//2, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, S, H, D)
        cos_e = cos.unsqueeze(1)  # [S, 1, D//2]
        sin_e = sin.unsqueeze(1)  # [S, 1, D//2]
        x1 = x_int[..., :D//2]
        x2 = x_int[..., D//2:]
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x1 * sin_e + x2 * cos_e
        return torch.cat([o1, o2], dim=-1)

m.reset()
int_compiled = torch.compile(InterleavedRoPE().cuda())
with torch.no_grad(): int_compiled(x_ds, cos_ds, sin_ds)
nk3 = m.generated_kernel_count
t_inductor_int = bench_graph(int_compiled, [x_ds, cos_ds, sin_ds])
print(f"  Inductor (interleaved):               {t_inductor_int:.1f} us  ({nk3} kernels)")

# Traffic for interleaved case
total_bytes_int = B2 * S2 * H2 * D_rope * 2 * 2 + S2 * D_rope * 2
print(f"\n  Traffic: {total_bytes_int/1e6:.1f} MB")
print(f"  SOL at 7.2 TB/s: {total_bytes_int/7.2e12*1e6:.1f} us")

print()
print("=" * 80)
print("CASE 3: Full DeepSeek pattern with cat (the actual problematic case)")
print("  Output: [4, 128, 512, 192] with pass-through + interleaved RoPE")
print("=" * 80)

B3, H3, S3, D_pass, D_r = 4, 128, 512, 128, 64

# Flash-Attention handles ONLY the rotary portion - the cat is separate
x_rope_part = torch.randn(B3, S3, H3, D_r, dtype=torch.bfloat16, device='cuda')
cos3 = torch.randn(S3, D_r // 2, dtype=torch.bfloat16, device='cuda')
sin3 = torch.randn(S3, D_r // 2, dtype=torch.bfloat16, device='cuda')
x_pass = torch.randn(B3, H3, S3, D_pass, dtype=torch.bfloat16, device='cuda')

# Flash: just the rotary part (interleaved)
t_flash_part = bench_eager(flash_rope_interleaved, [x_rope_part, cos3, sin3])
print(f"  Flash-Attention (just RoPE part):      {t_flash_part:.1f} us")

# Full inductor pattern with cat
class FullDeepSeekCat(torch.nn.Module):
    def forward(self, x_pass, x_rope, cos, sin):
        B, H, S, D_p = x_pass.shape
        _, S2, H2, D_r = x_rope.shape
        # Interleave + RoPE
        x_int = x_rope.view(B, S, H, D_r//2, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, S, H, D_r)
        cos_e = cos.unsqueeze(1)
        sin_e = sin.unsqueeze(1)
        x1 = x_int[..., :D_r//2]
        x2 = x_int[..., D_r//2:]
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x1 * sin_e + x2 * cos_e
        rope_out = torch.cat([o1, o2], dim=-1)  # [B, S, H, D_r]
        # Permute to [B, H, S, D_r] then cat with pass-through
        rope_bhsd = rope_out.permute(0, 2, 1, 3)
        return torch.cat([x_pass, rope_bhsd], dim=-1)

m.reset()
full_compiled = torch.compile(FullDeepSeekCat().cuda())
with torch.no_grad(): full_compiled(x_pass, x_rope_part, cos3, sin3)
nk4 = m.generated_kernel_count
t_full = bench_graph(full_compiled, [x_pass, x_rope_part, cos3, sin3])
print(f"  Inductor (full fused, current):        {t_full:.1f} us  ({nk4} kernels)")

total_bytes_full = (B3*H3*S3*(D_pass+D_r) + B3*H3*S3*D_pass + B3*S3*H3*D_r + S3*D_r) * 2
print(f"\n  Total traffic: {total_bytes_full/1e6:.1f} MB")
print(f"  SOL at 6.4 TB/s: {total_bytes_full/6.4e12*1e6:.1f} us")

print()
print("=" * 80)
print("STRATEGY COMPARISON")
print("=" * 80)
print("""
Flash-Attention strategy:
  - Two explicit loads at known offsets (first half + second half)
  - Direct: o0 = x0*cos - x1*sin, o1 = x0*sin + x1*cos
  - No cat(-x2, x1) pattern, no conditional branches
  - For interleaved: tl.split(tl.reshape(x, [..., D//2, 2])) in registers
  - 3D grid (heads, seqlen_blocks, batch) — good spatial locality

Inductor strategy:
  - cat(-x2, x1) creates conditional branching in innermost dimension
  - Single flat 1D kernel iterating all elements
  - Interleaved: fuses stride-2 reads into entire downstream kernel
  - When cat combines pass-through + RoPE, branches amplify

Key insight: Flash-Attention avoids the cat pattern entirely by doing
             two loads and computing rotation directly. This eliminates
             the branch overhead that costs Inductor ~2.1 us on GQA and
             ~136 us on DeepSeek (when amplified by fusion).
""")
