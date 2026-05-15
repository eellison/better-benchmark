"""Compare RoPE kernel strategies: hand-written two-load vs Inductor.

The key question: how much does Inductor's cat-based conditional branching cost
compared to an explicit two-load approach?
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch, time
import triton
import triton.language as tl
import torch._inductor.config as cfg
import torch._inductor.metrics as m
cfg.force_disable_caches = True

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2

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


# Strategy 1: Two-load approach (Flash-Attention style)
# Processes BLOCK positions, each loading first half and second half explicitly.
@triton.jit
def rope_two_load_kernel(
    OUT, X, COS, SIN,
    stride_xb, stride_xh, stride_xs,
    stride_cs,
    NUMEL: tl.constexpr,
    S: tl.constexpr, H: tl.constexpr,
    D_HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pos_start = pid * BLOCK
    pos_ids = pos_start + tl.arange(0, BLOCK)
    mask = pos_ids < NUMEL

    # Decode flat position -> (b, h, s)
    s_idx = pos_ids % S
    bh = pos_ids // S
    h_idx = bh % H
    b_idx = bh // H

    # Base offset in X for this position
    x_base = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs

    # Load cos/sin [S, D_HALF] — broadcast over B and H
    cs_base = s_idx * stride_cs
    dk = tl.arange(0, D_HALF)

    # Two explicit loads: first half and second half
    x0 = tl.load(X + (x_base[:, None] + dk[None, :]), mask=mask[:, None]).to(tl.float32)
    x1 = tl.load(X + (x_base[:, None] + (dk[None, :] + D_HALF)), mask=mask[:, None]).to(tl.float32)

    cos_val = tl.load(COS + (cs_base[:, None] + dk[None, :]), mask=mask[:, None]).to(tl.float32)
    sin_val = tl.load(SIN + (cs_base[:, None] + dk[None, :]), mask=mask[:, None]).to(tl.float32)

    # Direct rotation — no cat, no branches
    o0 = x0 * cos_val - x1 * sin_val
    o1 = x0 * sin_val + x1 * cos_val

    # Store both halves
    tl.store(OUT + (x_base[:, None] + dk[None, :]), o0.to(tl.bfloat16), mask=mask[:, None])
    tl.store(OUT + (x_base[:, None] + (dk[None, :] + D_HALF)), o1.to(tl.bfloat16), mask=mask[:, None])


# Strategy 2: Interleaved two-load with tl.split/join (Flash style)
@triton.jit
def rope_interleaved_kernel(
    OUT, X, COS, SIN,
    stride_xb, stride_xh, stride_xs,
    stride_cs,
    NUMEL: tl.constexpr,
    S: tl.constexpr, H: tl.constexpr,
    D: tl.constexpr, D_HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pos_start = pid * BLOCK
    pos_ids = pos_start + tl.arange(0, BLOCK)
    mask = pos_ids < NUMEL

    s_idx = pos_ids % S
    bh = pos_ids // S
    h_idx = bh % H
    b_idx = bh // H

    x_base = b_idx * stride_xb + h_idx * stride_xh + s_idx * stride_xs
    cs_base = s_idx * stride_cs

    # Load full D dimension contiguously
    dk = tl.arange(0, D)
    x = tl.load(X + (x_base[:, None] + dk[None, :]), mask=mask[:, None]).to(tl.float32)

    # Deinterleave in registers: [BLOCK, D] -> [BLOCK, D//2, 2] -> split
    x_reshaped = tl.reshape(x, [BLOCK, D_HALF, 2])
    x0, x1 = tl.split(x_reshaped)  # x0 = evens, x1 = odds

    # Load cos/sin
    dk_half = tl.arange(0, D_HALF)
    cos_val = tl.load(COS + (cs_base[:, None] + dk_half[None, :]), mask=mask[:, None]).to(tl.float32)
    sin_val = tl.load(SIN + (cs_base[:, None] + dk_half[None, :]), mask=mask[:, None]).to(tl.float32)

    # Rotation
    o0 = x0 * cos_val - x1 * sin_val
    o1 = x0 * sin_val + x1 * cos_val

    # Re-interleave in registers
    o = tl.reshape(tl.join(o0, o1), [BLOCK, D])
    tl.store(OUT + (x_base[:, None] + dk[None, :]), o.to(tl.bfloat16), mask=mask[:, None])


def run_two_load(x, cos, sin):
    out = torch.empty_like(x)
    num_pos = x.shape[0] * x.shape[1] * x.shape[2]
    BLOCK = 64
    grid = (triton.cdiv(num_pos, BLOCK),)
    rope_two_load_kernel[grid](
        out, x, cos, sin,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0),
        num_pos, S, H, D_HALF, BLOCK,
    )
    return out


def run_interleaved(x, cos, sin):
    out = torch.empty_like(x)
    num_pos = x.shape[0] * x.shape[1] * x.shape[2]
    BLOCK = 64
    grid = (triton.cdiv(num_pos, BLOCK),)
    rope_interleaved_kernel[grid](
        out, x, cos, sin,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0),
        num_pos, S, H, D, D_HALF, BLOCK,
    )
    return out


print("=" * 70)
print("CASE 1: Standard (non-interleaved) RoPE — GQA shape [4,32,512,128]")
print("=" * 70)

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D_HALF, dtype=torch.bfloat16, device='cuda')
sin = torch.randn(S, D_HALF, dtype=torch.bfloat16, device='cuda')

# Verify
out_hand = run_two_load(x, cos, sin)

class RoPEBHSD(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x1 = x[..., :D_HALF]
        x2 = x[..., D_HALF:]
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x1 * sin_e + x2 * cos_e
        return torch.cat([o1, o2], dim=-1)

m.reset()
compiled = torch.compile(RoPEBHSD().cuda())
with torch.no_grad():
    out_inductor = compiled(x, cos, sin)
nk = m.generated_kernel_count

print(f"Correctness check: {torch.allclose(out_hand, out_inductor, atol=0.05, rtol=0.05)}")
total_bytes = B * H * S * D * 2 * 2 + S * D * 2
print(f"Traffic: {total_bytes/1e6:.1f} MB, SOL at 7.2 TB/s: {total_bytes/7.2e12*1e6:.1f} us")
print()

t_hand = bench_graph(run_two_load, [x, cos, sin])
t_ind = bench_graph(compiled, [x, cos, sin])
print(f"  Hand-written two-load:  {t_hand:.1f} us")
print(f"  Inductor (cat-based):   {t_ind:.1f} us  ({nk} kernel{'s' if nk>1 else ''})")
print(f"  Ratio:                  {t_hand/t_ind:.2f}x (hand/inductor)")

print()
print("=" * 70)
print("CASE 2: Interleaved RoPE — DeepSeek shape [4,128,512,64]")
print("=" * 70)

B2, H2, S2, D2 = 4, 128, 512, 64
D2_HALF = D2 // 2

x_int = torch.randn(B2, H2, S2, D2, dtype=torch.bfloat16, device='cuda')
cos_int = torch.randn(S2, D2_HALF, dtype=torch.bfloat16, device='cuda')
sin_int = torch.randn(S2, D2_HALF, dtype=torch.bfloat16, device='cuda')

# Redefine run_interleaved for these shapes
def run_int(x, cos, sin):
    out = torch.empty_like(x)
    num_pos = x.shape[0] * x.shape[1] * x.shape[2]
    BLOCK = 64
    grid = (triton.cdiv(num_pos, BLOCK),)
    rope_interleaved_kernel[grid](
        out, x, cos, sin,
        x.stride(0), x.stride(1), x.stride(2),
        cos.stride(0),
        num_pos, S2, H2, D2, D2_HALF, BLOCK,
    )
    return out

class InterleavedRoPEBHSD(torch.nn.Module):
    def forward(self, x, cos, sin):
        B, H, S, D = x.shape
        D_half = D // 2
        x_int = x.view(B, H, S, D_half, 2).permute(0, 1, 2, 4, 3).contiguous().view(B, H, S, D)
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x1 = x_int[..., :D_half]
        x2 = x_int[..., D_half:]
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x1 * sin_e + x2 * cos_e
        return torch.cat([o1, o2], dim=-1)

m.reset()
compiled_int = torch.compile(InterleavedRoPEBHSD().cuda())
with torch.no_grad(): compiled_int(x_int, cos_int, sin_int)
nk2 = m.generated_kernel_count

t_hand_int = bench_graph(run_int, [x_int, cos_int, sin_int])
t_ind_int = bench_graph(compiled_int, [x_int, cos_int, sin_int])

total_bytes2 = B2 * H2 * S2 * D2 * 2 * 2 + S2 * D2 * 2
print(f"Traffic: {total_bytes2/1e6:.1f} MB, SOL at 7.2 TB/s: {total_bytes2/7.2e12*1e6:.1f} us")
print()
print(f"  Hand-written (split/join): {t_hand_int:.1f} us")
print(f"  Inductor (interleaved):    {t_ind_int:.1f} us  ({nk2} kernel{'s' if nk2>1 else ''})")
print(f"  Ratio:                     {t_hand_int/t_ind_int:.2f}x (hand/inductor)")

print()
print("=" * 70)
print("CASE 3: Just memcopy baseline (same traffic)")
print("=" * 70)

class JustClone(torch.nn.Module):
    def forward(self, x):
        return x.clone()

m.reset()
compiled_clone = torch.compile(JustClone().cuda())
with torch.no_grad(): compiled_clone(x)
nk3 = m.generated_kernel_count
t_clone = bench_graph(compiled_clone, [x])
print(f"  Clone [4,32,512,128]:   {t_clone:.1f} us  ({nk3} kernel) — pure memcopy bound")

m.reset()
compiled_clone2 = torch.compile(JustClone().cuda())
with torch.no_grad(): compiled_clone2(x_int)
nk4 = m.generated_kernel_count
t_clone2 = bench_graph(compiled_clone2, [x_int])
print(f"  Clone [4,128,512,64]:   {t_clone2:.1f} us  ({nk4} kernel) — pure memcopy bound")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Standard RoPE [4,32,512,128]:
  Hand (two-load):   {t_hand:.1f} us
  Inductor (cat):    {t_ind:.1f} us
  Memcopy bound:     {t_clone:.1f} us
  SOL:               {total_bytes/7.2e12*1e6:.1f} us

Interleaved RoPE [4,128,512,64]:
  Hand (split/join): {t_hand_int:.1f} us
  Inductor:          {t_ind_int:.1f} us
  Memcopy bound:     {t_clone2:.1f} us
  SOL:               {total_bytes2/7.2e12*1e6:.1f} us

Key finding: The two-load strategy {'is' if t_hand < t_ind else 'is NOT'} faster than Inductor.
{'If so, the cat-based branching costs ' + f"{t_ind - t_hand:.1f} us overhead." if t_hand < t_ind else 'Inductor fusion optimizes well for this case.'}
""")
