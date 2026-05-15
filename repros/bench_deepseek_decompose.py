"""Decompose DeepSeek-V3 RoPE kernel overhead.

The kernel outputs [4,128,512,192] = 50M elements with:
- First 128/192 elements: pass-through (just copy)
- Last 64/192 elements: interleaved RoPE + 128x broadcast expand for K

This benchmark isolates each overhead component.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch, time
import torch._inductor.config as cfg
import torch._inductor.metrics as m
cfg.force_disable_caches = True

B, H, S, D_pass, D_rope = 4, 128, 512, 128, 64

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

# Total output: 2 * [4,128,512,192] = 2 * 50M bf16 = 200 MB writes
# Total inputs: q_rope[4,128,512,64]=32MB + cos[1,512,64]=64KB + sin[1,512,64]=64KB
#              + q_pass[4,128,512,128]=64MB + k_rope[4,512,64]=256KB + k_pass[4,128,512,128]=64MB
# Total read: ~160 MB, Total write: ~200 MB, Grand total: ~360 MB
total_bytes = 360 * 1e6
print(f"Total traffic: ~360 MB")
print(f"SOL at 6.4 TB/s (measured for 360MB): {total_bytes/6.4e12*1e6:.1f} us")
print()

# --- 1. Full DeepSeek RoPE (the actual kernel) ---
class FullDeepSeek(torch.nn.Module):
    def forward(self, q_rope, cos, sin, q_pass, k_rope, k_pass):
        # Q: interleave + RoPE + cat with pass
        q_int = q_rope.view(B,H,S,D_rope//2,2).permute(0,1,2,4,3).contiguous().view(B,H,S,D_rope)
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        q_mul = q_int * cos_q
        q1, q2 = q_int[..., :D_rope//2], q_int[..., D_rope//2:]
        q_rot = torch.cat((-q2, q1), dim=-1)
        q_embed = q_mul + q_rot * sin_q
        query = torch.cat((q_pass, q_embed), dim=-1)
        # K: interleave + RoPE + expand + cat with pass
        k_r = k_rope.view(B,1,S,D_rope)
        k_int = k_r.view(B,1,S,D_rope//2,2).permute(0,1,2,4,3).contiguous().view(B,1,S,D_rope)
        k_mul = k_int * cos_q
        k1, k2 = k_int[..., :D_rope//2], k_int[..., D_rope//2:]
        k_rot = torch.cat((-k2, k1), dim=-1)
        k_embed = k_mul + k_rot * sin_q
        k_exp = k_embed.expand(B, H, S, D_rope)
        key = torch.cat((k_pass, k_exp), dim=-1)
        return query, key

# --- 2. Just the cat (pass-through copy + identity for rope portion) ---
class JustCat(torch.nn.Module):
    def forward(self, q_rope, cos, sin, q_pass, k_rope, k_pass):
        # Q: just copy rope portion + cat
        query = torch.cat((q_pass, q_rope), dim=-1)
        # K: expand + cat
        k_r = k_rope.view(B,1,S,D_rope).expand(B,H,S,D_rope)
        key = torch.cat((k_pass, k_r.contiguous()), dim=-1)
        return query, key

# --- 3. No cat, just copy all at output size ---
class JustCopy(torch.nn.Module):
    def forward(self, q_out, k_out):
        return q_out.clone(), k_out.clone()

# --- 4. RoPE without interleave, no cat (to isolate interleave cost) ---
class RoPENoCatNoInterleave(torch.nn.Module):
    def forward(self, q_rope, cos, sin, q_pass, k_rope, k_pass):
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        # Q: standard RoPE (no interleave)
        q_mul = q_rope * cos_q
        q1, q2 = q_rope[..., :D_rope//2], q_rope[..., D_rope//2:]
        q_rot = torch.cat((-q2, q1), dim=-1)
        q_embed = q_mul + q_rot * sin_q
        query = torch.cat((q_pass, q_embed), dim=-1)
        # K: standard RoPE + expand + cat
        k_r = k_rope.view(B,1,S,D_rope)
        k_mul = k_r * cos_q
        k1, k2 = k_r[..., :D_rope//2], k_r[..., D_rope//2:]
        k_rot = torch.cat((-k2, k1), dim=-1)
        k_embed = k_mul + k_rot * sin_q
        k_exp = k_embed.expand(B, H, S, D_rope)
        key = torch.cat((k_pass, k_exp), dim=-1)
        return query, key

# --- 5. Simple ALU, same traffic, no branches ---
class SimpleALU(torch.nn.Module):
    def forward(self, q_rope, cos, sin, q_pass, k_rope, k_pass):
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        q_embed = q_rope * cos_q + q_rope * sin_q
        query = torch.cat((q_pass, q_embed), dim=-1)
        k_r = k_rope.view(B,1,S,D_rope)
        k_embed = k_r * cos_q + k_r * sin_q
        k_exp = k_embed.expand(B, H, S, D_rope)
        key = torch.cat((k_pass, k_exp), dim=-1)
        return query, key

inputs = [
    torch.randn(B, H, S, D_rope, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, S, D_rope, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, S, D_rope, dtype=torch.bfloat16, device='cuda'),
    torch.randn(B, H, S, D_pass, dtype=torch.bfloat16, device='cuda'),
    torch.randn(B, S, D_rope, dtype=torch.bfloat16, device='cuda'),
    torch.randn(B, H, S, D_pass, dtype=torch.bfloat16, device='cuda'),
]

copy_inputs = [
    torch.randn(B, H, S, D_pass+D_rope, dtype=torch.bfloat16, device='cuda'),
    torch.randn(B, H, S, D_pass+D_rope, dtype=torch.bfloat16, device='cuda'),
]

models = [
    ("Full DeepSeek (interleave+RoPE+cat+expand)", FullDeepSeek(), inputs),
    ("No interleave (RoPE+cat+expand)", RoPENoCatNoInterleave(), inputs),
    ("Simple ALU (mul+add+cat+expand)", SimpleALU(), inputs),
    ("Just cat (copy+expand+cat)", JustCat(), inputs),
    ("Just copy (2x clone 192MB)", JustCopy(), copy_inputs),
]

print(f"{'Variant':50s} {'Kernels':>7s} {'Time(us)':>8s} {'vs copy':>8s}")
print("-" * 80)
for name, model, ins in models:
    m.reset()
    compiled = torch.compile(model.cuda())
    with torch.no_grad(): compiled(*ins)
    nk = m.generated_kernel_count
    us = bench_graph(compiled, ins)
    print(f"  {name:48s} {nk:7d} {us:8.1f}")
