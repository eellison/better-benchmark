"""Benchmark: baseline vs fused GQA RoPE + expand"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch, time, os
import torch._inductor.config as cfg
import torch._inductor.metrics as m

cfg.force_disable_caches = True

# Control via env var
mode = os.environ.get("FUSION_MODE", "fused")
if mode == "baseline":
    cfg.expand_recompute_limit = 0
else:
    pass  # default: expand_recompute_limit = 8

B, S, H_q, H_kv, D = 4, 512, 32, 8, 128

class Repro(torch.nn.Module):
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
        k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, 4, S, D).clone().reshape(B, H_q, S, D)
        return q_embed, k_expanded

inputs = [
    torch.randn(2048, 4096, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, 512, 128, dtype=torch.bfloat16, device='cuda'),
    torch.randn(1, 512, 128, dtype=torch.bfloat16, device='cuda'),
    torch.randn(2048, 1024, dtype=torch.bfloat16, device='cuda'),
]

m.reset()
compiled = torch.compile(Repro().cuda())
with torch.no_grad():
    out = compiled(*inputs)
print(f'{mode}: {m.generated_kernel_count} kernels')

for _ in range(50):
    with torch.no_grad(): compiled(*inputs)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(500):
    with torch.no_grad(): compiled(*inputs)
torch.cuda.synchronize()
us = (time.perf_counter() - t0) / 500 * 1e6
print(f'  Time: {us:.1f} us')
