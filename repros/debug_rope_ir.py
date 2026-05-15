"""Debug what IR Inductor produces for Llama-style RoPE."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
cfg.force_disable_caches = True

import logging
logging.getLogger("torch._inductor.graph").setLevel(logging.DEBUG)

B, S, H, D = 4, 512, 32, 128
D_HALF = D // 2

# Llama form: x*cos + cat(-x2, x1)*sin -> single Pointwise via pointwise_cat
class LlamaRoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)  # [1,1,S,D]
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x2 = x[..., D_HALF:]
        x1 = x[..., :D_HALF]
        rotated = torch.cat([-x2, x1], dim=-1)
        return x * cos_e + rotated * sin_e

# Two-output form: cat([x1*cos-x2*sin, x1*sin+x2*cos]) -> ConcatKernel
class TwoOutputRoPE(torch.nn.Module):
    def forward(self, x, cos, sin):
        cos_e = cos.unsqueeze(0).unsqueeze(0)
        sin_e = sin.unsqueeze(0).unsqueeze(0)
        x1 = x[..., :D_HALF]
        x2 = x[..., D_HALF:]
        o1 = x1 * cos_e - x2 * sin_e
        o2 = x1 * sin_e + x2 * cos_e
        return torch.cat([o1, o2], dim=-1)

x = torch.randn(B, H, S, D, dtype=torch.bfloat16, device='cuda')
cos = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')  # full D for Llama form
sin = torch.randn(S, D, dtype=torch.bfloat16, device='cuda')

print("=" * 60)
print("Llama form IR (x*cos + cat(-x2,x1)*sin)")
print("=" * 60)
torch._dynamo.reset()
compiled = torch.compile(LlamaRoPE().cuda())
with torch.no_grad():
    out = compiled(x, cos, sin)

print("\n" + "=" * 60)
print("Two-output form IR (cat([x1*cos-x2*sin, x1*sin+x2*cos]))")
print("=" * 60)

cos_half = torch.randn(S, D_HALF, dtype=torch.bfloat16, device='cuda')
sin_half = torch.randn(S, D_HALF, dtype=torch.bfloat16, device='cuda')

torch._dynamo.reset()
compiled2 = torch.compile(TwoOutputRoPE().cuda())
with torch.no_grad():
    out2 = compiled2(x, cos_half, sin_half)
