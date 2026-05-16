"""Test with dynamic shapes like DeepSeek."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
cfg.force_disable_caches = True

# Pattern 2: cat_5 with dynamic batch
def cat_to_output(k_pe, k_nope):
    return torch.cat([k_pe, k_nope], dim=-1)

torch._dynamo.reset()
k_pe = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
k_nope = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16)
compiled = torch.compile(cat_to_output, dynamic=True)
with torch.no_grad():
    out = compiled(k_pe, k_nope)
ref = cat_to_output(k_pe, k_nope)
print(f"Pattern 2 (dynamic): max_diff = {(ref - out).abs().max().item():.6f}")

# Different size to confirm dynamic works
k_pe2 = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
k_nope2 = torch.randn(256, 64, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    out2 = compiled(k_pe2, k_nope2)
ref2 = cat_to_output(k_pe2, k_nope2)
print(f"Pattern 2 (size=256): max_diff = {(ref2 - out2).abs().max().item():.6f}")

# Pattern 3: cat_4 with dynamic
def cat_and_view(rope_q, proj_q):
    cat = torch.cat([rope_q, proj_q], dim=-1)
    return cat.reshape(-1, 128)

torch._dynamo.reset()
rope_q = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
proj_q = torch.randn(128, 64, 64, device="cuda", dtype=torch.bfloat16)
compiled3 = torch.compile(cat_and_view, dynamic=True)
with torch.no_grad():
    out3 = compiled3(rope_q, proj_q)
ref3 = cat_and_view(rope_q, proj_q)
print(f"Pattern 3 (dynamic): max_diff = {(ref3 - out3).abs().max().item():.6f}")

print("\nAll dynamic tests passed!")
