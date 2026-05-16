"""Test the integrated split_cat_pointwise pass."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import torch._inductor.config as cfg
cfg.force_disable_caches = True

# Test 1: Basic RoPE-style pattern (should split)
print("=== Test 1: RoPE-style cat (should split) ===")
def rope_cat(x, y):
    return torch.cat([x, y], dim=-1)

torch._dynamo.reset()
x = torch.randn(4, 32, 512, 64, device="cuda")
y = torch.randn(4, 32, 512, 64, device="cuda")
ref = rope_cat(x, y)
compiled = torch.compile(rope_cat)
with torch.no_grad():
    out = compiled(x, y)
print(f"  max_diff = {(ref - out).abs().max().item():.6f}")
assert (ref - out).abs().max().item() < 1e-5, "Correctness check failed!"
print("  PASS")

# Test 2: Cat with downstream consumer (should NOT split)
print("\n=== Test 2: Cat with consumer (should NOT split) ===")
def cat_with_consumer(x, y):
    cat = torch.cat([x, y], dim=-1)
    return cat * 2.0  # consumer Pointwise with same numel

torch._dynamo.reset()
x2 = torch.randn(4, 32, 512, 64, device="cuda")
y2 = torch.randn(4, 32, 512, 64, device="cuda")
ref2 = cat_with_consumer(x2, y2)
compiled2 = torch.compile(cat_with_consumer)
with torch.no_grad():
    out2 = compiled2(x2, y2)
print(f"  max_diff = {(ref2 - out2).abs().max().item():.6f}")
assert (ref2 - out2).abs().max().item() < 1e-5, "Correctness check failed!"
print("  PASS")

# Test 3: Non-contiguous strides (like Qwen3-0.6B)
print("\n=== Test 3: Non-contiguous layout ===")
def cat_permuted(x, y):
    # Simulate the RoPE pattern with permuted layout
    x_p = x.permute(0, 2, 1, 3)  # [B, H, S, D] -> [B, S, H, D]
    y_p = y.permute(0, 2, 1, 3)
    cat = torch.cat([x_p, y_p], dim=-1)
    return cat  # No consumer = should split

torch._dynamo.reset()
x3 = torch.randn(8, 16, 512, 64, device="cuda")
y3 = torch.randn(8, 16, 512, 64, device="cuda")
ref3 = cat_permuted(x3, y3)
compiled3 = torch.compile(cat_permuted)
with torch.no_grad():
    out3 = compiled3(x3, y3)
print(f"  max_diff = {(ref3 - out3).abs().max().item():.6f}")
assert (ref3 - out3).abs().max().item() < 1e-5, "Correctness check failed!"
print("  PASS")

print("\n=== All tests passed ===")
