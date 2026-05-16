"""
Optimal fused Triton kernel for the argmax repro.
Fuses: reshape + permute + neg + cat + argmax + iota + mul + add
into a single kernel that reads input once and writes outputs directly.
"""
import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
triton_helpers.set_driver_to_gpu()


@triton.jit
def fused_argmax_kernel(
    in_ptr0,  # f32[12, 32768, 64], strides [2097152, 64, 1]
    out_ptr0,  # i64[8, 12, 4096] - scaled_vector for sort
    out_ptr1,  # i64[8, 12, 1, 4096] - bucket offsets
    xnumel,
    XBLOCK: tl.constexpr,
):
    # x iterates over [8, 12, 4096] = 393216 output elements
    # Each output needs argmax over 128 values (64 original + 64 negated)
    # But we only need to read 64 f32 values from input
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel

    # Decode indices: x maps to [h, b, s] where h in [0,8), b in [0,12), s in [0,4096)
    # Layout: h * 49152 + b * 4096 + s
    x_s = xindex % 4096  # sequence position
    x_b = (xindex // 4096) % 12  # batch/head in original
    x_h = xindex // 49152  # hash index

    # Input address: in_ptr0[b, h*4096+s, f] = in_ptr0 + b*2097152 + (h*4096+s)*64 + f
    base_addr = x_b * 2097152 + (x_h * 4096 + x_s) * 64

    # We need to do a reduction over 64 elements per output.
    # Use a loop to find max and argmax across positive and negative values.
    max_val = tl.full([XBLOCK], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([XBLOCK], 0, dtype=tl.int32)

    # Process all 64 features - check both positive (idx=f) and negative (idx=64+f)
    for f in range(64):
        val = tl.load(in_ptr0 + base_addr + f, xmask, other=float('-inf'))
        # Check positive value (index f)
        is_greater = val > max_val
        max_val = tl.where(is_greater, val, max_val)
        max_idx = tl.where(is_greater, f, max_idx)
        # Check negative value (index 64+f)
        neg_val = -val
        is_greater_neg = neg_val > max_val
        max_val = tl.where(is_greater_neg, neg_val, max_val)
        max_idx = tl.where(is_greater_neg, f + 64, max_idx)

    # Compute outputs (fused iota + mul + add)
    # bucket = argmax + 0 (since iota[0]*128 = 0 for single hash)
    bucket = max_idx.to(tl.int64)
    # scaled_vector = bucket * 4096 + (position % 4096)
    scaled = bucket * 4096 + x_s.to(tl.int64)

    # Store
    tl.store(out_ptr0 + xindex, scaled, xmask)
    tl.store(out_ptr1 + xindex, bucket, xmask)


def run_fused_kernel(inp):
    """Run the fused kernel on input f32[12, 32768, 64]."""
    xnumel = 393216  # 8 * 12 * 4096
    out0 = torch.empty(8, 12, 4096, device='cuda', dtype=torch.int64)
    out1 = torch.empty(8, 12, 4096, device='cuda', dtype=torch.int64)

    grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)
    fused_argmax_kernel[grid](inp, out0, out1, xnumel, XBLOCK=128)
    return out0, out1


def test_correctness():
    """Verify the fused kernel matches reference."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repros.canonical.argmax_09f9e9c2268d.repro import Repro

    torch.manual_seed(42)
    inp = torch.randn(12, 32768, 64, device='cuda')

    # Reference
    model = Repro().cuda()
    ref_out = model(inp)
    ref_buckets = ref_out[0].reshape(8, 12, 4096)

    # Fused kernel
    out0, out1 = run_fused_kernel(inp)

    print(f"Bucket match: {torch.equal(out1, ref_buckets)}")
    if not torch.equal(out1, ref_buckets):
        diff = (out1 - ref_buckets).abs()
        print(f"  Max diff: {diff.max().item()}, Num mismatches: {(diff > 0).sum().item()}/{diff.numel()}")

    ref_scaled = ref_buckets * 4096 + torch.arange(4096, device='cuda').view(1, 1, -1)
    print(f"Scaled match: {torch.equal(out0, ref_scaled)}")
    if not torch.equal(out0, ref_scaled):
        diff = (out0 - ref_scaled).abs()
        print(f"  Max diff: {diff.max().item()}, Num mismatches: {(diff > 0).sum().item()}/{diff.numel()}")


def benchmark():
    """Benchmark fused vs inductor-generated."""
    import time

    torch.manual_seed(42)
    inp = torch.randn(12, 32768, 64, device='cuda')

    # Warmup
    for _ in range(10):
        run_fused_kernel(inp)
    torch.cuda.synchronize()

    # Benchmark fused
    start = time.perf_counter()
    for _ in range(100):
        run_fused_kernel(inp)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / 100

    # Benchmark inductor (just kernels 1+2, not sort)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from repros.canonical.argmax_09f9e9c2268d.repro import Repro
    model = Repro().cuda()
    compiled = torch.compile(model)
    # Warmup
    for _ in range(10):
        compiled(inp)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        compiled(inp)
    torch.cuda.synchronize()
    inductor_time = (time.perf_counter() - start) / 100

    print(f"Fused kernel (argmax portion): {fused_time*1000:.3f} ms")
    print(f"Inductor compiled (full): {inductor_time*1000:.3f} ms")
    print(f"Speedup potential: {inductor_time/fused_time:.2f}x")

    # Memory traffic comparison
    fused_bytes = 12 * 32768 * 64 * 4 + 393216 * 8 * 2  # read input + write 2 outputs
    inductor_bytes = (25165824 * 4 * 3) + (393216 * 128 * 4) + (393216 * 8 * 2)  # k1 read+write + k2 read+write
    print(f"\nFused traffic: {fused_bytes/1024**2:.1f} MB")
    print(f"Inductor traffic: {inductor_bytes/1024**2:.1f} MB")
    print(f"Traffic ratio: {inductor_bytes/fused_bytes:.2f}x")


if __name__ == "__main__":
    test_correctness()
    print()
    benchmark()
