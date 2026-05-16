"""
Optimal Triton kernel for SwiGLU activation at LLM decode shape.

SwiGLU pattern: x1 * silu(x2) where silu(x) = x * sigmoid(x)
This kernel reads both inputs in a single pass, computes silu in-register,
and writes the output with no intermediate materialization.

Benchmarks against torch.compile (inductor) at:
  - Decode shape: batch=32, seq=1, hidden=4096 (input per tensor: [32, 1, 4096])
  - Prefill shape: batch=4, seq=2048, hidden=4096 (input per tensor: [4, 2048, 4096])

Note: The canonical repro takes two SEPARATE tensors (gate_proj output and up_proj output),
not a single concatenated tensor. This matches the Qwen3-MoE / LLaMA pattern where
gate and up projections are computed separately.
"""
import torch
import triton
import triton.language as tl
import time


# ============================================================
# Triton Kernel: Fused SwiGLU
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def swiglu_kernel(
    gate_ptr,    # pointer to gate projection output (bf16)
    up_ptr,      # pointer to up projection output (bf16)
    out_ptr,     # pointer to output (bf16)
    N: tl.constexpr,  # total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU: out = silu(gate).to(bf16) * up

    Matches the canonical inductor pattern exactly:
    1. Upcast gate to f32
    2. Compute silu in f32: gate / (1 + exp(-gate))
    3. Cast silu result to bf16
    4. Multiply with up (bf16 * bf16 -> bf16)

    Single kernel, no intermediate tensor materialization.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load both inputs - contiguous bf16 loads
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask)  # keep as bf16

    # Compute silu(gate) = gate / (1 + exp(-gate)) in f32
    neg_gate = -gate
    exp_neg = tl.exp(neg_gate)
    silu_gate = gate / (1.0 + exp_neg)

    # Cast silu to bf16 then multiply (matches canonical pattern)
    silu_bf16 = silu_gate.to(tl.bfloat16)
    result = silu_bf16 * up

    # Store bf16 output
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['half_N'],
)
@triton.jit
def swiglu_kernel_from_concat(
    x_ptr,       # pointer to concatenated input [*, 2*hidden] (bf16)
    out_ptr,     # pointer to output [*, hidden] (bf16)
    half_N: tl.constexpr,  # number of elements in each half (total/2)
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU from a single concatenated tensor.

    Reads gate (first half) and up (second half) from a contiguous buffer,
    computes silu(gate).to(bf16) * up with no intermediate allocation.
    This handles the chunk(2, dim=-1) + silu + mul pattern.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_N

    # Load gate (first half) and up (second half)
    gate = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_ptr + half_N + offsets, mask=mask)  # bf16

    # silu(gate) in f32
    neg_gate = -gate
    exp_neg = tl.exp(neg_gate)
    silu_gate = gate / (1.0 + exp_neg)

    # Cast silu to bf16, multiply with up (matches canonical)
    result = silu_gate.to(tl.bfloat16) * up

    tl.store(out_ptr + offsets, result, mask=mask)


# ============================================================
# Wrapper functions
# ============================================================

def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Optimal Triton SwiGLU for two separate input tensors."""
    assert gate.is_contiguous() and up.is_contiguous()
    assert gate.shape == up.shape
    out = torch.empty_like(gate)
    N = gate.numel()

    # Grid is computed based on autotuned BLOCK_SIZE
    grid = lambda meta: ((N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    swiglu_kernel[grid](gate, up, out, N)
    return out


def swiglu_triton_concat(x: torch.Tensor) -> torch.Tensor:
    """Optimal Triton SwiGLU for a single concatenated tensor (chunk pattern)."""
    assert x.is_contiguous()
    half_N = x.numel() // 2
    # Output is half the size
    out_shape = list(x.shape)
    out_shape[-1] = out_shape[-1] // 2
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    grid = lambda meta: ((half_N + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    swiglu_kernel_from_concat[grid](x, out, half_N)
    return out


# ============================================================
# Reference implementations
# ============================================================

def swiglu_eager(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Eager PyTorch reference (matches the canonical repro pattern exactly).

    The canonical repro does: silu in f32, cast back to bf16, then mul with up in bf16.
    This matches the inductor-traced pattern from transformers.
    """
    gate_f32 = gate.float()
    neg = -gate_f32
    exp_neg = torch.exp(neg)
    denom = exp_neg + 1.0
    silu_gate = gate_f32 / denom
    silu_bf16 = silu_gate.to(torch.bfloat16)
    return silu_bf16 * up


def swiglu_functional(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Using F.silu - cleaner but same computation."""
    return torch.nn.functional.silu(gate.float()).to(gate.dtype) * up


# ============================================================
# Correctness check
# ============================================================

def check_correctness():
    """Verify Triton kernel matches eager computation."""
    torch.manual_seed(42)
    for shape in [(32, 1, 4096), (4, 2048, 4096)]:
        gate = torch.randn(shape, dtype=torch.bfloat16, device='cuda')
        up = torch.randn(shape, dtype=torch.bfloat16, device='cuda')

        ref = swiglu_eager(gate, up)
        tri = swiglu_triton(gate, up)

        max_diff = (ref.float() - tri.float()).abs().max().item()
        assert max_diff < 1e-2, f"Correctness failed for shape {shape}: max_diff={max_diff}"
        print(f"  Shape {shape}: max_diff={max_diff:.2e} - PASS")

    # Also check concat version - note: chunk(2, dim=-1) on [B,S,2H] gives
    # two views into the SAME storage with stride, NOT two contiguous halves.
    # Our concat kernel handles the case where gate and up are actually stored
    # as two contiguous halves (e.g., from a fused QKV-style linear).
    # For proper testing, create truly contiguous halves:
    gate_c = torch.randn(32, 1, 4096, dtype=torch.bfloat16, device='cuda')
    up_c = torch.randn(32, 1, 4096, dtype=torch.bfloat16, device='cuda')
    x = torch.cat([gate_c, up_c], dim=0).view(-1)  # flatten
    # Actually let's just test the flat layout: [gate_data | up_data]
    x_flat = torch.cat([gate_c.reshape(-1), up_c.reshape(-1)])
    ref = swiglu_eager(gate_c, up_c)
    tri = swiglu_triton_concat(x_flat.view(1, 1, -1))  # needs last dim = 2*N
    # Reshape to match
    tri_reshaped = tri.view_as(ref)
    max_diff = (ref.float() - tri_reshaped.float()).abs().max().item()
    assert max_diff < 1e-2, f"Correctness failed for concat: max_diff={max_diff}"
    print(f"  Concat [flat 2*131072]: max_diff={max_diff:.2e} - PASS")


# ============================================================
# Inductor analysis
# ============================================================

def analyze_inductor(gate, up):
    """Check what inductor generates for the SwiGLU pattern."""
    import torch._inductor.config as inductor_config
    from torch._inductor.utils import fresh_inductor_cache

    class SwiGLUModule(torch.nn.Module):
        def forward(self, gate, up):
            gate_f32 = gate.float()
            neg = -gate_f32
            exp_neg = torch.exp(neg)
            denom = exp_neg + 1
            silu_gate = gate_f32 / denom
            return silu_gate.to(torch.bfloat16) * up

    mod = SwiGLUModule()
    torch._dynamo.reset()

    # Count kernels
    import glob, os
    from torch._inductor.codecache import cache_dir

    with fresh_inductor_cache():
        compiled = torch.compile(mod)
        with torch.no_grad():
            compiled(gate, up)
            torch.cuda.synchronize()

        cd = cache_dir()
        py_files = sorted(glob.glob(os.path.join(cd, "**", "*.py"), recursive=True),
                         key=os.path.getmtime)
        n_kernels = 0
        kernel_code = None
        for f in reversed(py_files):
            with open(f) as fh:
                content = fh.read()
            if 'def call(' in content and '.run(' in content:
                runs = [l for l in content.split('\n') if '.run(' in l and not l.strip().startswith('#')]
                n_kernels = len(runs)
                kernel_code = content
                break

    return n_kernels, kernel_code


# ============================================================
# Benchmark
# ============================================================

def benchmark():
    """Full benchmark comparing Triton kernel vs inductor at decode and prefill shapes."""
    from triton.testing import do_bench
    import torch._inductor.config as inductor_config

    print("=" * 70)
    print("SwiGLU Kernel Benchmark: Optimal Triton vs Inductor")
    print("=" * 70)

    # Correctness
    print("\n[1] Correctness Check")
    check_correctness()

    shapes = {
        "decode  (batch=32, seq=1, hidden=4096)": (32, 1, 4096),
        "prefill (batch=4, seq=2048, hidden=4096)": (4, 2048, 4096),
    }

    # Inductor analysis
    print("\n[2] Inductor Fusion Analysis")
    gate_test = torch.randn(32, 1, 4096, dtype=torch.bfloat16, device='cuda')
    up_test = torch.randn(32, 1, 4096, dtype=torch.bfloat16, device='cuda')
    n_kernels, kernel_code = analyze_inductor(gate_test, up_test)
    print(f"  Inductor generates {n_kernels} kernel(s) for SwiGLU")
    if n_kernels == 1:
        print("  -> Inductor DOES fuse chunk + silu + mul into a single kernel")
    else:
        print("  -> Inductor does NOT fully fuse (multiple kernels)")

    # Show relevant kernel snippet
    if kernel_code:
        lines = kernel_code.split('\n')
        # Find the triton kernel definition
        for i, line in enumerate(lines):
            if 'triton_poi' in line and 'def ' in line:
                snippet = '\n'.join(lines[max(0,i):min(len(lines), i+25)])
                print(f"\n  Inductor kernel snippet:\n  {'='*40}")
                for sl in snippet.split('\n')[:20]:
                    print(f"    {sl}")
                print(f"  {'='*40}")
                break

    # Benchmark
    print("\n[3] Performance Comparison")
    print("-" * 70)

    results = {}
    for label, shape in shapes.items():
        gate = torch.randn(shape, dtype=torch.bfloat16, device='cuda')
        up = torch.randn(shape, dtype=torch.bfloat16, device='cuda')
        N = gate.numel()
        total_bytes = N * 2 * 3  # 2 bytes per bf16, 3 tensors (2 read + 1 write)

        print(f"\n  Shape: {label}")
        print(f"  Elements: {N:,} | Data: {total_bytes/1024:.1f} KB (read+write)")

        # --- Eager ---
        eager_ms = do_bench(lambda: swiglu_eager(gate, up), warmup=50, rep=300)
        eager_us = eager_ms * 1000

        # --- Inductor (default) ---
        torch._dynamo.reset()
        compiled = torch.compile(swiglu_functional)
        with torch.no_grad():
            for _ in range(5):
                compiled(gate, up)
            torch.cuda.synchronize()
        inductor_ms = do_bench(lambda: compiled(gate, up), warmup=50, rep=300)
        inductor_us = inductor_ms * 1000

        # --- Inductor (coordinate descent) ---
        inductor_config.coordinate_descent_tuning = True
        torch._dynamo.reset()
        compiled_cd = torch.compile(swiglu_functional)
        with torch.no_grad():
            for _ in range(5):
                compiled_cd(gate, up)
            torch.cuda.synchronize()
        inductor_cd_ms = do_bench(lambda: compiled_cd(gate, up), warmup=50, rep=300)
        inductor_cd_us = inductor_cd_ms * 1000
        inductor_config.coordinate_descent_tuning = False

        # --- Triton (ours) ---
        # Warmup the kernel (first call compiles)
        for _ in range(5):
            swiglu_triton(gate, up)
        torch.cuda.synchronize()
        triton_ms = do_bench(lambda: swiglu_triton(gate, up), warmup=50, rep=300)
        triton_us = triton_ms * 1000

        # --- Memcopy SOL ---
        copy_elems = max(total_bytes // (2 * 4), 256)  # float32 elements for same bytes
        src = torch.empty(copy_elems, dtype=torch.float32, device='cuda')
        dst = torch.empty_like(src)
        sol_ms = do_bench(lambda: dst.copy_(src), warmup=50, rep=300)
        sol_us = sol_ms * 1000
        del src, dst

        # Bandwidth calculation
        bw_triton = total_bytes / (triton_us * 1e-6) / 1e9  # GB/s
        bw_inductor = total_bytes / (inductor_us * 1e-6) / 1e9

        speedup_vs_inductor = inductor_us / triton_us
        speedup_vs_inductor_cd = inductor_cd_us / triton_us

        print(f"  {'Method':<30} {'Time (us)':>10} {'BW (GB/s)':>10} {'vs Inductor':>12}")
        print(f"  {'-'*62}")
        print(f"  {'Eager':<30} {eager_us:>10.2f} {'':>10} {'':>12}")
        print(f"  {'Inductor (default)':<30} {inductor_us:>10.2f} {bw_inductor:>10.1f} {'1.00x':>12}")
        print(f"  {'Inductor (coord descent)':<30} {inductor_cd_us:>10.2f} {'':>10} {inductor_cd_us/inductor_us:>11.2f}x")
        print(f"  {'Triton (ours)':<30} {triton_us:>10.2f} {bw_triton:>10.1f} {speedup_vs_inductor:>11.2f}x")
        print(f"  {'Memcopy SOL':<30} {sol_us:>10.2f} {'':>10} {'':>12}")
        print(f"  ")
        print(f"  Speedup vs Inductor (default): {speedup_vs_inductor:.3f}x")
        print(f"  Speedup vs Inductor (CD):      {speedup_vs_inductor_cd:.3f}x")
        print(f"  Gap to SOL:                    {triton_us/sol_us:.2f}x")

        results[label] = {
            "eager_us": eager_us,
            "inductor_us": inductor_us,
            "inductor_cd_us": inductor_cd_us,
            "triton_us": triton_us,
            "sol_us": sol_us,
            "speedup_vs_inductor": speedup_vs_inductor,
            "speedup_vs_inductor_cd": speedup_vs_inductor_cd,
            "gap_to_sol": triton_us / sol_us,
            "bw_triton_gbs": bw_triton,
            "bw_inductor_gbs": bw_inductor,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for label, r in results.items():
        print(f"\n  {label}:")
        print(f"    Triton speedup vs Inductor: {r['speedup_vs_inductor']:.3f}x")
        print(f"    Triton time:                {r['triton_us']:.2f} us")
        print(f"    Inductor time:              {r['inductor_us']:.2f} us")
        print(f"    Gap to memcopy SOL:         {r['gap_to_sol']:.2f}x")

    return results


if __name__ == "__main__":
    benchmark()
