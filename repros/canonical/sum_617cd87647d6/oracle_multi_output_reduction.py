"""
Oracle kernel for sum_617cd87647d6 (Longformer sliding-window attention backward).

Pattern: A single channel-wise reduction (sum over dim [-1]) on an
f32[8, 1024, 12, 513] tensor, followed by a fused-multiply-add (fma)
to compute the softmax backward gradient:

    sum_i = sum_{k} (mul_tensor_2[..., k])  # keepdim
    out[..., k] = fma(-arg221, sum_i, mul_tensor_2[..., k])

The full repro also includes complex sliding-window attention assembly
(permutes, slice_scatters, padding, etc.) that produces the input to this
reduction, and post-reduction diagonal scatter that produces the final output.
The oracle focuses on the reduction + fma portion, which is the compute-critical
section that determines whether Inductor's multi-output reduction template
can fuse properly.

Oracle strategy:
  - Each program handles one row of the flattened [8*1024*12, 513] matrix.
  - Since rnumel=513 is small, we use a persistent kernel that loads the
    entire row in one shot, computes the sum, then writes the fma result.
  - This avoids the two-kernel overhead (one for reduction, one for pointwise).

Memory traffic (reduction + fma only):
  - Read: 8*1024*12*513 * 4 bytes (mul_tensor_2) + 8*1024*12*513 * 4 bytes (arg221)
  - Write: 8*1024*12*513 * 4 bytes (fma_result)
  - Total: ~603 MB
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPRO_ID = "sum_617cd87647d6"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

# Problem dimensions for the reduction kernel
# The softmax backward operates on [8, 1024, 12, 513]
# Flattened as M=8*1024*12=98304 rows, N=513 columns
M = 8 * 1024 * 12  # 98304
N = 513


# --- Triton Kernel: Fused reduction + fma (softmax backward) ---

@triton.jit
def fused_softmax_backward_kernel(
    mul_tensor_2_ptr,  # input: softmax_output * upstream_grad (masked)
    arg221_ptr,        # input: softmax probabilities
    output_ptr,        # output: fma result
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused sum reduction + fma for softmax backward.

    For each row:
        s = sum(mul_tensor_2[row, :])
        out[row, k] = mul_tensor_2[row, k] - arg221[row, k] * s
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Load entire row (N=513 fits in one block with BLOCK_N=512 or 1024)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load mul_tensor_2 row
    mt2 = tl.load(mul_tensor_2_ptr + row_start + cols, mask=mask, other=0.0)

    # Compute sum reduction
    s = tl.sum(mt2, axis=0)

    # Load arg221 (softmax probs)
    probs = tl.load(arg221_ptr + row_start + cols, mask=mask, other=0.0)

    # fma: out = mul_tensor_2 - probs * sum
    out = mt2 - probs * s

    # Store result
    tl.store(output_ptr + row_start + cols, out, mask=mask)


def oracle_softmax_backward(mul_tensor_2: torch.Tensor, arg221: torch.Tensor) -> torch.Tensor:
    """
    Compute fused softmax backward: out = mt2 - probs * sum(mt2, dim=-1, keepdim=True)

    Args:
        mul_tensor_2: [8, 1024, 12, 513] - masked gradient * probs
        arg221: [8, 1024, 12, 513] - softmax probabilities

    Returns:
        fma_result: [8, 1024, 12, 513]
    """
    assert mul_tensor_2.shape == arg221.shape
    original_shape = mul_tensor_2.shape
    M_val = mul_tensor_2.numel() // original_shape[-1]
    N_val = original_shape[-1]

    # Flatten to 2D for the kernel
    mt2_flat = mul_tensor_2.reshape(M_val, N_val)
    probs_flat = arg221.reshape(M_val, N_val)
    out_flat = torch.empty_like(mt2_flat)

    # BLOCK_N must be power of 2 >= N
    BLOCK_N = triton.next_power_of_2(N_val)

    grid = (M_val,)
    fused_softmax_backward_kernel[grid](
        mt2_flat, probs_flat, out_flat,
        N=N_val,
        BLOCK_N=BLOCK_N,
    )

    return out_flat.reshape(original_shape)


# --- Reference implementation ---

def softmax_backward_reference(mul_tensor_2: torch.Tensor, arg221: torch.Tensor) -> torch.Tensor:
    """Reference: sum + fma matching the repro pattern."""
    s = mul_tensor_2.sum(dim=-1, keepdim=True)
    return mul_tensor_2 - arg221 * s


# --- Load repro module ---

def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prepare_oracle_inputs():
    """
    Run the full repro to the point where the reduction happens, then
    extract the inputs to the softmax backward kernel.

    The key reduction in this repro is:
        sum_dim_int_list: sum(mul_tensor_2, [-1], keepdim=True)
    where mul_tensor_2 is f32[8, 1024, 12, 513].

    We generate synthetic inputs matching these shapes.
    """
    torch.manual_seed(42)
    # mul_tensor_2 shape = [8, 1024, 12, 513]
    # This is where_self * arg221_1 after masking
    mul_tensor_2 = torch.randn(8, 1024, 12, 513, device="cuda", dtype=torch.float32)
    # arg221 = softmax probs (non-negative, sums to ~1 along last dim)
    arg221 = torch.softmax(torch.randn(8, 1024, 12, 513, device="cuda", dtype=torch.float32), dim=-1)

    return mul_tensor_2, arg221


# --- Correctness check ---

def run_check():
    """Verify oracle produces same results as reference."""
    print(f"Correctness check for {REPRO_ID}")
    print(f"Shape: f32[8, 1024, 12, 513], reduction dim=-1")

    mul_tensor_2, arg221 = prepare_oracle_inputs()

    with torch.no_grad():
        ref = softmax_backward_reference(mul_tensor_2, arg221)
        out = oracle_softmax_backward(mul_tensor_2, arg221)

    max_diff = (ref - out).abs().max().item()
    mean_diff = (ref - out).abs().mean().item()
    allclose = torch.allclose(ref, out, rtol=1e-4, atol=1e-4)

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  torch.allclose (rtol=1e-4, atol=1e-4): {allclose}")

    if allclose:
        print("PASSED")
    else:
        print("FAILED")

    return allclose


# --- Benchmark ---

def run_bench(rep=100, warmup=25):
    """Benchmark oracle kernel vs torch.compile on the full repro."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {REPRO_ID} (multi_output_reduction_templates)")
    print(f"Oracle: fused softmax backward (sum + fma)")
    print(f"Shape: f32[8, 1024, 12, 513], M={M}, N={N}")
    print(f"{'='*60}")

    mul_tensor_2, arg221 = prepare_oracle_inputs()

    # Memory metrics for the reduction+fma portion
    # Read: mul_tensor_2 + arg221, Write: output
    total_bytes = 3 * 8 * 1024 * 12 * 513 * 4  # 3 tensors x shape x f32
    print(f"\nMemory traffic (reduction+fma): {total_bytes / 1e6:.1f} MB")

    # Benchmark oracle
    with torch.no_grad():
        _ = oracle_softmax_backward(mul_tensor_2, arg221)
    torch.cuda.synchronize()

    oracle_us = triton.testing.do_bench(
        lambda: oracle_softmax_backward(mul_tensor_2, arg221),
        warmup=warmup * 20,
        rep=rep * 10,
    ) * 1000  # ms -> us

    oracle_bw = total_bytes / (oracle_us * 1e-6) / 1e12
    print(f"\nOracle (fused softmax backward):")
    print(f"  Median: {oracle_us:.1f} us")
    print(f"  Effective BW: {oracle_bw:.3f} TB/s")

    # Benchmark torch.compile on full repro
    repro_mod = _load_repro_module()
    model = repro_mod.Repro().cuda()
    inputs = repro_mod.make_inputs()
    inputs = tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    with torch.no_grad():
        compiled = torch.compile(model)
        _ = compiled(*inputs)
        torch.cuda.synchronize()

    compile_us = triton.testing.do_bench(
        lambda: compiled(*inputs),
        warmup=warmup * 20,
        rep=rep * 10,
    ) * 1000  # ms -> us

    print(f"\ntorch.compile (full repro):")
    print(f"  Median: {compile_us:.1f} us")

    print(f"\nSpeedup (compile / oracle): {compile_us / oracle_us:.2f}x")
    print(f"\nNote: Oracle measures only the fused reduction+fma kernel.")
    print(f"The full repro includes sliding-window assembly + diagonal scatter.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--rep", type=int, default=100, help="Benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check:
        ok = run_check()
        if not ok:
            sys.exit(1)

    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup)


if __name__ == "__main__":
    with torch.no_grad():
        main()
