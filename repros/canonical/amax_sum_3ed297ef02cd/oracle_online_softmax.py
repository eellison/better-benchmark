"""
Oracle kernel for amax_sum_3ed297ef02cd: online softmax forward.

Pattern: bf16[8192, 262144] -> cast f32 -> amax -> sub -> exp -> sum -> div -> cast bf16
This is the Qwen3-0.6B softmax forward with large vocabulary (262144).

Strategy:
  - Online softmax (Milakov & Gimelshein 2018): single pass over the reduction
    dimension computing running max and sum_exp simultaneously, then a second
    pass to normalize and write output.
  - Each program instance handles one row (or a few rows via ROWS_PER_PROGRAM).
  - Use scalar accumulators for max and sum_exp within each tile iteration.
  - BLOCK_SIZE chosen to balance register pressure vs loop iterations.

Memory traffic:
  - Read: 8192 * 262144 * 2 bytes (bf16 input)
  - Write: 8192 * 262144 * 2 bytes (bf16 output)
  - Total: 8,589,934,592 bytes (~8 GB)
  - SOL at 3.35 TB/s (H100 HBM): ~2564 us
  - SOL at 3.8e6 us/byte formula: total_bytes / 3.8e6 = 2261 us
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_3ed297ef02cd"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

# Problem dimensions
M = 8192       # number of rows
N = 262144     # reduction dimension (vocab size)


# --- Triton Kernel ---

@triton.jit
def online_softmax_fwd_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Online softmax forward: fused max + sum_exp + normalize.

    Each program handles one row of the input matrix.
    Pass 1: compute running max and sum_exp (online algorithm)
    Pass 2: normalize and write output
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Pass 1: Online computation of max and sum_exp
    m_i = float("-inf")  # running max
    l_i = 0.0           # running sum of exp(x - m)

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        # Load block of input, cast to f32
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        # Rescale previous sum and add new contributions
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    # Pass 2: Normalize and write output
    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
        # softmax(x) = exp(x - max) / sum_exp
        out = tl.exp(x - m_i) / l_i
        # Cast back to bf16
        tl.store(output_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


def online_softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Launch the online softmax Triton kernel."""
    assert x.ndim == 2
    M_val, N_val = x.shape
    assert x.dtype == torch.bfloat16

    output = torch.empty_like(x)

    # Choose BLOCK_N: larger blocks mean fewer loop iterations but more registers.
    # For N=262144, BLOCK_N=8192 gives 32 iterations per pass.
    BLOCK_N = 8192

    grid = (M_val,)
    online_softmax_fwd_kernel[grid](
        x, output,
        N=N_val,
        BLOCK_N=BLOCK_N,
    )
    return output


# --- Reference implementation ---

def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference softmax matching the repro pattern exactly."""
    x_f32 = x.to(torch.float32)
    m = x_f32.max(dim=-1, keepdim=True).values
    e = torch.exp(x_f32 - m)
    s = e.sum(dim=-1, keepdim=True)
    return (e / s).to(torch.bfloat16)


# --- Benchmarking ---

def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def benchmark_cuda_graph(fn, warmup=25, rep=100):
    """Benchmark a function using CUDA graphs for minimal launch overhead."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Capture CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    # Time with events
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(rep):
        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    # Return median in microseconds
    times.sort()
    median_ms = times[len(times) // 2]
    return median_ms * 1000.0  # convert to us


def benchmark_triton_softmax(x: torch.Tensor, warmup=25, rep=100):
    """Benchmark the Triton online softmax kernel."""
    output = torch.empty_like(x)
    BLOCK_N = 8192
    grid = (x.shape[0],)

    def run():
        online_softmax_fwd_kernel[grid](
            x, output,
            N=x.shape[1],
            BLOCK_N=BLOCK_N,
        )

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def benchmark_compiled(x: torch.Tensor, warmup=25, rep=100):
    """Benchmark torch.compile baseline."""
    repro_mod = _load_repro_module()
    model = repro_mod.Repro().cuda()

    compiled = torch.compile(model, fullgraph=True)

    # Warmup
    for _ in range(warmup):
        compiled(x)
    torch.cuda.synchronize()

    def run():
        compiled(x)

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def correctness_check():
    """Verify the Triton kernel matches the reference implementation."""
    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    ref = softmax_reference(x)
    out = online_softmax_triton(x)

    # Check correctness - bf16 allows some tolerance
    max_diff = (ref.float() - out.float()).abs().max().item()
    mean_diff = (ref.float() - out.float()).abs().mean().item()
    allclose = torch.allclose(ref.float(), out.float(), atol=1e-2, rtol=1e-2)

    print(f"Correctness check:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  torch.allclose (atol=1e-2, rtol=1e-2): {allclose}")

    if not allclose:
        print("  WARNING: Output does not match reference within tolerance!")
        # Check a few specific rows
        for row in [0, 100, 1000, 8000]:
            if row < M:
                row_diff = (ref[row].float() - out[row].float()).abs().max().item()
                print(f"  Row {row} max diff: {row_diff:.6e}")
    return allclose


def main():
    parser = argparse.ArgumentParser(description="Oracle online softmax benchmark")
    parser.add_argument("--check-only", action="store_true", help="Only run correctness check")
    parser.add_argument("--rep", type=int, default=100, help="Number of benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--csv", type=str, default=None, help="Append results to CSV file")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile baseline")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"Oracle Online Softmax Benchmark: {REPRO_ID}")
    print(f"Shape: bf16[{M}, {N}]")
    print(f"=" * 70)

    # Compute memory metrics
    total_bytes = M * N * 2 * 2  # read + write, bf16 = 2 bytes
    sol_us = total_bytes / 3.35e6  # H100 HBM ~3.35 TB/s = 3.35e6 MB/s = 3.35e12 B/s
    print(f"\nMemory: {total_bytes / 1e9:.3f} GB total traffic")
    print(f"SOL (3.35 TB/s): {sol_us:.1f} us")

    # Correctness
    print(f"\n--- Correctness ---")
    ok = correctness_check()
    if not ok:
        print("FAILED correctness check. Exiting.")
        sys.exit(1)
    print("PASSED")

    if args.check_only:
        return

    # Benchmark
    print(f"\n--- Benchmark (rep={args.rep}, warmup={args.warmup}) ---")
    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # Triton oracle
    triton_us = benchmark_triton_softmax(x, warmup=args.warmup, rep=args.rep)
    triton_bw = total_bytes / (triton_us * 1e-6) / 1e12  # TB/s
    triton_pct_sol = sol_us / triton_us * 100.0
    print(f"\nTriton online softmax:")
    print(f"  Median: {triton_us:.1f} us")
    print(f"  Effective BW: {triton_bw:.3f} TB/s")
    print(f"  % of SOL: {triton_pct_sol:.1f}%")

    # torch.compile baseline
    compile_us = None
    if not args.no_compile:
        print(f"\nRunning torch.compile baseline...")
        compile_us = benchmark_compiled(x, warmup=args.warmup, rep=args.rep)
        compile_bw = total_bytes / (compile_us * 1e-6) / 1e12
        compile_pct_sol = sol_us / compile_us * 100.0
        print(f"torch.compile:")
        print(f"  Median: {compile_us:.1f} us")
        print(f"  Effective BW: {compile_bw:.3f} TB/s")
        print(f"  % of SOL: {compile_pct_sol:.1f}%")
        speedup = compile_us / triton_us
        print(f"\nSpeedup (compile / triton): {speedup:.2f}x")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Oracle (Triton online softmax): {triton_us:.1f} us")
    if compile_us:
        print(f"  Baseline (torch.compile): {compile_us:.1f} us")
    print(f"  SOL: {sol_us:.1f} us")
    print(f"  Oracle achieves {triton_pct_sol:.1f}% of SOL")

    # Write CSV if requested
    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    if args.csv is not None or DEFAULT_CSV.exists():
        _write_csv(csv_path, triton_us, compile_us, total_bytes)


def _write_csv(csv_path: Path, triton_us: float, compile_us: float | None, total_bytes: int):
    """Append measurement to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repro_id": REPRO_ID,
        "kernel_type": "triton_online_softmax",
        "shape": f"bf16[{M},{N}]",
        "oracle_us": f"{triton_us:.1f}",
        "compile_us": f"{compile_us:.1f}" if compile_us else "",
        "total_bytes": str(total_bytes),
        "effective_bw_tb_s": f"{total_bytes / (triton_us * 1e-6) / 1e12:.3f}",
        "sol_us": f"{total_bytes / 3.35e6:.1f}",
        "pct_sol": f"{(total_bytes / 3.35e6) / triton_us * 100:.1f}",
    }

    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nResults appended to {csv_path}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
