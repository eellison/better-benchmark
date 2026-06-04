"""
Oracle kernel for amax_sum_87e1fb077f24: Longformer online softmax (persistent, small rnumel).

Pattern: f32[8, 1024, 12, 513] -> amax(dim=-1) -> sub -> exp -> sum(dim=-1) -> div
         then where(mask, 0, softmax) -> dropout -> permute -> reshape -> transpose
This is hf_AllenaiLongformerBase_train_002 with sliding window attention.

Key insight: rnumel=513 is SMALL. The entire reduction fits in a single tile
(no loop needed). This is a "persistent reduction" case where each program
loads one complete row into registers, computes softmax in a single shot,
and writes the normalized output.

The full repro includes:
  1. Complex pre-softmax attention assembly (view/permute/slice/scatter/pad)
     converting [288, 512, 512] bmm output into [8, 1024, 12, 513] attention scores
  2. Softmax: amax -> sub -> exp -> sum -> div on last dim (513)
  3. Masked where (zero out padding positions)
  4. Dropout (threshold ~1e-30, effectively a no-op mask)
  5. Post-softmax permute + pad + reshape + transpose to [384, 768, 256]

This oracle measures only the softmax+dropout reduction kernel floor.

Strategy:
  - Persistent single-tile softmax: BLOCK_N=1024 (next power-of-2 >= 513)
  - Fuse softmax + conditional mask + dropout into one kernel
  - One program per row; 98304 total rows (8 * 1024 * 12)
  - Input is already f32, no dtype conversion needed

Memory traffic (softmax+dropout fused kernel only):
  - Read: 98304 * 513 * 4 bytes = ~192 MB (f32 input)
  - Write: 98304 * 513 * 4 bytes = ~192 MB (f32 output)
  - Total: ~384 MB
  - SOL at 3.35 TB/s (H100 HBM): ~115 us

Note: The full repro includes extensive pre-softmax attention pattern assembly
(views/permutes/slices/scatters/pads) that dominates total runtime. This oracle
measures only the reduction kernel floor.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "amax_sum_87e1fb077f24"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

# Problem dimensions for the softmax reduction
# Input to softmax: [8, 1024, 12, 513] -> flatten first 3 dims = 98304 rows
BATCH = 8
SEQ_LEN = 1024
N_HEADS = 12
RNUMEL = 513  # reduction dimension (small -- fits in one tile)
M = BATCH * SEQ_LEN * N_HEADS  # 98304 rows


# --- Triton Kernel: Persistent Softmax + Dropout (small rnumel) ---

@triton.jit
def persistent_softmax_dropout_kernel(
    input_ptr,
    output_ptr,
    mask_ptr,
    seed,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DROPOUT_P: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Fused softmax + optional mask + dropout for rnumel=513 (persistent).

    Since N=513 fits entirely in one tile (BLOCK_N=1024), we load the full row
    into registers, compute softmax without any loop, apply mask and dropout.
    Zero intermediate memory traffic.
    """
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_start = row_idx * N
    cols = tl.arange(0, BLOCK_N)
    valid_mask = cols < N

    # Load entire row (single tile -- no loop needed)
    x = tl.load(input_ptr + row_start + cols, mask=valid_mask, other=float("-inf"))

    # Softmax: amax -> sub -> exp -> sum -> div (all in registers)
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    # Optional: apply padding mask (where(unsqueeze_11, 0, softmax))
    if HAS_MASK:
        padding_mask = tl.load(mask_ptr + row_idx).to(tl.int1)
        # If mask is True, output is 0 (masking out padded rows)
        zero = tl.zeros([BLOCK_N], dtype=tl.float32)
        softmax_out = tl.where(padding_mask, zero, softmax_out)

    # Fused dropout: generate random mask and apply
    # In the repro, dropout threshold is 1e-30 (effectively keeping everything)
    rng_offsets = row_idx * BLOCK_N + cols
    random_vals = tl.rand(seed, rng_offsets)
    dropout_mask = random_vals > DROPOUT_P
    result = softmax_out * dropout_mask.to(tl.float32)

    tl.store(output_ptr + row_start + cols, result, mask=valid_mask)


@triton.jit
def persistent_softmax_kernel(
    input_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Pure softmax kernel (no dropout/mask) for rnumel=513 (persistent).

    Single-tile persistent reduction: the entire row fits in registers.
    """
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_start = row_idx * N
    cols = tl.arange(0, BLOCK_N)
    valid_mask = cols < N

    # Load entire row
    x = tl.load(input_ptr + row_start + cols, mask=valid_mask, other=float("-inf"))

    # Softmax in registers (no loops, no intermediate writes)
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    tl.store(output_ptr + row_start + cols, softmax_out, mask=valid_mask)


def oracle_softmax(x: torch.Tensor, with_dropout: bool = False) -> torch.Tensor:
    """Launch the persistent softmax Triton kernel.

    Args:
        x: f32 tensor of shape [M, N] where N=513
        with_dropout: whether to apply fused dropout
    """
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"
    M_val, N_val = x.shape
    assert N_val == RNUMEL, f"Expected N={RNUMEL}, got {N_val}"
    assert x.dtype == torch.float32

    output = torch.empty_like(x)

    # BLOCK_N must be >= N and power of 2
    BLOCK_N = 1024  # next power of 2 above 513

    grid = (M_val,)

    if with_dropout:
        seed = torch.randint(0, 2**31, (1,), device=x.device, dtype=torch.int64).item()
        # No row-level mask for basic benchmark
        persistent_softmax_dropout_kernel[grid](
            x, output, x,  # mask_ptr unused when HAS_MASK=False
            seed,
            M_val,
            N=N_val,
            BLOCK_N=BLOCK_N,
            DROPOUT_P=1e-30,
            HAS_MASK=False,
        )
    else:
        persistent_softmax_kernel[grid](
            x, output,
            M_val,
            N=N_val,
            BLOCK_N=BLOCK_N,
        )
    return output


# --- Reference implementation ---

def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference softmax matching the repro pattern (f32 in, f32 out)."""
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    s = e.sum(dim=-1, keepdim=True)
    return e / s


# --- Benchmarking ---

def benchmark_cuda_graph(fn, warmup=25, rep=100):
    """Benchmark a function using CUDA graphs for minimal launch overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(rep):
        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    median_ms = times[len(times) // 2]
    return median_ms * 1000.0  # convert to us


def benchmark_triton_softmax(x: torch.Tensor, warmup=25, rep=100, with_dropout=False):
    """Benchmark the Triton persistent softmax kernel."""
    output = torch.empty_like(x)
    BLOCK_N = 1024
    grid = (x.shape[0],)

    if with_dropout:
        seed = 42

        def run():
            persistent_softmax_dropout_kernel[grid](
                x, output, x,  # mask_ptr unused
                seed,
                x.shape[0],
                N=x.shape[1],
                BLOCK_N=BLOCK_N,
                DROPOUT_P=1e-30,
                HAS_MASK=False,
            )
    else:
        def run():
            persistent_softmax_kernel[grid](
                x, output,
                x.shape[0],
                N=x.shape[1],
                BLOCK_N=BLOCK_N,
            )

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def benchmark_compiled_full(warmup=25, rep=100):
    """Benchmark the full repro with torch.compile + CUDAGraph."""
    from repro_harness import parse_shapes_config, make_inputs_safely

    repro_mod = _load_repro_module()
    model = repro_mod.Repro()
    inputs = repro_mod._default_make_inputs()

    import torch._inductor.config as inductor_config
    inductor_config.coordinate_descent_tuning = True

    torch._dynamo.reset()
    compiled = torch.compile(model)

    with torch.no_grad():
        for _ in range(warmup):
            compiled(*inputs)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(*inputs)
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(rep):
        start_event.record()
        graph.replay()
        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    times.sort()
    median_ms = times[len(times) // 2]
    return median_ms * 1000.0


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    module.device = torch.device
    module.inf = math.inf
    module.nan = math.nan
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def correctness_check():
    """Verify the Triton kernel matches the reference implementation."""
    torch.manual_seed(42)
    # Create input matching the softmax portion: f32[98304, 513]
    x = torch.randn(M, RNUMEL, dtype=torch.float32, device="cuda")

    ref = softmax_reference(x)
    out = oracle_softmax(x, with_dropout=False)

    max_diff = (ref - out).abs().max().item()
    mean_diff = (ref - out).abs().mean().item()
    allclose = torch.allclose(ref, out, atol=1e-5, rtol=1e-5)

    print(f"Correctness check (softmax only, f32):")
    print(f"  Shape: [{M}, {RNUMEL}]")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  torch.allclose (atol=1e-5, rtol=1e-5): {allclose}")

    if not allclose:
        print("  WARNING: Output does not match reference within tolerance!")
        for row in [0, 100, 1000, 50000]:
            if row < M:
                row_diff = (ref[row] - out[row]).abs().max().item()
                print(f"  Row {row} max diff: {row_diff:.6e}")
    return allclose


def main():
    parser = argparse.ArgumentParser(description="Oracle persistent softmax benchmark")
    parser.add_argument("--check-only", action="store_true", help="Only run correctness check")
    parser.add_argument("--rep", type=int, default=100, help="Number of benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--csv", type=str, default=None, help="Append results to CSV file")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile baseline")
    parser.add_argument("--with-dropout", action="store_true", help="Include fused dropout")
    args = parser.parse_args()

    print(f"=" * 70)
    print(f"Oracle Persistent Softmax Benchmark: {REPRO_ID}")
    print(f"Longformer sliding window attention (online softmax cross-entropy)")
    print(f"Softmax shape: f32[{M}, {RNUMEL}] ({M} rows, {RNUMEL} cols)")
    print(f"=" * 70)

    # Compute memory metrics for the softmax kernel alone
    total_bytes = M * RNUMEL * 4 * 2  # read + write, f32 = 4 bytes
    sol_us = total_bytes / 3.35e6  # H100 HBM ~3.35 TB/s
    print(f"\nMemory (softmax kernel only): {total_bytes / 1e6:.1f} MB total traffic")
    print(f"SOL (3.35 TB/s): {sol_us:.1f} us")
    print(f"Note: Full repro includes extensive pre-softmax assembly (views/slices/scatters/pads)")

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
    x = torch.randn(M, RNUMEL, dtype=torch.float32, device="cuda")

    # Triton oracle (softmax only)
    triton_us = benchmark_triton_softmax(x, warmup=args.warmup, rep=args.rep,
                                         with_dropout=args.with_dropout)
    triton_bw = total_bytes / (triton_us * 1e-6) / 1e12
    triton_pct_sol = sol_us / triton_us * 100.0
    label = "softmax+dropout" if args.with_dropout else "softmax"
    print(f"\nTriton persistent {label} (rnumel={RNUMEL}):")
    print(f"  Median: {triton_us:.1f} us")
    print(f"  Effective BW: {triton_bw:.3f} TB/s")
    print(f"  % of SOL: {triton_pct_sol:.1f}%")

    # With dropout fused
    if not args.with_dropout:
        triton_dropout_us = benchmark_triton_softmax(x, warmup=args.warmup, rep=args.rep,
                                                     with_dropout=True)
        print(f"\nTriton persistent softmax+dropout:")
        print(f"  Median: {triton_dropout_us:.1f} us")
        dropout_bw = total_bytes / (triton_dropout_us * 1e-6) / 1e12
        print(f"  Effective BW: {dropout_bw:.3f} TB/s")

    # torch.compile full repro baseline
    compile_us = None
    if not args.no_compile:
        print(f"\nRunning torch.compile full repro baseline (with CD tuning)...")
        try:
            compile_us = benchmark_compiled_full(warmup=args.warmup, rep=args.rep)
            print(f"torch.compile (full repro with CD):")
            print(f"  Median: {compile_us:.1f} us")
            # Ratio: how much of compiled time is our softmax kernel
            ratio = triton_us / compile_us * 100.0
            print(f"  Oracle softmax kernel = {ratio:.1f}% of full compiled time")
        except Exception as e:
            print(f"  torch.compile baseline failed: {e}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Oracle (persistent softmax, BLOCK_N=1024): {triton_us:.1f} us")
    print(f"  SOL: {sol_us:.1f} us")
    print(f"  Oracle achieves {triton_pct_sol:.1f}% of SOL")
    if compile_us:
        print(f"  Full compiled repro: {compile_us:.1f} us")
        print(f"  Softmax kernel is {triton_us/compile_us*100:.1f}% of total")
    print(f"\n  Key observation: rnumel=513 fits entirely in one tile (BLOCK_N=1024)")
    print(f"  No loop iteration needed - true persistent single-pass reduction")

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
        "kernel_type": "triton_persistent_softmax",
        "shape": f"f32[{M},{RNUMEL}]",
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
