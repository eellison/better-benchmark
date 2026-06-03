"""
Gap diagnosis (classification: NEW_PATTERN): this oracle lowers the two T5
attention softmax+dropout reductions as persistent row kernels, with a dual
launch variant that processes both same-shaped attention passes without
materializing amax, exp, sum, or div intermediates. Inductor currently sees the
two decomposed amax/sub/exp/sum/div/dropout/permute paths as generic reduction
and RNG pointwise/layout work around the relative-position-bias producers, so it
does not form a persistent online-softmax template or co-schedule the sibling
softmax passes. The fix is a NEW_PATTERN lowering for T5-style last-dimension
attention softmax with dropout that emits persistent row softmax kernels and can
use a paired dual-softmax launch when sibling reductions have the same shape and
epilogue.

Oracle kernel for amax_sum_amax_2a81770def44: T5 dual online softmax (persistent).

Pattern: f32[8, 12, 1024, 1024] -> amax(dim=-1) -> sub -> exp -> sum(dim=-1) -> div
         -> dropout -> permute/reshape (done TWICE for encoder + decoder attention)

This is hf_T5_base_train with T5-style relative position bias attention.
The kernel performs two independent softmax+dropout passes over the last dimension.

Key insight: rnumel=1024 fits perfectly in a single tile (BLOCK_N=1024).
This is a persistent reduction -- each program loads one complete row into
registers, computes softmax in a single shot, applies dropout, and writes output.
No loop is needed.

Dimensions:
  - Input per softmax: [8, 12, 1024, 1024] = 96*1024 = 98304 rows x 1024 cols
  - Two softmax passes -> 196608 total rows processed
  - Outputs: two f32[96, 1024, 1024] tensors (permuted)

Memory traffic (dual softmax+dropout fused):
  - Read:  2 * 98304 * 1024 * 4 bytes = ~768 MB (f32 input after bias add)
  - Write: 2 * 98304 * 1024 * 4 bytes = ~768 MB (f32 softmax+dropout output)
  - Total: ~1536 MB (1.5 GB)
  - SOL at 3.35 TB/s (H100 HBM): ~458 us

Note: The full repro includes relative position bias computation (embedding
lookups, log-binning, etc.) before the softmax. This oracle only measures
the reduction kernel floor (the softmax+dropout portion).
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


REPRO_ID = "amax_sum_amax_2a81770def44"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

# Problem dimensions
# Two softmax passes, each on [8, 12, 1024, 1024] reshaped as [98304, 1024]
BATCH = 8
N_HEADS = 12
SEQ_LEN = 1024
RNUMEL = 1024  # reduction dimension -- fits in one tile!
M = BATCH * N_HEADS * SEQ_LEN  # 98304 rows per softmax pass
DROPOUT_P = 0.1  # dropout probability from the repro


# --- Triton Kernel: Persistent Softmax + Dropout (single tile, no loop) ---

@triton.jit
def persistent_softmax_dropout_kernel(
    input_ptr,
    output_ptr,
    seed,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DROPOUT_P: tl.constexpr,
    DROPOUT_SCALE: tl.constexpr,
):
    """Fused softmax + dropout for rnumel=1024 (persistent, single tile).

    Since N=1024 = BLOCK_N, we load the full row into registers, compute
    softmax without any loop, apply dropout, and write back. Zero intermediate
    memory traffic.
    """
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_start = row_idx * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load entire row (single tile -- no loop needed)
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf"))

    # Softmax: amax -> sub -> exp -> sum -> div (all in registers)
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    # Fused dropout: generate mask, apply scale
    rng_offsets = row_idx * BLOCK_N + cols
    random_vals = tl.rand(seed, rng_offsets)
    dropout_mask = random_vals > DROPOUT_P
    result = softmax_out * dropout_mask.to(tl.float32) * DROPOUT_SCALE

    tl.store(output_ptr + row_start + cols, result, mask=mask)


@triton.jit
def persistent_softmax_kernel(
    input_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Pure softmax kernel (no dropout) for rnumel=1024 (persistent).

    Single-tile persistent reduction: the entire row fits in registers.
    """
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_start = row_idx * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load entire row
    x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf"))

    # Softmax in registers (no loops, no intermediate writes)
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    tl.store(output_ptr + row_start + cols, softmax_out, mask=mask)


@triton.jit
def dual_persistent_softmax_dropout_kernel(
    input1_ptr,
    output1_ptr,
    input2_ptr,
    output2_ptr,
    seed1,
    seed2,
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DROPOUT_P: tl.constexpr,
    DROPOUT_SCALE: tl.constexpr,
):
    """Dual fused softmax + dropout: processes both attention heads in one launch.

    Each program handles one row. Programs 0..M-1 handle the first softmax,
    programs M..2M-1 handle the second. This reduces kernel launch overhead.
    """
    prog_id = tl.program_id(0)
    is_second = prog_id >= M
    row_idx = prog_id - M * is_second.to(tl.int32)

    # Select input/output pointers based on which softmax pass
    inp_ptr = tl.where(is_second, input2_ptr, input1_ptr)
    out_ptr = tl.where(is_second, output2_ptr, output1_ptr)
    seed = tl.where(is_second, seed2, seed1)

    row_start = row_idx * N
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # Load entire row
    x = tl.load(inp_ptr + row_start + cols, mask=mask, other=float("-inf"))

    # Softmax in registers
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_out = exp_x / sum_exp

    # Fused dropout
    rng_offsets = row_idx * BLOCK_N + cols
    random_vals = tl.rand(seed, rng_offsets)
    dropout_mask = random_vals > DROPOUT_P
    result = softmax_out * dropout_mask.to(tl.float32) * DROPOUT_SCALE

    tl.store(out_ptr + row_start + cols, result, mask=mask)


def oracle_softmax(x: torch.Tensor, with_dropout: bool = True) -> torch.Tensor:
    """Launch the persistent softmax Triton kernel on a single [M, N] input.

    Args:
        x: f32 tensor of shape [M, 1024]
        with_dropout: whether to apply fused dropout
    """
    assert x.ndim == 2
    M_val, N_val = x.shape
    assert N_val == RNUMEL, f"Expected N={RNUMEL}, got {N_val}"
    assert x.dtype == torch.float32

    output = torch.empty_like(x)
    BLOCK_N = 1024  # exact fit for rnumel=1024

    grid = (M_val,)

    if with_dropout:
        seed = torch.randint(0, 2**31, (1,), device=x.device, dtype=torch.int64).item()
        persistent_softmax_dropout_kernel[grid](
            x, output, seed,
            M_val,
            N=N_val,
            BLOCK_N=BLOCK_N,
            DROPOUT_P=DROPOUT_P,
            DROPOUT_SCALE=1.0 / (1.0 - DROPOUT_P),  # 1.1111...
        )
    else:
        persistent_softmax_kernel[grid](
            x, output,
            M_val,
            N=N_val,
            BLOCK_N=BLOCK_N,
        )
    return output


def oracle_dual_softmax(
    x1: torch.Tensor, x2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the dual fused softmax+dropout kernel for both attention passes.

    This models the actual workload: two independent softmax+dropout passes
    executed in a single kernel launch.
    """
    assert x1.shape == x2.shape == (M, RNUMEL)
    assert x1.dtype == x2.dtype == torch.float32

    out1 = torch.empty_like(x1)
    out2 = torch.empty_like(x2)

    BLOCK_N = 1024
    seed1 = torch.randint(0, 2**31, (1,), device=x1.device, dtype=torch.int64).item()
    seed2 = torch.randint(0, 2**31, (1,), device=x1.device, dtype=torch.int64).item()

    grid = (2 * M,)  # first M programs for pass 1, next M for pass 2
    dual_persistent_softmax_dropout_kernel[grid](
        x1, out1, x2, out2,
        seed1, seed2,
        M,
        N=RNUMEL,
        BLOCK_N=BLOCK_N,
        DROPOUT_P=DROPOUT_P,
        DROPOUT_SCALE=1.0 / (1.0 - DROPOUT_P),
    )
    return out1, out2


# --- Reference implementation ---

def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    """Reference softmax matching the repro pattern (f32 in, f32 out)."""
    m = x.max(dim=-1, keepdim=True).values
    e = torch.exp(x - m)
    s = e.sum(dim=-1, keepdim=True)
    return e / s


def softmax_dropout_reference(x: torch.Tensor, p: float = DROPOUT_P) -> torch.Tensor:
    """Reference softmax + dropout (for validating dropout scaling)."""
    s = softmax_reference(x)
    mask = torch.rand_like(s) > p
    return s * mask * (1.0 / (1.0 - p))


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


def benchmark_triton_single(x: torch.Tensor, warmup=25, rep=100, with_dropout=True):
    """Benchmark a single softmax+dropout kernel."""
    output = torch.empty_like(x)
    BLOCK_N = 1024
    M_val = x.shape[0]
    grid = (M_val,)

    if with_dropout:
        seed = 42

        def run():
            persistent_softmax_dropout_kernel[grid](
                x, output, seed,
                M_val,
                N=RNUMEL,
                BLOCK_N=BLOCK_N,
                DROPOUT_P=DROPOUT_P,
                DROPOUT_SCALE=1.0 / (1.0 - DROPOUT_P),
            )
    else:
        def run():
            persistent_softmax_kernel[grid](
                x, output,
                M_val,
                N=RNUMEL,
                BLOCK_N=BLOCK_N,
            )

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def benchmark_triton_dual(x1: torch.Tensor, x2: torch.Tensor, warmup=25, rep=100):
    """Benchmark the dual softmax+dropout kernel (both passes in one launch)."""
    out1 = torch.empty_like(x1)
    out2 = torch.empty_like(x2)
    BLOCK_N = 1024
    seed1 = 42
    seed2 = 137

    grid = (2 * M,)

    def run():
        dual_persistent_softmax_dropout_kernel[grid](
            x1, out1, x2, out2,
            seed1, seed2,
            M,
            N=RNUMEL,
            BLOCK_N=BLOCK_N,
            DROPOUT_P=DROPOUT_P,
            DROPOUT_SCALE=1.0 / (1.0 - DROPOUT_P),
        )

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def benchmark_compiled_full(warmup=25, rep=100):
    """Benchmark the full repro with torch.compile for comparison."""
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


def correctness_check(M_val=M):
    """Verify the Triton kernel matches the reference softmax."""
    torch.manual_seed(42)
    x = torch.randn(M_val, RNUMEL, dtype=torch.float32, device="cuda")

    # Check pure softmax (without dropout, since dropout is stochastic)
    ref = softmax_reference(x)
    out = oracle_softmax(x, with_dropout=False)

    max_diff = (ref - out).abs().max().item()
    mean_diff = (ref - out).abs().mean().item()
    allclose = torch.allclose(ref, out, atol=1e-5, rtol=1e-5)

    print(f"Correctness check (softmax only, f32):")
    print(f"  Shape: [{M_val}, {RNUMEL}]")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  torch.allclose (atol=1e-5, rtol=1e-5): {allclose}")

    if not allclose:
        print("  WARNING: Output does not match reference within tolerance!")
        for row in [0, 100, 1000, 50000]:
            if row < M_val:
                row_diff = (ref[row] - out[row]).abs().max().item()
                print(f"  Row {row} max diff: {row_diff:.6e}")

    # Also verify dropout scaling is correct (statistically)
    torch.manual_seed(123)
    out_drop = oracle_softmax(x, with_dropout=True)
    # After softmax+dropout, non-zero elements should be scaled by 1/(1-p)
    nonzero_mask = out_drop != 0
    if nonzero_mask.any():
        # Ratio of dropout output to softmax output (for non-zero entries)
        ratios = out_drop[nonzero_mask] / ref[nonzero_mask]
        expected_scale = 1.0 / (1.0 - DROPOUT_P)
        scale_ok = torch.allclose(ratios, torch.full_like(ratios, expected_scale), atol=1e-4)
        # Fraction dropped should be approximately DROPOUT_P
        frac_dropped = 1.0 - nonzero_mask.float().mean().item()
        print(f"\n  Dropout verification:")
        print(f"    Scale correct (1/{1-DROPOUT_P:.1f} = {expected_scale:.4f}): {scale_ok}")
        print(f"    Fraction dropped: {frac_dropped:.4f} (expected ~{DROPOUT_P})")

    return allclose


def main():
    parser = argparse.ArgumentParser(
        description="Oracle persistent softmax benchmark (T5 dual attention)"
    )
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--check-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rep", type=int, default=100, help="Number of benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--check-m", type=int, default=1024, help="Rows for correctness check")
    parser.add_argument("--csv", type=str, default=None, help="Append results to CSV file")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile baseline")
    args = parser.parse_args()

    if args.check_only:
        args.check = True
        args.bench = False
    elif not args.check and not args.bench:
        args.check = True
        args.bench = True

    print("=" * 70)
    print(f"Oracle Persistent Softmax Benchmark: {REPRO_ID}")
    print(f"T5 dual attention softmax+dropout, rnumel={RNUMEL} (persistent)")
    print(f"Per-pass shape: f32[{M}, {RNUMEL}] ({M} rows, {RNUMEL} cols)")
    print(f"Two passes (encoder + decoder attention)")
    print("=" * 70)

    # Compute memory metrics for the reduction kernels
    # Two softmax+dropout passes: each reads and writes M*N*4 bytes
    bytes_per_pass = M * RNUMEL * 4 * 2  # read + write
    total_bytes = bytes_per_pass * 2  # two passes
    sol_us = total_bytes / 3.35e6  # H100 HBM ~3.35 TB/s
    print(f"\nMemory (dual softmax+dropout kernel):")
    print(f"  Per-pass: {bytes_per_pass / 1e6:.1f} MB (read + write)")
    print(f"  Total (2 passes): {total_bytes / 1e6:.1f} MB")
    print(f"  SOL (3.35 TB/s): {sol_us:.1f} us")
    print(f"  Note: Full repro includes relative position bias (embedding + arithmetic)")

    if args.check:
        print(f"\n--- Correctness ---")
        ok = correctness_check(M_val=args.check_m)
        if not ok:
            print("FAILED correctness check. Exiting.")
            sys.exit(1)
        print("PASSED")

    if not args.bench:
        return

    # Benchmark
    print(f"\n--- Benchmark (rep={args.rep}, warmup={args.warmup}) ---")
    torch.manual_seed(42)
    x1 = torch.randn(M, RNUMEL, dtype=torch.float32, device="cuda")
    x2 = torch.randn(M, RNUMEL, dtype=torch.float32, device="cuda")

    # Single pass softmax+dropout
    single_us = benchmark_triton_single(x1, warmup=args.warmup, rep=args.rep, with_dropout=True)
    single_bw = bytes_per_pass / (single_us * 1e-6) / 1e12
    single_pct_sol = (bytes_per_pass / 3.35e6) / single_us * 100.0
    print(f"\nTriton persistent softmax+dropout (single pass, rnumel={RNUMEL}):")
    print(f"  Median: {single_us:.1f} us")
    print(f"  Effective BW: {single_bw:.3f} TB/s")
    print(f"  % of SOL: {single_pct_sol:.1f}%")

    # Dual pass (both softmax+dropout in one launch)
    dual_us = benchmark_triton_dual(x1, x2, warmup=args.warmup, rep=args.rep)
    dual_bw = total_bytes / (dual_us * 1e-6) / 1e12
    dual_pct_sol = sol_us / dual_us * 100.0
    print(f"\nTriton dual softmax+dropout (both passes, one launch):")
    print(f"  Median: {dual_us:.1f} us")
    print(f"  Effective BW: {dual_bw:.3f} TB/s")
    print(f"  % of SOL: {dual_pct_sol:.1f}%")

    # Two separate launches for comparison
    two_launch_us = single_us * 2  # approximate (ignoring second pass benchmark)
    print(f"\n  Two separate launches (estimated): {two_launch_us:.1f} us")
    if dual_us < two_launch_us:
        print(f"  Dual kernel saves: {two_launch_us - dual_us:.1f} us ({(1 - dual_us/two_launch_us)*100:.1f}%)")

    # torch.compile full repro baseline
    compile_us = None
    if not args.no_compile:
        print(f"\nRunning torch.compile full repro baseline (with CD tuning)...")
        try:
            compile_us = benchmark_compiled_full(warmup=args.warmup, rep=args.rep)
            print(f"torch.compile (full repro with CD):")
            print(f"  Median: {compile_us:.1f} us")
            ratio = dual_us / compile_us * 100.0
            print(f"  Oracle dual softmax = {ratio:.1f}% of full compiled time")
        except Exception as e:
            print(f"  torch.compile baseline failed: {e}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Oracle single softmax+dropout: {single_us:.1f} us")
    print(f"  Oracle dual softmax+dropout:   {dual_us:.1f} us")
    print(f"  SOL (dual, 3.35 TB/s):         {sol_us:.1f} us")
    print(f"  Oracle achieves {dual_pct_sol:.1f}% of SOL")
    if compile_us:
        print(f"  Full compiled repro:           {compile_us:.1f} us")
        print(f"  Softmax kernels are {dual_us/compile_us*100:.1f}% of total")
    print(f"\n  Key: rnumel={RNUMEL} = BLOCK_N -> perfect persistent single-tile reduction")
    print(f"  No loop iteration needed, entire softmax computed in registers")

    # Use single-pass as the oracle floor (conservative: excludes pre-softmax work)
    oracle_us = dual_us

    # Write CSV if requested
    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    if args.csv is not None or DEFAULT_CSV.exists():
        _write_csv(csv_path, oracle_us, compile_us, total_bytes)


def _write_csv(csv_path: Path, oracle_us: float, compile_us: float | None, total_bytes: int):
    """Append measurement to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repro_id": REPRO_ID,
        "kernel_type": "triton_persistent_dual_softmax_dropout",
        "shape": f"f32[{M},{RNUMEL}]x2",
        "oracle_us": f"{oracle_us:.1f}",
        "compile_us": f"{compile_us:.1f}" if compile_us else "",
        "total_bytes": str(total_bytes),
        "effective_bw_tb_s": f"{total_bytes / (oracle_us * 1e-6) / 1e12:.3f}",
        "sol_us": f"{total_bytes / 3.35e6:.1f}",
        "pct_sol": f"{(total_bytes / 3.35e6) / oracle_us * 100:.1f}",
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
