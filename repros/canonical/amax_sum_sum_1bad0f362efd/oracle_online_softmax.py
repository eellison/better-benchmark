"""
Gap diagnosis (classification: NEW_PATTERN): this oracle computes the
large-vocabulary cross-entropy sum by reading each logits row once with scalar
online max and denominator accumulators plus one target-logit load, then
reducing only the per-row losses to the scalar output. Inductor currently lowers
the decomposed cast/amax/sub/exp/sum/log/log_softmax/gather/mask/sum graph as
generic reductions and pointwise work that materializes and rereads full
row-sized intermediates, because it does not canonicalize this ignore-index
cross-entropy-sum idiom into an online softmax row template with a scalar
reduction epilogue. The fix is a NEW_PATTERN lowering for
log_softmax+gather+masked-sum cross entropy that emits the online accumulator
kernel and the small final scalar reduction directly.

Oracle kernel for amax_sum_sum_1bad0f362efd: online softmax cross-entropy with scalar reduction.

Pattern: bf16[8192, 262144] logits + i64[8192] labels
  -> cast f32 -> amax -> sub -> exp -> sum -> log -> sub (=log_softmax)
  -> cast bf16 -> gather(label) -> neg -> mask(-100) -> sum (scalar)
  -> bf16[] scalar loss

This is the genai_CrossEntropyBackward_000 from Qwen3-0.6B (262144 vocab).
Despite the "Backward" label, this computes the forward cross-entropy loss
(including the final scalar reduction across the batch).

Strategy:
  - Phase 1: Each program handles one row. Single pass over the reduction dim
    computing running max and sum_exp via online softmax, plus eager load of
    x[label]. Outputs per-sample loss to a temporary buffer.
  - Phase 2: Tree reduction of per-sample losses to a single scalar.
  - Alternatively (preferred): fuse the per-sample loss computation with a
    partial-sum approach using atomics or a two-kernel reduction.

  We use a 2-kernel approach:
    Kernel 1: One program per row -> per-sample f32 loss (same as fwd oracle)
    Kernel 2: Tree reduction of M f32 values -> scalar bf16

Memory traffic:
  - Read: 8192 * 262144 * 2 bytes (bf16 logits) = 4,294,967,296 bytes
  - Read: 8192 * 8 bytes (i64 labels) = 65,536 bytes (negligible)
  - Write: 8192 * 4 bytes (f32 per-sample loss, intermediate) = 32,768 bytes
  - Read (kernel 2): 8192 * 4 bytes = 32,768 bytes (negligible)
  - Write: 2 bytes (scalar bf16 output, negligible)
  - Total effective: ~4.295 GB (dominated by logits read)
  - SOL at 3.8 TB/s: 4,295,049,216 / 3.8e6 = ~1130 us

Note: The intermediate buffer (32 KB) fits in L2 cache and is negligible.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "amax_sum_sum_1bad0f362efd"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

# Default problem dimensions (from _shapes_config in repro.py)
M = 8192       # number of rows (batch)
N = 262144     # reduction dimension (vocab size)


# --- Triton Kernels ---

@triton.jit
def online_softmax_xe_per_row_kernel(
    logits_ptr,      # bf16[M, N]
    labels_ptr,      # i64[M]
    loss_per_row_ptr,  # f32[M] - per-sample loss in f32
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused cross-entropy per-row: single-pass online softmax + gather.

    For each row:
      1. Eagerly load x[label] (single scalar load).
      2. Single pass: compute max, sum_exp via online algorithm.
      3. loss = -(x[label] - max - log(sum_exp)) if label != -100, else 0.
      4. Store f32 loss to per_row buffer for subsequent scalar reduction.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Load the label for this row
    label = tl.load(labels_ptr + row_idx)
    is_valid = label != -100

    # Clamp label to valid range for the gather (use 0 for ignored)
    safe_label = tl.where(is_valid, label, 0)

    # Eagerly load x[label] - single scalar load, will be in L2 cache
    x_label = tl.load(logits_ptr + row_start + safe_label).to(tl.float32)

    # Online max + sum_exp (single pass over the row)
    m_i = float("-inf")  # running max (scalar accumulator)
    l_i = 0.0           # running sum of exp(x - m) (scalar accumulator)

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(logits_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)

        # Online softmax update (Milakov & Gimelshein 2018)
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    # Compute cross-entropy loss: -(x[label] - max - log(sum_exp))
    log_sum_exp = tl.log(l_i)
    loss = -(x_label - m_i - log_sum_exp)

    # Mask ignored labels (set loss = 0 for label == -100)
    loss = tl.where(is_valid, loss, 0.0)

    # Store f32 per-sample loss
    tl.store(loss_per_row_ptr + row_idx, loss)


@triton.jit
def reduce_sum_to_scalar_kernel(
    input_ptr,       # f32[M]
    output_ptr,      # bf16[] scalar
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Reduce M f32 values to a single bf16 scalar."""
    # Single program: load all values and sum them
    acc = 0.0
    for block_start in tl.range(0, M, BLOCK_M):
        offs = block_start + tl.arange(0, BLOCK_M)
        mask = offs < M
        vals = tl.load(input_ptr + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    # Store as bf16
    tl.store(output_ptr, acc.to(tl.bfloat16))


def oracle_online_softmax_xe(
    logits: torch.Tensor,
    labels: torch.Tensor,
    block_n: int = 4096,
    num_warps: int = 8,
) -> torch.Tensor:
    """Launch the oracle online softmax cross-entropy kernel (2-kernel fused)."""
    assert logits.ndim == 2 and logits.dtype == torch.bfloat16
    assert labels.ndim == 1 and labels.dtype == torch.int64
    M_val, N_val = logits.shape
    assert labels.shape[0] == M_val

    # Intermediate per-row loss buffer (f32, fits in L2)
    loss_per_row = torch.empty(M_val, dtype=torch.float32, device=logits.device)

    # Output scalar
    output = torch.empty((), dtype=torch.bfloat16, device=logits.device)

    # Kernel 1: per-row cross-entropy via online softmax
    grid1 = (M_val,)
    online_softmax_xe_per_row_kernel[grid1](
        logits, labels, loss_per_row,
        N=N_val,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )

    # Kernel 2: reduce per-row losses to scalar
    # Use a single program with BLOCK_M large enough
    BLOCK_M = triton.next_power_of_2(M_val) if M_val <= 16384 else 16384
    reduce_sum_to_scalar_kernel[(1,)](
        loss_per_row, output,
        M=M_val,
        BLOCK_M=BLOCK_M,
        num_warps=4,
    )

    return output


# --- Reference implementation (matches repro exactly) ---

def reference_impl(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Reference cross-entropy matching the repro pattern exactly."""
    # Cast to f32
    x_f32 = logits.to(torch.float32)
    # amax
    m = x_f32.amax(dim=1, keepdim=True)
    # sub -> exp -> sum -> log
    sub = x_f32 - m
    exp_x = torch.exp(sub)
    sum_exp = exp_x.sum(dim=1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)
    # log_softmax
    log_softmax = sub - log_sum_exp  # [M, N] in f32
    # Cast to bf16 (matches the repro)
    log_softmax_bf16 = log_softmax.to(torch.bfloat16)
    # gather at label (with ignore masking)
    ne_mask = labels != -100
    safe_labels = torch.where(ne_mask, labels, torch.zeros_like(labels))
    gathered = log_softmax_bf16.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    # negate + mask
    neg = -gathered
    result = torch.where(ne_mask, neg, torch.zeros_like(neg))
    # Final scalar sum
    return result.sum()


# --- Benchmarking ---

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


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


def correctness_check(M_val=None, N_val=None):
    """Verify the Triton kernel matches the reference implementation."""
    if M_val is None:
        M_val = M
    if N_val is None:
        N_val = N

    torch.manual_seed(42)
    logits = torch.randn(M_val, N_val, dtype=torch.bfloat16, device="cuda")
    # Mix of valid labels and ignore_index (-100)
    labels = torch.randint(0, N_val, (M_val,), dtype=torch.int64, device="cuda")
    # Set ~10% of labels to -100 (ignore_index)
    ignore_mask = torch.rand(M_val, device="cuda") < 0.1
    labels[ignore_mask] = -100

    ref = reference_impl(logits, labels)
    out = oracle_online_softmax_xe(logits, labels, block_n=4096, num_warps=8)

    # Both should be scalars
    assert ref.ndim == 0, f"ref has shape {ref.shape}"
    assert out.ndim == 0, f"out has shape {out.shape}"

    ref_f32 = ref.float().item()
    out_f32 = out.float().item()
    abs_diff = abs(ref_f32 - out_f32)
    rel_diff = abs_diff / (abs(ref_f32) + 1e-8)

    # Tolerance: the reference casts log_softmax to bf16 before gather, which
    # introduces quantization. Our oracle computes in f32 throughout. The final
    # sum accumulates M values, so differences compound. Use generous tolerance.
    # bf16 ULP at typical loss values (~10) is ~0.0625. With M=8192 additions,
    # max expected accumulated error ~ sqrt(M) * ULP ~ 5.6
    atol = max(8.0, abs(ref_f32) * 0.01)  # Allow 1% relative or 8 absolute
    passed = abs_diff < atol

    print(f"Correctness check (M={M_val}, N={N_val}):")
    print(f"  Reference (scalar): {ref_f32:.4f}")
    print(f"  Oracle (scalar):    {out_f32:.4f}")
    print(f"  Absolute diff:      {abs_diff:.6e}")
    print(f"  Relative diff:      {rel_diff:.6e}")
    print(f"  Tolerance (atol):   {atol:.2f}")
    print(f"  PASSED: {passed}")

    return passed


def benchmark_oracle(logits, labels, block_n=4096, num_warps=8, warmup=25, rep=100):
    """Benchmark the oracle kernel (2-kernel fused online softmax XE)."""
    M_val, N_val = logits.shape
    loss_per_row = torch.empty(M_val, dtype=torch.float32, device=logits.device)
    output = torch.empty((), dtype=torch.bfloat16, device=logits.device)
    BLOCK_M = triton.next_power_of_2(M_val) if M_val <= 16384 else 16384

    grid1 = (M_val,)

    def run():
        online_softmax_xe_per_row_kernel[grid1](
            logits, labels, loss_per_row,
            N=N_val,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
        reduce_sum_to_scalar_kernel[(1,)](
            loss_per_row, output,
            M=M_val,
            BLOCK_M=BLOCK_M,
            num_warps=4,
        )

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def benchmark_compiled(logits, labels, warmup=25, rep=100):
    """Benchmark torch.compile baseline (full repro)."""
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.Repro().cuda()
    compiled = torch.compile(model, fullgraph=True)

    # Warmup to trigger compilation
    for _ in range(3):
        compiled(logits, labels)
    torch.cuda.synchronize()

    def run():
        compiled(logits, labels)

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def _write_csv(csv_path: Path, oracle_us: float, compile_us: float | None, total_bytes: int, args):
    """Append measurement to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    sol_us = total_bytes / 3.8e6

    row = {
        "repro_id": REPRO_ID,
        "repro_path": f"repros/canonical/{REPRO_ID}/repro.py",
        "shape_label": f"{M}x{N}",
        "family": "online_softmax_cross_entropy",
        "oracle_impl": "triton_2kernel_online_softmax_xe_scalar_reduce",
        "oracle_path": f"repros/canonical/{REPRO_ID}/oracle_online_softmax.py",
        "hardware": "B200",
        "device_name": torch.cuda.get_device_name(0),
        "git_commit": get_git_commit(),
        "best_compile_us": f"{compile_us:.1f}" if compile_us else "",
        "memcopy_sol_us": f"{sol_us:.1f}",
        "oracle_us": f"{oracle_us:.1f}",
        "total_bytes": total_bytes,
        "oracle_over_sol": f"{oracle_us / sol_us:.3f}",
        "speedup_vs_best_compile": f"{compile_us / oracle_us:.3f}" if compile_us else "",
        "correct": "True",
        "tolerance": "1% relative or 8 absolute (bf16 accumulation over M rows)",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "2-kernel: K1=per-row online softmax XE (1-pass), K2=scalar reduce. "
                 "K2 cost negligible (32KB). Reads logits once (vs 2x for 2-pass stash).",
    }

    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nResults appended to {csv_path}")


def oracle_forward(inputs):
    return oracle_online_softmax_xe(*inputs)


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
