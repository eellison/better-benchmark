"""
Oracle kernel for amax_sum_f0661488d68c: online softmax cross-entropy forward.

Pattern: bf16[8192, 262144] logits + i64[8192] labels
  -> log_softmax -> gather(label) -> negate -> mask(-100)
  -> bf16[8192] per-sample loss

This is the genai_CrossEntropyForward_000 from Qwen3-0.6B (262144 vocab).

Strategy:
  - Single pass per row: compute max, sum_exp, and read x[label] simultaneously.
  - loss = -(x[label] - max - log(sum_exp)) if label != -100, else 0
  - No need to materialize the full log_softmax matrix (262144 elements per row).
  - Each program handles one row. BLOCK_N tiles over the reduction dimension.

Memory traffic (single pass):
  - Read: 8192 * 262144 * 2 bytes (bf16 logits) = 4,294,967,296 bytes
  - Read: 8192 * 8 bytes (i64 labels) = 65,536 bytes (negligible)
  - Write: 8192 * 2 bytes (bf16 output) = 16,384 bytes (negligible)
  - Total effective: ~4.295 GB
  - SOL at 3.8e6 us/byte formula: 4,295,049,216 / 3.8e6 = ~1130 us

Compared to 2-pass (the existing stash result reads input twice):
  - 2-pass reads: 8192 * 262144 * 2 * 2 = 8.59 GB
  - Our 1-pass reads: 4.295 GB -> potential 2x improvement over 2-pass SOL
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
from triton.testing import do_bench


REPRO_ID = "amax_sum_f0661488d68c"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

# Problem dimensions
M = 8192       # number of rows (batch)
N = 262144     # reduction dimension (vocab size)


# --- Triton Kernel ---

@triton.jit
def cross_entropy_fwd_kernel(
    logits_ptr,     # bf16[M, N]
    labels_ptr,     # i64[M]
    output_ptr,     # bf16[M]
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused cross-entropy forward: single-pass online softmax + gather.

    For each row:
      1. Single pass: compute max, sum_exp via online algorithm,
         and read the logit at the label position.
      2. loss = -(x[label] - max - log(sum_exp))
      3. If label == -100 (ignore_index), loss = 0.
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * N

    # Load the label for this row
    label = tl.load(labels_ptr + row_idx)
    is_valid = label != -100

    # Clamp label to valid range for the gather (use 0 for ignored)
    safe_label = tl.where(is_valid, label, 0)

    # Pass 1: Online max + sum_exp, and gather x[label]
    m_i = float("-inf")  # running max
    l_i = 0.0           # running sum of exp(x - m)
    x_label = 0.0       # logit at label position (f32)

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N

        # Load block of logits, cast to f32
        x = tl.load(logits_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

        # Check if label falls within this block and gather it
        label_in_block = (safe_label >= block_start) & (safe_label < block_start + BLOCK_N)
        if label_in_block:
            label_offset = safe_label - block_start
            # Load the specific element at label position
            x_at_label = tl.load(logits_ptr + row_start + safe_label).to(tl.float32)
            x_label = x_at_label

    # Compute cross-entropy loss: -(x[label] - max - log(sum_exp))
    log_sum_exp = tl.log(l_i)
    loss = -(x_label - m_i - log_sum_exp)

    # Mask ignored labels
    loss = tl.where(is_valid, loss, 0.0)

    # Store result as bf16
    tl.store(output_ptr + row_idx, loss.to(tl.bfloat16))


@triton.jit
def cross_entropy_fwd_kernel_v2(
    logits_ptr,     # bf16[M, N]
    labels_ptr,     # i64[M]
    output_ptr,     # bf16[M]
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused cross-entropy forward v2: gather label outside the main loop.

    This version loads x[label] in a separate scalar load, which avoids
    branch divergence inside the hot loop.
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
    m_i = float("-inf")
    l_i = 0.0

    for block_start in tl.range(0, N, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(logits_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(x, axis=0))
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    # Compute cross-entropy loss: -(x[label] - max - log(sum_exp))
    log_sum_exp = tl.log(l_i)
    loss = -(x_label - m_i - log_sum_exp)

    # Mask ignored labels
    loss = tl.where(is_valid, loss, 0.0)

    # Store result as bf16
    tl.store(output_ptr + row_idx, loss.to(tl.bfloat16))


def oracle_cross_entropy_fwd(
    logits: torch.Tensor,
    labels: torch.Tensor,
    block_n: int = 4096,
    num_warps: int = 8,
    version: int = 2,
) -> torch.Tensor:
    """Launch the oracle cross-entropy forward kernel."""
    assert logits.ndim == 2 and logits.dtype == torch.bfloat16
    assert labels.ndim == 1 and labels.dtype == torch.int64
    M_val, N_val = logits.shape
    assert labels.shape[0] == M_val

    output = torch.empty(M_val, dtype=torch.bfloat16, device=logits.device)

    grid = (M_val,)
    kernel = cross_entropy_fwd_kernel_v2 if version == 2 else cross_entropy_fwd_kernel
    kernel[grid](
        logits, labels, output,
        N=N_val,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return output


# --- Reference implementation (matches repro exactly) ---

def cross_entropy_reference(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
    log_softmax = sub - log_sum_exp  # [M, N]
    # Cast to bf16 (matches the repro)
    log_softmax_bf16 = log_softmax.to(torch.bfloat16)
    # gather at label
    ne_mask = labels != -100
    safe_labels = torch.where(ne_mask, labels, torch.zeros_like(labels))
    gathered = log_softmax_bf16.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    # negate + mask
    neg = -gathered
    result = torch.where(ne_mask, neg, torch.zeros_like(neg))
    return result


# --- Benchmarking ---

def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def benchmark_cuda_graph(fn, warmup=5, rep=50):
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


def correctness_check():
    """Verify the Triton kernel matches the reference implementation."""
    torch.manual_seed(42)
    logits = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    # Mix of valid labels and ignore_index (-100)
    labels = torch.randint(0, N, (M,), dtype=torch.int64, device="cuda")
    # Set ~10% of labels to -100 (ignore_index)
    ignore_mask = torch.rand(M, device="cuda") < 0.1
    labels[ignore_mask] = -100

    ref = cross_entropy_reference(logits, labels)
    out = oracle_cross_entropy_fwd(logits, labels, block_n=4096, num_warps=8, version=2)

    max_diff = (ref.float() - out.float()).abs().max().item()
    mean_diff = (ref.float() - out.float()).abs().mean().item()

    # The reference casts log_softmax to bf16 then gathers, while our oracle
    # computes everything in f32 then casts to bf16. This means we may differ
    # by up to 1 ULP in bf16 due to the intermediate cast. Use generous tolerance.
    allclose = torch.allclose(ref.float(), out.float(), atol=0.125, rtol=0.05)

    print(f"Correctness check:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  torch.allclose (atol=0.125, rtol=0.05): {allclose}")

    # Show a few samples
    valid_mask = labels != -100
    valid_ref = ref[valid_mask][:5]
    valid_out = out[valid_mask][:5]
    print(f"  Sample ref:    {valid_ref.tolist()}")
    print(f"  Sample oracle: {valid_out.tolist()}")

    # Check ignored entries are zero
    ignored_out = out[~valid_mask]
    ignored_zeros = (ignored_out == 0).all().item()
    print(f"  Ignored entries all zero: {ignored_zeros}")

    return allclose and ignored_zeros


def benchmark_oracle(logits, labels, block_n=4096, num_warps=8, warmup=25, rep=100):
    """Benchmark the oracle kernel."""
    output = torch.empty(logits.shape[0], dtype=torch.bfloat16, device=logits.device)
    grid = (logits.shape[0],)

    def run():
        cross_entropy_fwd_kernel_v2[grid](
            logits, labels, output,
            N=logits.shape[1],
            BLOCK_N=block_n,
            num_warps=num_warps,
        )

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def benchmark_compiled(logits, labels, warmup=25, rep=100):
    """Benchmark torch.compile baseline (full repro)."""
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)

    model = module.Repro().cuda()
    compiled = torch.compile(model, fullgraph=True)

    def run():
        compiled(labels, logits)

    return benchmark_cuda_graph(run, warmup=warmup, rep=rep)


def main():
    parser = argparse.ArgumentParser(description="Oracle cross-entropy forward benchmark")
    parser.add_argument("--check-only", action="store_true", help="Only run correctness check")
    parser.add_argument("--rep", type=int, default=100, help="Number of benchmark repetitions")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--block-n", type=int, default=4096, help="Tile size for reduction")
    parser.add_argument("--num-warps", type=int, default=8, help="Number of warps")
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile baseline")
    parser.add_argument("--csv", type=str, default=None, help="Append results to CSV file")
    args = parser.parse_args()

    print("=" * 70)
    print(f"Oracle Cross-Entropy Forward Benchmark: {REPRO_ID}")
    print(f"Shape: bf16[{M}, {N}] logits + i64[{M}] labels -> bf16[{M}] loss")
    print("=" * 70)

    # Compute memory metrics - single pass, read-only dominated
    read_bytes = M * N * 2 + M * 8  # bf16 logits + i64 labels
    write_bytes = M * 2             # bf16 output
    total_bytes = read_bytes + write_bytes
    sol_us = total_bytes / 3.8e6
    print(f"\nMemory traffic (single pass):")
    print(f"  Read:  {read_bytes / 1e9:.4f} GB (logits) + {M * 8 / 1e6:.3f} MB (labels)")
    print(f"  Write: {write_bytes / 1e6:.3f} MB")
    print(f"  Total: {total_bytes / 1e9:.4f} GB")
    print(f"  SOL (3.8 TB/s): {sol_us:.1f} us")
    print(f"\n  NOTE: This is 1-pass (read logits once). The stash baseline reads 2x (2-pass).")

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
    print(f"\n--- Benchmark (rep={args.rep}, warmup={args.warmup}, BLOCK_N={args.block_n}) ---")
    torch.manual_seed(42)
    logits = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    labels = torch.randint(0, N, (M,), dtype=torch.int64, device="cuda")

    # Oracle
    oracle_us = benchmark_oracle(logits, labels, block_n=args.block_n,
                                  num_warps=args.num_warps, warmup=args.warmup, rep=args.rep)
    oracle_bw = total_bytes / (oracle_us * 1e-6) / 1e12  # TB/s
    oracle_pct_sol = sol_us / oracle_us * 100.0
    print(f"\nOracle (single-pass fused cross-entropy):")
    print(f"  Median: {oracle_us:.1f} us")
    print(f"  Effective BW: {oracle_bw:.3f} TB/s")
    print(f"  % of SOL: {oracle_pct_sol:.1f}%")

    # torch.compile baseline
    compile_us = None
    if not args.no_compile:
        print(f"\nRunning torch.compile baseline...")
        compile_us = benchmark_compiled(logits, labels, warmup=args.warmup, rep=args.rep)
        compile_bw = total_bytes / (compile_us * 1e-6) / 1e12
        print(f"torch.compile:")
        print(f"  Median: {compile_us:.1f} us")
        print(f"  Effective BW (vs single-pass bytes): {compile_bw:.3f} TB/s")
        speedup = compile_us / oracle_us
        print(f"\nSpeedup (compile / oracle): {speedup:.2f}x")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Oracle (1-pass fused XE fwd): {oracle_us:.1f} us")
    if compile_us:
        print(f"  Baseline (torch.compile):     {compile_us:.1f} us")
    print(f"  SOL (single-pass):            {sol_us:.1f} us")
    print(f"  Oracle achieves {oracle_pct_sol:.1f}% of single-pass SOL")

    # Write CSV if requested
    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    if args.csv is not None:
        _write_csv(csv_path, oracle_us, compile_us, total_bytes, args)


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
        "oracle_impl": "triton_single_pass_fused_cross_entropy_fwd",
        "oracle_path": f"repros/canonical/{REPRO_ID}/oracle_cross_entropy_fwd.py",
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
        "tolerance": "atol=0.125,rtol=0.05",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "Single-pass: reads logits once (no 2nd pass). Fuses max+sumexp+gather.",
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
