"""
Oracle kernel for sum_e00c7291b6ee (softmax backward, multi-output reduction).

Pattern: Softmax backward producing a single bf16[M, N] output.
    Inputs:
      - arg3_1:  bf16 scalar (grad_output, broadcast to [M, N])
      - arg0_1:  bf16[M, N] (logits or activations from forward)
      - arg1_1:  f32[M, 1] (per-row max, i.e. log-sum-exp shift)
      - arg2_1:  f32[M, 1] (per-row sum of exps)
      - shape:   [M, N] = [8192, 262144] for the default config

    Computation:
      1. probs = to_bf16(exp(to_f32(arg0_1) - arg1_1) / arg2_1)  -- softmax probs
      2. probs_f32 = to_f32(probs)
      3. grad_f32 = to_f32(arg3_1)  -- scalar broadcast
      4. mul_tensor = grad_f32 * probs_f32
      5. row_sum = sum(mul_tensor, dim=-1, keepdim=True)  -- REDUCTION
      6. output = to_bf16(mul_tensor - probs_f32 * row_sum)

    Oracle strategy:
      Fuse the entire computation into a two-pass Triton kernel per row:
      - Pass 1: Compute probs from arg0_1/arg1_1/arg2_1, multiply by grad scalar,
        accumulate the row sum. Do NOT store intermediate probs (recompute in pass 2).
      - Pass 2: Recompute probs, compute output = grad*probs - probs*row_sum
        = probs * (grad - row_sum), cast to bf16, store.

      This avoids materializing the full [M, N] intermediate in f32, saving
      one full write + read of M*N*4 bytes (e.g., 8 GB for [8192, 262144]).

      Each row (rnumel=N) is processed by one Triton program. With M=8192 rows,
      we have 8192 programs -- enough to saturate the GPU.

    Gap diagnosis (classification: SCHEDULER_FUSION): this oracle differs from
      Inductor by keeping the softmax-backward row reduction and its dependent
      elementwise epilogue inside one persistent per-row kernel, recomputing the
      bf16-rounded probabilities instead of materializing the large f32
      intermediates around `sum(mul_tensor, dim=-1)`. Inductor currently lowers
      the row reduction and post-reduction pointwise as separate scheduled
      regions, so the full `[8192, 262144]` producer path is written and reread
      unnecessarily. The fix is SCHEDULER_FUSION support for dependent
      reduction-plus-epilogue templates that preserve the repro's precision
      round-trip without extra materialization.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

REPRO_ID = "sum_e00c7291b6ee"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Triton kernel: Fused softmax backward (two-pass per row, recomputing probs)
# ---------------------------------------------------------------------------

@triton.jit
def _softmax_backward_fused_kernel(
    arg0_ptr,       # bf16[M, N] -- logits/activations
    arg1_ptr,       # f32[M, 1] -- per-row max
    arg2_ptr,       # f32[M, 1] -- per-row sum_exp
    grad_scalar,    # f32 scalar (the broadcast grad_output value)
    out_ptr,        # bf16[M, N] -- output
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused softmax backward: one program per row.

    Pass 1: compute probs, accumulate grad*probs -> row_sum
    Pass 2: recompute probs, output = probs * (grad - row_sum), cast to bf16
    """
    row = tl.program_id(0)
    row_offset = row * N

    # Load per-row constants
    max_val = tl.load(arg1_ptr + row)      # f32
    sum_exp = tl.load(arg2_ptr + row)      # f32

    # Pass 1: accumulate row_sum = sum(grad * probs) over N elements
    row_sum = tl.zeros([1], dtype=tl.float32)

    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        # Load logits and compute probs
        x_bf16 = tl.load(arg0_ptr + row_offset + offsets, mask=mask, other=0.0)
        x_f32 = x_bf16.to(tl.float32)
        exp_val = tl.exp(x_f32 - max_val)
        prob_f32 = exp_val / sum_exp
        # bf16 round-trip (matching the repro's precision behavior)
        prob_bf16 = prob_f32.to(tl.bfloat16)
        prob_f32_rt = prob_bf16.to(tl.float32)

        mul_val = grad_scalar * prob_f32_rt
        row_sum += tl.sum(mul_val * mask.to(tl.float32), axis=0)

    # Pass 2: recompute probs and write output
    for block_start in range(0, N, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        x_bf16 = tl.load(arg0_ptr + row_offset + offsets, mask=mask, other=0.0)
        x_f32 = x_bf16.to(tl.float32)
        exp_val = tl.exp(x_f32 - max_val)
        prob_f32 = exp_val / sum_exp
        prob_bf16 = prob_f32.to(tl.bfloat16)
        prob_f32_rt = prob_bf16.to(tl.float32)

        # output = grad*probs - probs*row_sum = probs*(grad - row_sum)
        # But to match exact repro semantics:
        # fma(neg(probs), row_sum, grad*probs) = -probs*row_sum + grad*probs
        mul_val = grad_scalar * prob_f32_rt
        result_f32 = mul_val - prob_f32_rt * row_sum
        result_bf16 = result_f32.to(tl.bfloat16)

        tl.store(out_ptr + row_offset + offsets, result_bf16, mask=mask)


def oracle_softmax_backward(arg3_1, arg0_1, arg1_1, arg2_1, shape):
    """Fused softmax backward oracle.

    Args:
        arg3_1: bf16 scalar (grad_output)
        arg0_1: bf16[M, N] (logits)
        arg1_1: f32[M, 1] (per-row max)
        arg2_1: f32[M, 1] (per-row sum_exp)
        shape:  [M, N] shape tuple

    Returns:
        bf16[M, N] output
    """
    M, N = shape[0], shape[1]
    device = arg0_1.device

    # Extract scalar grad value
    grad_scalar = arg3_1.float().item()

    # Flatten per-row vectors for simple indexing
    arg1_flat = arg1_1.view(M)
    arg2_flat = arg2_1.view(M)

    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    # Choose BLOCK_SIZE: balance occupancy vs register pressure
    # For N=262144, BLOCK_SIZE=4096 gives 64 iterations per pass
    BLOCK_SIZE = 4096

    grid = (M,)
    _softmax_backward_fused_kernel[grid](
        arg0_1, arg1_flat, arg2_flat,
        grad_scalar,
        out,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def make_inputs(device: torch.device = None) -> tuple:
    """Generate inputs using the repro's make_inputs."""
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device) if device else value)
        else:
            moved.append(value)
    return tuple(moved)


def check_correctness(device: torch.device, rtol: float = 2e-2, atol: float = 1e-2):
    """Compare oracle output against eager reference."""
    inputs = make_inputs(device)
    module = _load_repro_module()

    # Unpack inputs: arg3_1 (scalar), arg0_1 (matrix), arg1_1, arg2_1, shape
    arg3_1, arg0_1, arg1_1, arg2_1, shape_param = inputs

    with torch.no_grad():
        oracle_out = oracle_softmax_backward(arg3_1, arg0_1, arg1_1, arg2_1, shape_param)
        ref_out = module.Repro()(*inputs)

    max_diff = (oracle_out.float() - ref_out.float()).abs().max().item()
    close = torch.allclose(oracle_out.float(), ref_out.float(), rtol=rtol, atol=atol)
    print(f"  output: shape={list(oracle_out.shape)}, dtype={oracle_out.dtype}")
    print(f"  max_abs_diff={max_diff:.6g}, allclose(rtol={rtol}, atol={atol})={close}")
    if not close:
        rel_err = ((oracle_out.float() - ref_out.float()).abs() /
                   (ref_out.float().abs() + 1e-8)).max().item()
        print(f"  max_rel_err={rel_err:.6g}")
    print(f"  OVERALL: {'PASS' if close else 'FAIL'}")
    return close


def benchmark_oracle(device: torch.device, warmup: int = 10, rep: int = 100):
    """Benchmark the fused oracle kernel vs torch.compile."""
    inputs = make_inputs(device)
    arg3_1, arg0_1, arg1_1, arg2_1, shape_param = inputs
    module = _load_repro_module()

    with torch.no_grad():
        # Warmup oracle
        oracle_softmax_backward(arg3_1, arg0_1, arg1_1, arg2_1, shape_param)
        torch.cuda.synchronize()

        # Benchmark oracle
        def oracle_fn():
            return oracle_softmax_backward(arg3_1, arg0_1, arg1_1, arg2_1, shape_param)

        ms_oracle = triton.testing.do_bench(oracle_fn, warmup=warmup, rep=rep)
        us_oracle = ms_oracle * 1000.0
        print(f"  oracle (fused softmax backward): {us_oracle:.3f} us")

        # Benchmark torch.compile
        repro_instance = module.Repro()
        compiled = torch.compile(repro_instance)
        compiled(*inputs)
        torch.cuda.synchronize()

        ms_compiled = triton.testing.do_bench(lambda: compiled(*inputs), warmup=warmup, rep=rep)
        us_compiled = ms_compiled * 1000.0
        print(f"  compiled (torch.compile):         {us_compiled:.3f} us")

        # Memory bandwidth analysis
        M, N = shape_param[0], shape_param[1]
        # Reads: arg0_1 (bf16, M*N*2) + arg1_1 (f32, M*4) + arg2_1 (f32, M*4)
        # Writes: output (bf16, M*N*2)
        # The oracle reads arg0_1 TWICE (two passes) but avoids intermediate storage
        bytes_read = M * N * 2 * 2 + M * 4 + M * 4  # arg0 read twice, arg1, arg2
        bytes_written = M * N * 2
        total_bytes = bytes_read + bytes_written
        bw_oracle = total_bytes / (ms_oracle * 1e-3) / 1e9  # GB/s
        print(f"\n  Memory traffic (oracle): {total_bytes / 1e9:.3f} GB")
        print(f"  Effective bandwidth:     {bw_oracle:.1f} GB/s")

        # Speedup
        if us_compiled > 0:
            speedup = us_compiled / us_oracle
            print(f"\n  Speedup vs compile: {speedup:.2f}x")

    return us_oracle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Run correctness check")
    parser.add_argument("--bench", action="store_true", help="Run benchmark")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.check:
        print(f"Correctness check ({REPRO_ID}):")
        ok = check_correctness(device, rtol=args.rtol, atol=args.atol)
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmark ({REPRO_ID}):")
        benchmark_oracle(device, warmup=args.warmup, rep=args.rep)

    if not args.check and not args.bench:
        print("Use --check for correctness or --bench for benchmarking")
        print("Running both by default...")
        print(f"\nCorrectness check ({REPRO_ID}):")
        ok = check_correctness(device, rtol=args.rtol, atol=args.atol)
        if not ok:
            sys.exit(1)
        print(f"\nBenchmark ({REPRO_ID}):")
        benchmark_oracle(device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
