"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete hard-swish gate and channel sum in one shape-specialized tiled reduction kernel over the viewed `[512, 1280, 1, 1]` tensor, whereas Inductor's generic scheduled reduction is measurably slower on the same full scope; Inductor cannot do this today because its scheduler/codegen does not choose this narrow-channel column-reduction specialization for the MobileNet hard-swish gate feeding `sum([0, 2, 3])`; the fix is SCHEDULER_FUSION: recognize pointwise hard-swish gates feeding singleton-spatial channel sums and emit a fused column reduction with the gate computed inside the accumulator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.jit
    def _hardswish_sum_kernel(
        mm_ptr,
        gate_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offsets = rows[:, None] * N + cols[None, :]

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate_arg = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        linear_gate = gate_arg / 3.0 + 0.5
        gated = mm * linear_gate
        value = tl.where(gate_arg < 3.0, gated, mm)
        value = tl.where(gate_arg <= -3.0, 0.0, value)
        reduced = tl.sum(tl.where(mask, value, 0.0), axis=0)

        tl.store(out_ptr + cols, reduced, mask=cols < N)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 3:
        raise ValueError(f"expected 3 inputs, got {len(inputs)}")

    mm, arg319_1, shape_param = inputs
    if not isinstance(mm, torch.Tensor) or not isinstance(arg319_1, torch.Tensor):
        raise TypeError("expected tensor inputs for mm and arg319_1")
    if mm.device.type != "cuda" or arg319_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if mm.dtype != torch.float32 or arg319_1.dtype != torch.float32:
        raise ValueError("expected f32 mm and f32 arg319_1")
    if mm.ndim != 2 or arg319_1.ndim != 4:
        raise ValueError(f"unexpected input ranks: mm={mm.ndim}, arg319_1={arg319_1.ndim}")

    m, n = int(mm.shape[0]), int(mm.shape[1])
    expected_view = (m, n, 1, 1)
    if tuple(arg319_1.shape) != expected_view:
        raise ValueError(f"unexpected arg319_1 shape: {tuple(arg319_1.shape)}")
    if tuple(shape_param) != expected_view:
        raise ValueError(f"unexpected view shape parameter: {shape_param}")
    if not mm.is_contiguous() or not arg319_1.is_contiguous():
        raise ValueError("oracle expects the captured contiguous inputs")

    out = torch.empty_strided((n,), (1,), device=mm.device, dtype=torch.float32)
    block_m = 512
    block_n = 4
    _hardswish_sum_kernel[(triton.cdiv(n, block_n),)](
        mm,
        arg319_1,
        out,
        M=m,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return out


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # All timing must go through bench_oracle(). Direct do_bench or
            # compiled(*inputs) timing includes dispatch overhead and can invent
            # fake gaps for fast kernels.
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
