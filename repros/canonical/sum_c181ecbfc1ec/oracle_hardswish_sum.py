"""Gap diagnosis: this oracle computes the full captured reshape, thresholded hard-swish-style gate, and channel sum in one Triton reduction kernel, whereas Inductor currently lowers the decomposed gate and reduction through its generic scheduler path for this pointwise producer plus column-reduction shape; Inductor cannot do this today because scheduler/codegen does not specialize simple thresholded activation producers as reduction epilogues for fixed singleton-spatial layouts; the fix is SCHEDULER_FUSION: inline the hard-swish gate into the cross-batch reduction schedule and emit one full-scope column-reduction kernel without intermediate pointwise materialization."""
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 512, "BLOCK_N": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 512, "BLOCK_N": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 512, "BLOCK_N": 16}, num_warps=8, num_stages=3),
        ],
        key=["M", "N"],
    )
    @triton.jit
    def _hardswish_sum_kernel(
        mm_ptr,
        convolution_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets = rows[:, None] * N + cols[None, :]
        mask = (rows[:, None] < M) & (cols[None, :] < N)

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg = tl.load(convolution_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        linear = mm * (arg / 3.0 + 0.5)
        below_upper = tl.where(arg < 3.0, linear, mm)
        gated = tl.where(arg <= -3.0, 0.0, below_upper)
        reduced = tl.sum(gated, axis=0)

        tl.store(out_ptr + cols, reduced, mask=cols < N)


def _validate_inputs(
    mm: torch.Tensor,
    convolution_62: torch.Tensor,
    shape_param: object,
) -> tuple[int, int]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if convolution_62.device != mm.device:
        raise ValueError("inputs must be on the same CUDA device")
    if mm.dtype != torch.float32 or convolution_62.dtype != torch.float32:
        raise ValueError("expected f32 mm and convolution_62 inputs")
    if len(mm.shape) != 2 or len(convolution_62.shape) != 4:
        raise ValueError("expected mm [M,N] and convolution_62 [M,N,1,1]")

    m, n = int(mm.shape[0]), int(mm.shape[1])
    expected_conv_shape = (m, n, 1, 1)
    if tuple(convolution_62.shape) != expected_conv_shape:
        raise ValueError(
            f"unexpected convolution_62 shape: got={tuple(convolution_62.shape)} "
            f"expected={expected_conv_shape}"
        )
    if tuple(shape_param) != expected_conv_shape:
        raise ValueError(
            f"unexpected reshape parameter: got={tuple(shape_param)} "
            f"expected={expected_conv_shape}"
        )
    if not mm.is_contiguous() or not convolution_62.is_contiguous():
        raise ValueError("oracle expects contiguous captured inputs")
    return m, n


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
    mm, convolution_62, shape_param = inputs
    m, n = _validate_inputs(mm, convolution_62, shape_param)

    out = torch.empty((n,), device=mm.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)
    _hardswish_sum_kernel[grid](
        mm,
        convolution_62,
        out,
        M=m,
        N=n,
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
