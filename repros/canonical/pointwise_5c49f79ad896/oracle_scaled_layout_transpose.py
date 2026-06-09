"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ALBERT scaled attention-score layout scope by materializing the required fresh contiguous `[8, 512, 4096]` clone storage with one Triton kernel and returning the same `[4096, 4096]` transpose view with stride `(1, 4096)`, whereas Inductor already lowers the decomposed view/mul/permute/permute/view/clone/view/permute chain to the same mandatory layout-copy envelope; Inductor cannot materially improve this local repro because the returned tensor aliases a newly materialized clone, so every f32 element must be read, scaled, and written; the fix is BANDWIDTH_BOUND: record this as an at-floor layout-copy case unless broader memory-throughput or launch-overhead improvements move both paths."""
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
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
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
            triton.Config({"YBLOCK": 16, "XBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 64, "XBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 16, "XBLOCK": 128}, num_warps=4, num_stages=3),
            triton.Config({"YBLOCK": 32, "XBLOCK": 128}, num_warps=8, num_stages=3),
        ],
        key=["R", "E"],
    )
    @triton.jit
    def _scaled_layout_clone_kernel(
        input_ptr,
        base_ptr,
        R: tl.constexpr,
        E: tl.constexpr,
        YBLOCK: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        row = tl.program_id(1) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
        col = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[None, :]
        mask = (row < R) & (col < E)

        # Base row is `a * 512 + d`; base column is `b * 64 + c`.
        a = row // 512
        d = row - a * 512
        b = col // 64
        c = col - b * 64

        input_offsets = (a * 64 + b) * 32768 + c * 512 + d
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(base_ptr + row * E + col, values * 0.3535533905932738, mask=mask)


def _expect_shape(value, expected: tuple[int, ...], name: str) -> None:
    actual = tuple(int(dim) for dim in value)
    if actual != expected:
        raise ValueError(f"{name} must be {expected}, got {actual}")


@oracle_impl(hardware="H100", shapes="(T([512, 64, 512], f32), S([8, 64, 64, 512]), S([8, 512, 4096]), S([4096, 4096]))")
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

    bmm_42, shape0, shape1, shape2 = inputs
    if not isinstance(bmm_42, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not bmm_42.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if bmm_42.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {bmm_42.dtype}")
    if tuple(bmm_42.shape) != (512, 64, 512):
        raise ValueError(f"unexpected input shape {tuple(bmm_42.shape)}")
    if not bmm_42.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(bmm_42.stride())}")

    _expect_shape(shape0, (8, 64, 64, 512), "shape0")
    _expect_shape(shape1, (8, 512, 4096), "shape1")
    _expect_shape(shape2, (4096, 4096), "shape2")

    base = torch.empty((4096, 4096), device=bmm_42.device, dtype=bmm_42.dtype)
    grid = lambda meta: (
        triton.cdiv(base.shape[1], meta["XBLOCK"]),
        triton.cdiv(base.shape[0], meta["YBLOCK"]),
    )
    _scaled_layout_clone_kernel[grid](
        bmm_42,
        base,
        R=base.shape[0],
        E=base.shape[1],
    )
    return base.permute(1, 0)


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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
