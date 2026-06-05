"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete `aten.glu.default(dim=1)` scope with one output-space Triton pointwise kernel that directly reads the two contiguous channel halves and stores the final contiguous `[B, C/2, T]` result, whereas Inductor already lowers this isolated GLU region to a single fused pointwise kernel over the same data movement; Inductor cannot materially improve this captured repro today because there is no producer/consumer fusion, reduction, scatter, split-K, or algebraic dead work left beyond the mandatory two f32 input reads, sigmoid/multiply math, and f32 output store; the fix is BANDWIDTH_BOUND: record this as the launch plus memory/math floor unless a broader graph capture exposes adjacent work to fuse."""
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
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=4),
        ],
        key=["INNER"],
    )
    @triton.jit
    def oracle_kernel(
        input_ptr,
        output_ptr,
        INNER: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        batch = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < INNER

        input_base = batch * (INNER * 2)
        output_base = batch * INNER
        gate = tl.load(input_ptr + input_base + offsets + INNER, mask=mask, other=0.0)
        value = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0)
        result = value * tl.sigmoid(gate)
        tl.store(output_ptr + output_base + offsets, result, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_glu.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")

    (convolution_11,) = inputs
    if not isinstance(convolution_11, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(convolution_11)!r}")
    if convolution_11.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if convolution_11.dtype != torch.float32:
        raise TypeError(f"expected f32 input, got {convolution_11.dtype}")
    if convolution_11.dim() != 3:
        raise ValueError(f"expected rank-3 GLU input, got shape={tuple(convolution_11.shape)}")
    if not convolution_11.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(convolution_11.stride())}")

    batch, channels, width = (int(dim) for dim in convolution_11.shape)
    if channels % 2 != 0:
        raise ValueError(f"GLU channel dimension must be even, got {channels}")
    return convolution_11, batch, channels // 2, width


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
    convolution_11, batch, half_channels, width = _validate_inputs(inputs)
    output_shape = (batch, half_channels, width)
    output = torch.empty_strided(
        output_shape,
        (half_channels * width, width, 1),
        device=convolution_11.device,
        dtype=convolution_11.dtype,
    )
    inner = half_channels * width
    grid = lambda meta: (batch, triton.cdiv(inner, meta["BLOCK_N"]))
    oracle_kernel[grid](convolution_11, output, INNER=inner)
    return output


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
