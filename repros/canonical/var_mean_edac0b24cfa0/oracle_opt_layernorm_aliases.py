"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete OPT/vLLM fp16 `[4,512,768]` to fp32 hidden-size-768 layernorm, affine epilogue, fp16 cast, and three returned `[2048,768]` views aliasing one output allocation in a single shape-specialized Triton row kernel, whereas Inductor's norm-template lowering has the same essential row-local reduction, affine reads, fp16 output store, and metadata-only alias views for this scope; Inductor cannot materially improve this today because the remaining work is the required input/weight/bias traffic, two hidden-dimension reductions, rsqrt, and output write rather than a missed producer/consumer fusion; the fix is BANDWIDTH_BOUND: record this as an at-floor normalization-template case unless broader layernorm codegen or launch-overhead work moves the family."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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
BASE_SHAPE = (4, 512, 768)
BASE_STRIDE = (512 * 768, 768, 1)
ROWS = 4 * 512
HIDDEN = 768
OUTPUT_SHAPE = (ROWS, HIDDEN)
EPS = 1.0e-5
BLOCK_H = 1024

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=1, num_stages=4),
            triton.Config({}, num_warps=2, num_stages=4),
            triton.Config({}, num_warps=4, num_stages=4),
            triton.Config({}, num_warps=8, num_stages=4),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _opt_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden

        centered_for_reduce = tl.where(mask, x - mean, 0.0)
        variance = tl.sum(centered_for_reduce * centered_for_reduce, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * invstd * weight + bias
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    x, weight, bias, *view_shapes_raw = inputs
    tensor_inputs = (x, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")

    expected_shapes = (BASE_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if weight.device != x.device or bias.device != x.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    view_shapes = tuple(_shape_tuple(shape) for shape in view_shapes_raw)
    if len(view_shapes) != 3:
        raise ValueError(f"expected 3 output shape parameters, got {len(view_shapes)}")
    for index, shape in enumerate(view_shapes, start=3):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"input {index} unexpected output shape parameter: {shape!r}")

    return x, weight, bias, view_shapes


@oracle_impl(hardware="H100", shapes="(T([4, 512, 768], f16), T([768], f16), T([768], f16), S([2048, 768]), S([2048, 768]), S([2048, 768]))")
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
        raise RuntimeError("Triton is required for oracle_opt_layernorm_aliases.py")

    x, weight, bias, view_shapes = _validate_inputs(inputs)
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=x.device,
        dtype=torch.float16,
    )
    _opt_layernorm_kernel[(ROWS,)](
        x,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
    )
    return tuple(base.view(shape) for shape in view_shapes)


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
