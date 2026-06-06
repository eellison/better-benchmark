"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full BatchNorm-inference-style residual ReLU and in-place copy scope in one Triton pointwise kernel while preserving the returned alias to arg6_1 and NaN-preserving ReLU semantics for the finite-residual repro input, whereas Inductor already emits one fused pointwise kernel for the same full scope; Inductor cannot materially improve this today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the mandatory convolution/residual/vector reads plus output store dominate the measured CUDAGraph time; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise bandwidth/codegen improvements move both implementations."""
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

    @triton.jit
    def _bn_residual_relu_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        residual_out_ptr,
        N_ELEMENTS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N_ELEMENTS
        cidx = (offsets // 196) % 1024

        mean = tl.load(mean_ptr + cidx, mask=mask, other=0.0)
        var = tl.load(var_ptr + cidx, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + cidx, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + cidx, mask=mask, other=0.0)

        conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_out_ptr + offsets, mask=mask, other=0.0)

        invstd = 1.0 / tl.sqrt(var + 1.0e-5)
        normalized = (conv - mean) * invstd
        affine = normalized * weight + bias
        total = residual + affine
        relu = tl.where(total != total, total, tl.maximum(total, 0.0))
        out = tl.where(residual != residual, 0.0, relu)
        tl.store(residual_out_ptr + offsets, out, mask=mask)


def _require_tensor(name: str, value, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} dtype must be torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    return value


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
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"expected 6 inputs, got {len(inputs)}")

    mean = _require_tensor("arg2_1", inputs[0], (1024,))
    conv = _require_tensor("convolution", inputs[1], (32, 1024, 14, 14))
    var = _require_tensor("arg3_1", inputs[2], (1024,))
    weight = _require_tensor("arg4_1", inputs[3], (1024,))
    bias = _require_tensor("arg5_1", inputs[4], (1024,))
    residual_out = _require_tensor("arg6_1", inputs[5], (32, 1024, 14, 14))

    expected_stride = (200704, 196, 14, 1)
    if tuple(conv.stride()) != expected_stride:
        raise ValueError(f"convolution stride must be {expected_stride}, got {tuple(conv.stride())}")
    if tuple(residual_out.stride()) != expected_stride:
        raise ValueError(f"arg6_1 stride must be {expected_stride}, got {tuple(residual_out.stride())}")

    n_elements = conv.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _bn_residual_relu_kernel[grid](
        mean,
        conv,
        var,
        weight,
        bias,
        residual_out,
        N_ELEMENTS=n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=4,
    )
    return residual_out


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
