"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Reformer residual-add LayerNorm scope in one shape-specialized Triton row kernel, including the `addmm_9.view([1,4096,256]) + add_33` fp16-rounded residual add, fp32 hidden-size-256 `var_mean(correction=0, keepdim=True)`, eps=1e-12 affine epilogue, fp16 cast, and two returned `[4096,256]` views aliasing one `[1,4096,256]` backing output, whereas tuned Inductor already targets the same fused normalization envelope for this norm-template canonicalization case; Inductor cannot materially improve this repro through local scheduler fusion or algebraic rewrites because the remaining work is required activation/residual/affine memory traffic plus the fixed hidden-dimension reduction and rsqrt latency; the fix is BANDWIDTH_BOUND: record this as a full-scope at-floor normalization case unless broader norm-template or launch-overhead improvements move the baseline."""
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
ROWS = 4096
HIDDEN = 256
EPS = 1.0e-12
INPUT_SHAPE = (ROWS, HIDDEN)
BASE_SHAPE = (1, ROWS, HIDDEN)
BASE_STRIDE = (ROWS * HIDDEN, HIDDEN, 1)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
BLOCK_H = 256
BLOCK_M = 2
CLASSIFICATION = "BANDWIDTH_BOUND"

if triton is not None:

    @triton.jit
    def _reformer_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_m: tl.constexpr,
        block_h: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_h)
        offsets = rows[:, None] * hidden + cols[None, :]

        addmm = tl.load(addmm_ptr + offsets)
        residual = tl.load(residual_ptr + offsets)

        # aten.add on fp16 tensors materializes fp16 before the explicit fp32 cast.
        x = (addmm + residual).to(tl.float16).to(tl.float32)

        mean = tl.sum(x, axis=1) / hidden
        sq_mean = tl.sum(x * x, axis=1) / hidden
        variance = tl.maximum(sq_mean - mean * mean, 0.0)
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        centered = x - mean[:, None]
        y = centered * invstd[:, None] * weight[None, :] + bias[None, :]

        tl.store(out_ptr + offsets, y)


def _shape_tuple(value):
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs):
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm_9, add_33, weight, bias, shape0, shape1, shape2 = inputs
    tensor_inputs = (addmm_9, add_33, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, BASE_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape0) != BASE_SHAPE:
        raise ValueError(f"unexpected addmm view shape parameter: {shape0!r}")
    output_shapes = (_shape_tuple(shape1), _shape_tuple(shape2))
    for index, shape in enumerate(output_shapes, start=1):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"unexpected output shape{index} parameter: {shape!r}")

    return addmm_9, add_33, weight, bias, output_shapes


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
        raise RuntimeError("Triton is required for oracle_reformer_layernorm_aliases.py")

    addmm_9, add_33, weight, bias, output_shapes = _validate_inputs(inputs)
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=addmm_9.device,
        dtype=torch.float16,
    )
    _reformer_layernorm_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        addmm_9,
        add_33,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_m=BLOCK_M,
        block_h=BLOCK_H,
        num_warps=1,
        num_stages=1,
    )

    return (
        base.view(output_shapes[0]),
        base.view(output_shapes[1]),
    )


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
