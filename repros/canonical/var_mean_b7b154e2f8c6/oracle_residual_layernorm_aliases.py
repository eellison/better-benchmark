"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Longformer residual LayerNorm alias scope in one shape-specialized Triton row kernel, including the `[4096,768] -> [1,4096,768]` view, fp16-rounded residual add, fp32 population `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 affine epilogue, fp16 cast, singleton-dim permute, and all three final `[4096,768]` aliasing view outputs, whereas tuned Inductor already lowers the same fixed-hidden normalization region to the canonical norm-template schedule plus metadata-only view returns; Inductor cannot materially improve this case through a local scheduler-fusion, split-K, algebraic, recompute, or scatter rewrite because the remaining full-scope work is the mandatory activation/residual/affine reads, one 768-wide row reduction, rsqrt, fp16 output store, and view metadata rather than avoidable intermediate traffic; the fix is BANDWIDTH_BOUND: record this as an at-floor norm-template-canonicalization repro unless broader LayerNorm codegen or launch-overhead work moves the family."""
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

ROWS = 4096
HIDDEN = 768
INPUT_SHAPE = (ROWS, HIDDEN)
BASE_SHAPE = (1, ROWS, HIDDEN)
BASE_STRIDE = (ROWS * HIDDEN, HIDDEN, 1)
OUTPUT_SHAPE = (ROWS, HIDDEN)
EPS = 1.0e-5
BLOCK_H = 1024
DEFAULT_NUM_WARPS = 4
CLASSIFICATION = "BANDWIDTH_BOUND"

if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
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

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

        # aten.add on fp16 inputs rounds to fp16 before the explicit fp32 cast.
        summed = (addmm + residual).to(tl.float16).to(tl.float32)
        summed = tl.where(mask, summed, 0.0)

        mean = tl.sum(summed, axis=0) / hidden
        centered = tl.where(mask, summed - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = centered * invstd * weight + bias
        tl.store(out_ptr + offsets, out.to(tl.float16), mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"shape parameter {value!r} is not iterable") from exc


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    addmm_65, residual, weight, bias, base_shape, shape1, shape2, shape3 = inputs
    tensor_inputs = (addmm_65, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, BASE_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if value.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if len({value.device for value in tensor_inputs}) != 1:
        raise ValueError("all tensor inputs must be on the same device")

    if _shape_tuple(base_shape) != BASE_SHAPE:
        raise ValueError(f"unexpected input view shape parameter: {base_shape!r}")

    output_shapes = (_shape_tuple(shape1), _shape_tuple(shape2), _shape_tuple(shape3))
    for index, shape in enumerate(output_shapes, start=1):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"unexpected output shape parameter {index}: {shape!r}")

    return addmm_65, residual, weight, bias, output_shapes


@oracle_impl(hardware="H100", shapes="(T([4096, 768], f16), T([1, 4096, 768], f16), T([768], f16), T([768], f16), S([1, 4096, 768]), S([4096, 768]), S([4096, 768]), S([4096, 768]))")
def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same three fp16 [4096,768] view outputs. The returned views alias one
    [1,4096,768] base tensor and preserve the captured permute/view metadata.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_layernorm_aliases.py")

    addmm_65, residual, weight, bias, output_shapes = _validate_inputs(inputs)
    base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=addmm_65.device,
        dtype=torch.float16,
    )
    _residual_layernorm_kernel[(ROWS,)](
        addmm_65,
        residual,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=4,
    )

    permuted = base.permute(1, 0, 2)
    return tuple(permuted.view(shape) for shape in output_shapes)


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
