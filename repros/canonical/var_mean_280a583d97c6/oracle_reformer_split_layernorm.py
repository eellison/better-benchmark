"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Reformer split-layernorm scope in one shape-specialized Triton row kernel plus the required duplicated-input split view, including the expand/contiguous-clone value semantics, fp32 hidden-size-256 var_mean LayerNorm with eps=1e-12, fp16 affine output, three sibling `[4096,256]` views over one result buffer, and the fourth `[1,4096,256]` strided view over the second duplicated half, whereas Inductor lowers the decomposed expand/clone/view/split/clone/var_mean/affine/view graph through generic normalization fusion and separate view-producing buffer work; Inductor cannot emit this exact full-scope floor today because normalization pattern canonicalization does not recognize this fixed duplicated-split producer with aliasing sibling returns as one semantic Reformer split-layernorm template; the fix is NEW_PATTERN: add a split-layernorm lowering that folds the duplicated producer into a fixed-hidden row LayerNorm kernel while preserving the required second split view layout."""
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


ROWS = 4096
HIDDEN = 256
EXPANDED = 2
EPS = 1.0e-12
OUTPUT_SHAPE = (1, ROWS, HIDDEN)
OUTPUT_STRIDE = (ROWS * HIDDEN, HIDDEN, 1)
DUP_SHAPE = (1, ROWS, EXPANDED, HIDDEN)
DUP_STRIDE = (ROWS * EXPANDED * HIDDEN, EXPANDED * HIDDEN, HIDDEN, 1)
CLASSIFICATION = "NEW_PATTERN"

if triton is not None:

    @triton.jit
    def _split_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        dup_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden

        x_offsets = row * hidden + cols
        x_f16 = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
        x = x_f16.to(tl.float32)
        x = tl.where(mask, x, 0.0)

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + x_offsets, y.to(tl.float16), mask=mask)
        tl.store(dup_ptr + row * (2 * hidden) + hidden + cols, x_f16, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1, shape2, shape3, shape4 = inputs
    tensor_inputs = (x, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")

    expected_shapes = ((1, ROWS, HIDDEN), (HIDDEN,), (HIDDEN,))
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape0) != DUP_SHAPE:
        raise ValueError(f"unexpected expand shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != (1, ROWS, EXPANDED * HIDDEN):
        raise ValueError(f"unexpected flattened split shape parameter: {shape1!r}")
    for index, shape in enumerate((shape2, shape3, shape4), start=2):
        if _shape_tuple(shape) != (ROWS, HIDDEN):
            raise ValueError(f"unexpected output shape parameter {index}: {shape!r}")

    return x, weight, bias


def oracle_forward(inputs):
    """Run the full Repro.forward split + LayerNorm computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_reformer_split_layernorm.py")

    x, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float16,
    )
    dup = torch.empty_strided(
        DUP_SHAPE,
        DUP_STRIDE,
        device=x.device,
        dtype=torch.float16,
    )
    _split_layernorm_kernel[(ROWS,)](
        x,
        weight,
        bias,
        out,
        dup,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=triton.next_power_of_2(HIDDEN),
        num_warps=8,
    )

    split_1 = dup.view(1, ROWS, EXPANDED * HIDDEN).split(HIDDEN, -1)[1]
    return (
        out.view(ROWS, HIDDEN),
        out.view(ROWS, HIDDEN),
        out.view(ROWS, HIDDEN),
        split_1,
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
