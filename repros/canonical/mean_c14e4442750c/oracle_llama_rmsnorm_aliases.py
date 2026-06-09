"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full captured LLaMA fp32 residual-add RMSNorm path, including mm_52.view([32,32,512]), add_34, mean(square) over the hidden dimension with eps=1e-5, fp32 weight multiply, and two aliasing [1024,512] view outputs over one final buffer, whereas tuned Inductor already executes this full scope within the same row-reduction and output-store envelope; Inductor cannot materially improve this repro with a local fusion or canonicalization change because the remaining cost is the required residual/input/weight reads, hidden-dimension RMS reduction, and output writes; the fix is BANDWIDTH_BOUND: record this as an at-floor RMSNorm alias case rather than a new actionable pattern."""
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

ROWS = 1024
HIDDEN = 512
MM_SHAPE = (ROWS, HIDDEN)
ADD_SHAPE = (32, 32, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_COUNT = 2
ADD_STRIDE = (32 * HIDDEN, HIDDEN, 1)
EPS = 1.0e-5
BLOCK_H = 512

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

    @triton.jit
    def _llama_rmsnorm_kernel(
        mm_ptr,
        add_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(add_ptr + offsets, mask=mask, other=0.0)
        x = mm + residual

        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        inv_rms = tl.rsqrt(sum_sq / hidden + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        out = x * inv_rms * weight
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm_52, add_34, arg87_1, view_shape, *output_shapes = inputs
    tensor_inputs = (mm_52, add_34, arg87_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")

    expected_shapes = (MM_SHAPE, ADD_SHAPE, WEIGHT_SHAPE)
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if add_34.device != mm_52.device or arg87_1.device != mm_52.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    if _shape_tuple(view_shape) != ADD_SHAPE:
        raise ValueError(f"unexpected mm_52 view shape parameter: {view_shape!r}")
    if len(output_shapes) != OUTPUT_COUNT:
        raise ValueError(f"expected {OUTPUT_COUNT} output shape parameters, got {len(output_shapes)}")

    view_shapes = tuple(_shape_tuple(shape) for shape in output_shapes)
    for index, shape in enumerate(view_shapes, start=1):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"unexpected output shape parameter {index}: {shape!r}")

    return mm_52, add_34, arg87_1, view_shapes


@oracle_impl(hardware="H100", shapes="(T([1024, 512], f32), T([32, 32, 512], f32), T([512], f32), S([32, 32, 512]), S([1024, 512]), S([1024, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same two fp32 [1024,512] view outputs. The returned views alias one
    [32,32,512] base buffer, matching eager view alias/layout behavior.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_llama_rmsnorm_aliases.py")

    mm_52, add_34, arg87_1, view_shapes = _validate_inputs(inputs)
    base = torch.empty_strided(
        ADD_SHAPE,
        ADD_STRIDE,
        device=mm_52.device,
        dtype=torch.float32,
    )

    _llama_rmsnorm_kernel[(ROWS,)](
        mm_52,
        add_34,
        arg87_1,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
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
