"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DeBERTa attention softmax returned by Repro.forward, including the [192,512,512] to [8,24,512,512] view contract, elimination of the broadcast full(False) mask and min-float fill, stable last-dimension softmax, and final contiguous [192,512,512] view in one Triton row kernel, whereas Inductor lowers the decomposed full/where/amax/sub/exp/sum/div/view graph through generic producers around the reduction; Inductor cannot do this today because its scheduler/codegen does not canonicalize a broadcast constant-false where into the identity before softmax scheduling; the fix is ALGEBRAIC_ELIMINATION: add constant-mask predicate simplification that removes the dead fill/where producer and feeds the row-softmax template directly."""
from __future__ import annotations

import argparse
import math
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


def _resolve_view_shape(shape_param, numel: int) -> tuple[int, ...]:
    dims = [int(dim) for dim in shape_param]
    inferred = None
    known = 1
    for idx, dim in enumerate(dims):
        if dim == -1:
            if inferred is not None:
                raise ValueError(f"multiple inferred dimensions in shape {dims}")
            inferred = idx
        else:
            known *= dim

    if inferred is None:
        if known != numel:
            raise ValueError(f"shape {dims} has {known} elements, expected {numel}")
    else:
        if known == 0 or numel % known != 0:
            raise ValueError(f"shape {dims} cannot view {numel} elements")
        dims[inferred] = numel // known

    return tuple(dims)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


if triton is not None:

    @triton.jit
    def _softmax_rows_kernel(
        input_ptr,
        output_ptr,
        n_cols: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        offsets = row * n_cols + cols

        scores = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(scores, axis=0)
        numer = tl.exp(scores - row_max)
        denom = tl.sum(numer, axis=0)
        out = numer / denom

        tl.store(output_ptr + offsets, out, mask=mask)


def _validate_inputs(
    bmm_46: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not bmm_46.is_cuda:
        raise RuntimeError("CUDA tensor is required")
    if bmm_46.dtype != torch.float32:
        raise TypeError(f"expected fp32 input, got {bmm_46.dtype}")
    if not bmm_46.is_contiguous():
        raise ValueError("expected contiguous input matching repro make_inputs")

    numel = bmm_46.numel()
    view_shape = _resolve_view_shape(_shape_param_0, numel)
    out_shape = _resolve_view_shape(_shape_param_1, numel)
    if len(view_shape) < 1:
        raise ValueError(f"invalid softmax view shape {view_shape}")
    if math.prod(view_shape) != numel or math.prod(out_shape) != numel:
        raise ValueError("view shapes must preserve input element count")

    n_cols = int(view_shape[-1])
    if n_cols <= 0:
        raise ValueError(f"invalid softmax column count {n_cols}")
    n_rows = numel // n_cols
    if math.prod(view_shape[:-1]) != n_rows:
        raise ValueError(f"invalid row count for view shape {view_shape}")

    return view_shape, out_shape, n_rows, n_cols


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
    bmm_46, _shape_param_0, _shape_param_1 = inputs
    _, out_shape, n_rows, n_cols = _validate_inputs(
        bmm_46,
        _shape_param_0,
        _shape_param_1,
    )

    output = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=bmm_46.device,
        dtype=bmm_46.dtype,
    )
    block_n = _next_power_of_2(n_cols)
    num_warps = 4 if block_n <= 1024 else 8
    _softmax_rows_kernel[(n_rows,)](
        bmm_46,
        output,
        n_cols=n_cols,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
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
