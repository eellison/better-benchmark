"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete metadata-view plus three-add BART pointwise scope as one storage-linear Triton kernel over the contiguous inputs and fresh contiguous output, whereas tuned Inductor already has the same full-scope single pointwise memory-traffic envelope for the captured view/add/add/add graph; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a narrower new pattern because the remaining work is the required four fp32 input reads, three fp32 adds, and one fp32 output store; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise bandwidth or launch-overhead work moves both implementations."""
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 256}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 512}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 1024}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 2048}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _flat_add_views_kernel(
        mm_6_ptr,
        mul_22_ptr,
        mm_8_ptr,
        mm_10_ptr,
        out_ptr,
        N: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        mask = offsets < N
        base = tl.load(mul_22_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add0 = tl.load(mm_6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add1 = tl.load(mm_8_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add2 = tl.load(mm_10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        result = ((base + add0) + add1) + add2
        tl.store(out_ptr + offsets, result, mask=mask)


def _shape_tuple(name: str, value: Any) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc
    if len(shape) != 3:
        raise ValueError(f"{name} must be the captured 3D view shape, got {shape}")
    if any(dim < 0 for dim in shape):
        raise ValueError(f"{name} contains a negative dimension: {shape}")
    return shape


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    stride = 1
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _require_contiguous_f32_cuda(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride {tuple(value.stride())}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_flat_add_views.py")
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    mm_6 = _require_contiguous_f32_cuda("mm_6", inputs[0])
    mul_22 = _require_contiguous_f32_cuda("mul_22", inputs[1])
    mm_8 = _require_contiguous_f32_cuda("mm_8", inputs[2])
    mm_10 = _require_contiguous_f32_cuda("mm_10", inputs[3])
    view_shape = _shape_tuple("_shape_param_0", inputs[4])
    view_shape_1 = _shape_tuple("_shape_param_1", inputs[5])
    view_shape_2 = _shape_tuple("_shape_param_2", inputs[6])

    if view_shape != view_shape_1 or view_shape != view_shape_2:
        raise ValueError(f"shape params must match, got {view_shape}, {view_shape_1}, {view_shape_2}")
    if tuple(mul_22.shape) != view_shape:
        raise ValueError(f"mul_22 has shape {tuple(mul_22.shape)}, expected {view_shape}")

    n_elements = _numel(view_shape)
    for name, tensor in (("mm_6", mm_6), ("mm_8", mm_8), ("mm_10", mm_10)):
        if tensor.numel() != n_elements:
            raise ValueError(f"{name} has {tensor.numel()} elements, expected {n_elements}")
        if tensor.device != mul_22.device:
            raise RuntimeError(f"{name} is on {tensor.device}, expected {mul_22.device}")

    return mm_6, mul_22, mm_8, mm_10, view_shape


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
    mm_6, mul_22, mm_8, mm_10, view_shape = _validate_inputs(inputs)
    n_elements = _numel(view_shape)
    output = torch.empty_strided(
        view_shape,
        _contiguous_strides(view_shape),
        device=mul_22.device,
        dtype=mul_22.dtype,
    )
    grid = lambda meta: (triton.cdiv(n_elements, meta["XBLOCK"]),)
    _flat_add_views_kernel[grid](
        mm_6,
        mul_22,
        mm_8,
        mm_10,
        output,
        N=n_elements,
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
