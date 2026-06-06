"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full two-output SiLU/backward pointwise scope in one Triton kernel, preserves the returned transpose views, and shares `1 / (1 + exp(-x))` across both output expressions, whereas Inductor already emits one fused streaming pointwise kernel for the same scope and lands within measurement noise of this hand-written version; Inductor cannot materially improve this case today because there is no remaining launch or fusion gap after the views are represented as output reinterpretations, so the fix is BANDWIDTH_BOUND: no scheduler/codegen change is indicated beyond ordinary scalar cleanup that does not move the measured floor."""
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
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _silu_pair_kernel(
        mm_ptr,
        gate_ptr,
        grad_ptr,
        out0_ptr,
        out1_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        gate = tl.load(gate_ptr + offsets).to(tl.float32)
        grad = tl.load(grad_ptr + offsets).to(tl.float32)

        sigmoid = 1.0 / (1.0 + tl.exp(-gate))
        silu = gate * sigmoid
        out0 = mm * silu

        derivative = sigmoid * (1.0 + gate * (1.0 - sigmoid))
        out1 = (mm * grad) * derivative

        tl.store(out0_ptr + offsets, out0)
        tl.store(out1_ptr + offsets, out1)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects eight inputs, got {len(inputs)}")

    mm_381, arg330_1, arg331_1, shape0, shape1, shape2, shape3, shape4 = inputs
    tensors = (mm_381, arg330_1, arg331_1)
    for idx, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{REPRO_ID} input {idx} must be a tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{REPRO_ID} input {idx} must be CUDA")
        if tensor.dtype is not torch.bfloat16:
            raise ValueError(f"{REPRO_ID} input {idx} must be torch.bfloat16, got {tensor.dtype}")
        if tensor.ndim != 2:
            raise ValueError(f"{REPRO_ID} input {idx} must be rank 2, got shape={tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{REPRO_ID} input {idx} must be contiguous, got stride={tuple(tensor.stride())}")

    input_shape = tuple(int(dim) for dim in mm_381.shape)
    for idx, tensor in enumerate((arg330_1, arg331_1), start=1):
        if tuple(int(dim) for dim in tensor.shape) != input_shape:
            raise ValueError(
                f"{REPRO_ID} input {idx} shape={tuple(tensor.shape)} does not match {input_shape}"
            )

    view_shapes = (_shape_tuple(shape0), _shape_tuple(shape1), _shape_tuple(shape2))
    input_numel = int(mm_381.numel())
    for idx, view_shape in enumerate(view_shapes):
        if _numel(view_shape) != input_numel:
            raise ValueError(f"{REPRO_ID} view shape {idx}={view_shape} has wrong numel")

    out0_base_shape = _shape_tuple(shape3)
    out1_base_shape = _shape_tuple(shape4)
    if len(out0_base_shape) != 2 or len(out1_base_shape) != 2:
        raise ValueError(f"{REPRO_ID} output base shapes must be rank 2")
    if out0_base_shape != input_shape or out1_base_shape != input_shape:
        raise ValueError(
            f"{REPRO_ID} output base shapes {out0_base_shape}, {out1_base_shape} "
            f"must match input shape {input_shape}"
        )

    return mm_381, arg330_1, arg331_1, out0_base_shape, out1_base_shape


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

    mm_381, arg330_1, arg331_1, out0_base_shape, out1_base_shape = _validate_inputs(inputs)
    out0_base = torch.empty_strided(
        out0_base_shape,
        _contiguous_stride(out0_base_shape),
        device=mm_381.device,
        dtype=mm_381.dtype,
    )
    out1_base = torch.empty_strided(
        out1_base_shape,
        _contiguous_stride(out1_base_shape),
        device=mm_381.device,
        dtype=mm_381.dtype,
    )

    n_elements = int(mm_381.numel())
    if n_elements % 256 != 0:
        raise ValueError(f"{REPRO_ID} expects numel divisible by 256, got {n_elements}")
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _silu_pair_kernel[grid](
        mm_381,
        arg330_1,
        arg331_1,
        out0_base,
        out1_base,
        N=n_elements,
    )
    return (out0_base.permute(1, 0), out1_base.permute(1, 0))


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
