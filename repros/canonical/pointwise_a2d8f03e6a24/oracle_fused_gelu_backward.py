"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MT5 dropout-mask plus approximate-GELU backward pointwise scope in one Triton kernel, materializing the two contiguous `[4096, 1024]` backing tensors and returning the same transposed `[1024, 4096]` views as eager, whereas Inductor already lowers the same full multi-output tanh-GELU pointwise graph into the same graph-captured launch and memory/math envelope; Inductor cannot materially improve this local scope today because the remaining work is the required four input streams, two output streams, tanh-heavy GELU math, and returned view metadata rather than a missed stencil/layout fusion; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise transcendental codegen, memory bandwidth, or launch-overhead work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _fused_gelu_backward_kernel(
        mm_ptr,
        mask_ptr,
        gelu_input_ptr,
        grad_ptr,
        out0_ptr,
        out1_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_bounds = offsets < N

        mm = tl.load(mm_ptr + offsets, mask=in_bounds, other=0.0)
        mask = tl.load(mask_ptr + offsets, mask=in_bounds, other=0).to(tl.float32)
        x = tl.load(gelu_input_ptr + offsets, mask=in_bounds, other=0.0)
        grad = tl.load(grad_ptr + offsets, mask=in_bounds, other=0.0)

        dropout_mm = mm * mask * 1.1111111111111112
        x2 = x * x
        x3 = x2 * x
        tanh_arg = (x + 0.044715 * x3) * 0.7978845608028654
        tanh_val = libdevice.tanh(tanh_arg)
        tanh_plus_one = tanh_val + 1.0

        gelu = (x * 0.5) * tanh_plus_one
        out0 = dropout_mm * gelu

        upstream = dropout_mm * grad
        one_minus_tanh2 = 1.0 - tanh_val * tanh_val
        inner_derivative = 1.0 + (3.0 * 0.044715) * x2
        gelu_derivative = (
            0.5 * tanh_plus_one
            + (0.5 * x) * one_minus_tanh2 * 0.7978845608028654 * inner_derivative
        )
        out1 = upstream * gelu_derivative

        tl.store(out0_ptr + offsets, out0, mask=in_bounds)
        tl.store(out1_ptr + offsets, out1, mask=in_bounds)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known
    resolved = tuple(dims)
    if _numel(resolved) != numel:
        raise ValueError(f"shape {resolved} has {_numel(resolved)} elements, expected {numel}")
    return resolved


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int], int]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects nine inputs, got {len(inputs)}")

    (
        mm_277,
        arg207_1,
        arg205_1,
        arg206_1,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    tensor_inputs = (mm_277, arg207_1, arg205_1, arg206_1)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensor_inputs):
        raise TypeError(f"{REPRO_ID} expects the first four inputs to be tensors")
    if not all(tensor.is_cuda for tensor in tensor_inputs):
        raise ValueError(f"{REPRO_ID} expects CUDA tensor inputs")
    if mm_277.dtype is not torch.float32 or arg205_1.dtype is not torch.float32 or arg206_1.dtype is not torch.float32:
        raise ValueError(
            f"{REPRO_ID} expects float32 data tensors, got "
            f"{mm_277.dtype}, {arg205_1.dtype}, {arg206_1.dtype}"
        )
    if arg207_1.dtype is not torch.bool:
        raise ValueError(f"{REPRO_ID} expects a bool mask tensor, got {arg207_1.dtype}")
    if not all(tensor.is_contiguous() for tensor in tensor_inputs):
        strides = [tuple(tensor.stride()) for tensor in tensor_inputs]
        raise ValueError(f"{REPRO_ID} expects contiguous tensor inputs, got strides={strides}")

    n_elements = int(mm_277.numel())
    if int(arg207_1.numel()) != n_elements or int(arg205_1.numel()) != n_elements or int(arg206_1.numel()) != n_elements:
        raise ValueError(f"{REPRO_ID} expects all tensor inputs to have the same numel")

    expected_3d = _resolve_view_shape(shape0, n_elements)
    for shape_value in (shape1, shape2):
        if _resolve_view_shape(shape_value, n_elements) != expected_3d:
            raise ValueError(f"{REPRO_ID} expects all rank-3 view shapes to match {expected_3d}")

    output_base_shape0 = _resolve_view_shape(shape3, n_elements)
    output_base_shape1 = _resolve_view_shape(shape4, n_elements)
    if output_base_shape0 != tuple(mm_277.shape) or output_base_shape1 != tuple(mm_277.shape):
        raise ValueError(
            f"{REPRO_ID} output base shapes {output_base_shape0}, {output_base_shape1} "
            f"do not match input base shape {tuple(mm_277.shape)}"
        )
    if len(output_base_shape0) != 2:
        raise ValueError(f"{REPRO_ID} expects rank-2 output bases, got {output_base_shape0}")

    return mm_277, arg207_1, arg205_1, arg206_1, output_base_shape0, output_base_shape1, n_elements


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
    if triton is None or libdevice is None:
        raise RuntimeError("Triton is required for this oracle")

    mm_277, arg207_1, arg205_1, arg206_1, out0_shape, out1_shape, n_elements = _validate_inputs(inputs)
    out0_base = torch.empty(out0_shape, device=mm_277.device, dtype=mm_277.dtype)
    out1_base = torch.empty(out1_shape, device=mm_277.device, dtype=mm_277.dtype)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fused_gelu_backward_kernel[grid](
        mm_277,
        arg207_1,
        arg205_1,
        arg206_1,
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
