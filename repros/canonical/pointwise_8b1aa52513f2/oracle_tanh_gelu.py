"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured tanh-approximate GELU scope as one storage-linear Triton pointwise kernel, including the metadata-only `[M, N] -> [B, S, N] -> [M, N]` views, fp32 `0.5*x*(1+tanh(0.7978845608028654*(x+0.044715*x^3)))` computation, dtype-preserving store, and final contiguous output view, whereas Inductor already lowers the same full view-plus-pointwise graph as one fused flattened pointwise kernel; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or layout/stencil fusion because the remaining work is the required input read, tanh-heavy math, output write, and graph-captured launch; the fix is BANDWIDTH_BOUND: record this as a pointwise math/memory floor unless broader pointwise transcendental codegen or launch-overhead improvements move both paths."""
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

CLASSIFICATION = "BANDWIDTH_BOUND"


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
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        ],
        key=["n_elements"],
    )
    @triton.jit
    def _tanh_gelu_kernel(
        input_ptr,
        output_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        cubic = x * x * x
        tanh_arg = (x + cubic * 0.044715) * 0.7978845608028654
        y = (x * 0.5) * (libdevice.tanh(tanh_arg) + 1.0)
        tl.store(output_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"{REPRO_ID} expected a shape-list argument, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(shape_value: Any, numel: int) -> tuple[int, ...]:
    shape = list(_shape_tuple(shape_value))
    infer_index = None
    known_product = 1
    for index, dim in enumerate(shape):
        if dim == -1:
            if infer_index is not None:
                raise ValueError(f"{REPRO_ID} view shape has multiple inferred dims: {shape}")
            infer_index = index
        elif dim < 0:
            raise ValueError(f"{REPRO_ID} view shape has invalid negative dim: {shape}")
        else:
            known_product *= dim

    if infer_index is not None:
        if known_product == 0 or numel % known_product != 0:
            raise ValueError(f"{REPRO_ID} view shape {shape} is incompatible with {numel} elements")
        shape[infer_index] = numel // known_product
    elif known_product != numel:
        raise ValueError(f"{REPRO_ID} view shape {shape} is incompatible with {numel} elements")

    return tuple(shape)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, tuple[int, ...]]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_tanh_gelu.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    x, shape0, shape1 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor, got {type(x)!r}")
    if x.device.type != "cuda":
        raise ValueError(f"{REPRO_ID} expects CUDA input")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{REPRO_ID} expects a floating input, got {x.dtype}")
    if x.dim() != 2:
        raise ValueError(f"{REPRO_ID} expects a 2D input tensor, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layout")

    view_shape = _resolve_view_shape(shape0, x.numel())
    out_shape = _resolve_view_shape(shape1, x.numel())
    if len(view_shape) != 3:
        raise ValueError(f"{REPRO_ID} expects a 3D intermediate view shape, got {view_shape}")
    if out_shape != tuple(x.shape):
        raise ValueError(
            f"{REPRO_ID} output view shape {out_shape} must match input shape {tuple(x.shape)}"
        )

    return x, out_shape


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward tanh-approximate GELU scope."""
    x, out_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=x.device,
        dtype=x.dtype,
    )
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _tanh_gelu_kernel[grid](x, output, n_elements=n_elements)
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
