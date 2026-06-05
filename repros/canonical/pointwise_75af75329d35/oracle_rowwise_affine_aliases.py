"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileBERT hidden-dimension affine scope, including both metadata-only views and the duplicate aliased flattened returns, by writing one contiguous `[256,128,128]` backing tensor with a row-tiled Triton kernel that reuses the `[128]` scale and bias vectors across rows, whereas Inductor's compiled path already reaches the same CUDAGraph replay floor for the generic broadcast multiply/add materialization; Inductor cannot materially reduce this local region because the returned aliased views still require one fresh contiguous f32 `[32768,128]` result backed by the full input read and output store; the fix is BANDWIDTH_BOUND: keep this as an at-floor pointwise materialization unless broader allocation, launch, or memory-bandwidth improvements move both implementations."""
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


ROWS = 32768
COLS = 128
VIEW_SHAPE = (256, 128, 128)
OUTPUT_SHAPE = (ROWS, COLS)
VIEW_STRIDE = (128 * COLS, COLS, 1)
OUTPUT_STRIDE = (COLS, 1)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _mul_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _add_rn_f32(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=4),
        ],
        key=["M", "N"],
    )
    @triton.jit
    def _rowwise_affine_alias_kernel(
        x_ptr,
        scale_ptr,
        bias_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, N)
        offsets = rows[:, None] * N + cols[None, :]
        mask = rows[:, None] < M

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + cols)[None, :]
        bias = tl.load(bias_ptr + cols)[None, :]
        y = _add_rn_f32(_mul_rn_f32(x, scale), bias)
        tl.store(output_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_rowwise_affine_aliases.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_346, arg1073_1, arg1074_1, shape0, shape1, shape2 = inputs
    for name, tensor, shape in (
        ("addmm_346", addmm_346, OUTPUT_SHAPE),
        ("arg1073_1", arg1073_1, (COLS,)),
        ("arg1074_1", arg1074_1, (COLS,)),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if tensor.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be f32, got {tensor.dtype}")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tuple(tensor.stride())}")

    view_shape = _shape_tuple(shape0)
    output_shape_1 = _shape_tuple(shape1)
    output_shape_2 = _shape_tuple(shape2)
    if view_shape != VIEW_SHAPE:
        raise ValueError(f"unexpected first view shape: {view_shape}")
    if output_shape_1 != OUTPUT_SHAPE:
        raise ValueError(f"unexpected first output view shape: {output_shape_1}")
    if output_shape_2 != OUTPUT_SHAPE:
        raise ValueError(f"unexpected second output view shape: {output_shape_2}")

    return addmm_346, arg1073_1, arg1074_1, shape0, shape1, shape2


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
    addmm_346, arg1073_1, arg1074_1, shape0, shape1, shape2 = _validate_inputs(inputs)
    output_base = torch.empty_strided(
        _shape_tuple(shape0),
        VIEW_STRIDE,
        device=addmm_346.device,
        dtype=addmm_346.dtype,
    )

    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _rowwise_affine_alias_kernel[grid](
        addmm_346,
        arg1073_1,
        arg1074_1,
        output_base,
        M=ROWS,
        N=COLS,
    )
    return (
        torch.ops.aten.view.default(output_base, shape1),
        torch.ops.aten.view.default(output_base, shape2),
    )


def _check_layout_alias_and_exact_values(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    all_ok = True
    for i, (expected, observed) in enumerate(zip(eager, actual)):
        layout_ok = (
            tuple(observed.shape) == tuple(expected.shape)
            and tuple(observed.stride()) == tuple(expected.stride())
            and observed.dtype == expected.dtype
            and observed.storage_offset() == expected.storage_offset()
        )
        exact_ok = torch.equal(expected, observed)
        print(
            f"  output {i} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(observed.shape)} stride={list(observed.stride())} "
            f"dtype={observed.dtype} storage_offset={observed.storage_offset()})"
        )
        print(f"  output {i} exact: {'PASS' if exact_ok else 'FAIL'}")
        all_ok = all_ok and layout_ok and exact_ok

    expected_alias = eager[0].untyped_storage().data_ptr() == eager[1].untyped_storage().data_ptr()
    observed_alias = actual[0].untyped_storage().data_ptr() == actual[1].untyped_storage().data_ptr()
    alias_ok = observed_alias == expected_alias
    print(f"  output aliasing: {'PASS' if alias_ok else 'FAIL'} (aliased={observed_alias})")
    return all_ok and alias_ok


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
        ok = _check_layout_alias_and_exact_values(instance, inputs) and ok
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
