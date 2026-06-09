"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ALBERT tanh-GELU backward pointwise scope in one storage-linear Triton kernel, including the metadata-only `[4096, 16384] -> [8, 512, 16384] -> [4096, 16384]` views and the returned non-contiguous `[16384, 4096]` transpose layout, whereas Inductor already lowers this isolated expression and final permute view to the same required two f32 input streams, tanh-heavy pointwise math, and one f32 output stream; Inductor cannot materially do less work for this captured region because there is no reduction, scatter, producer-consumer fusion, removable materialization, or safe local algebraic rewrite that beats the required tanh and memory traffic; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope pointwise case unless broader pointwise math-codegen or tanh lowering improvements move both implementations."""
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


ROWS = 4096
COLS = 16384
VIEW_SHAPE = (8, 512, COLS)
FLAT_SHAPE = (ROWS, COLS)
OUT_SHAPE = (COLS, ROWS)
OUT_STRIDE = (1, COLS)
N_ELEMENTS = ROWS * COLS


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
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=4),
        ],
        key=["N"],
    )
    @triton.jit
    def _gelu_backward_transpose_kernel(
        grad_ptr,
        x_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mul_tensor = x * 0.5
        mul_tensor_1 = grad * mul_tensor
        pow_tensor_scalar = x * x * x
        mul_tensor_2 = pow_tensor_scalar * 0.044715
        add_tensor = x + mul_tensor_2
        mul_tensor_3 = add_tensor * 0.7978845608028654
        tanh_default = libdevice.tanh(mul_tensor_3)
        add_tensor_1 = tanh_default + 1.0
        mul_tensor_4 = grad * add_tensor_1
        mul_tensor_5 = tanh_default * tanh_default
        sub_tensor = 1.0 - mul_tensor_5
        mul_tensor_6 = mul_tensor_1 * sub_tensor
        mul_tensor_7 = mul_tensor_6 * 0.7978845608028654
        mul_tensor_8 = mul_tensor_7 * 0.044715
        pow_tensor_scalar_1 = x * x
        mul_scalar = pow_tensor_scalar_1 * 3.0
        mul_tensor_9 = mul_tensor_8 * mul_scalar
        add_tensor_2 = mul_tensor_7 + mul_tensor_9
        mul_tensor_10 = mul_tensor_4 * 0.5
        out = add_tensor_2 + mul_tensor_10

        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _expect_f32_contiguous(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if tuple(value.shape) != FLAT_SHAPE:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {FLAT_SHAPE}")
    if tuple(value.stride()) != (COLS, 1):
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {(COLS, 1)}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    grad, x, shape0, shape1, shape2 = inputs
    grad = _expect_f32_contiguous("mm_124", grad)
    x = _expect_f32_contiguous("arg32_1", x)

    if _shape_tuple(shape0) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_0 is {shape0!r}, expected {VIEW_SHAPE}")
    if _shape_tuple(shape1) != VIEW_SHAPE:
        raise ValueError(f"_shape_param_1 is {shape1!r}, expected {VIEW_SHAPE}")
    if _shape_tuple(shape2) != FLAT_SHAPE:
        raise ValueError(f"_shape_param_2 is {shape2!r}, expected {FLAT_SHAPE}")
    return grad, x


@oracle_impl(hardware="H100", shapes="(T([4096, 16384], f32), T([4096, 16384], f32), S([8, 512, 16384]), S([8, 512, 16384]), S([4096, 16384]))")
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
        raise RuntimeError("Triton is required for oracle_gelu_backward_transpose.py")

    grad, x = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=grad.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_N"]),)
    _gelu_backward_transpose_kernel[grid](
        grad,
        x,
        output,
        N=N_ELEMENTS,
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
