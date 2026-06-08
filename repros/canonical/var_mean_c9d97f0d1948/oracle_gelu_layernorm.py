"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete exact-erf GELU plus population LayerNorm scope in one Triton row-reduction kernel, whereas Inductor currently lowers the decomposed GELU, var_mean(correction=0), rsqrt, affine, and view graph through its generic fused normalization reduction schedule; Inductor cannot do this today because the scheduler/codegen normalization template does not have a guarded exact-GELU-producer LayerNorm specialization that keeps the activation tile through the statistics and affine epilogue for fixed hidden sizes; the fix is SCHEDULER_FUSION: add an exact-GELU-producing LayerNorm template that fuses the pointwise producer, row moments, epsilon rsqrt, affine epilogue, and final viewed store."""
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


RSQRT2 = 0.7071067811865476
EPS = 1.0e-5


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _gelu_layernorm_kernel(
        addmm_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        rsqrt2: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        mask = (row_offsets[:, None] < rows) & (cols[None, :] < hidden)
        offsets = row_offsets[:, None] * hidden + cols[None, :]

        x = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erf_term = tl.math.erf(x * rsqrt2) + 1.0
        gelu = (x * 0.5) * erf_term
        valid_cols = cols[None, :] < hidden
        gelu_for_reduce = tl.where(valid_cols, gelu, 0.0)

        mean = tl.sum(gelu_for_reduce, axis=1) / hidden
        centered = gelu - mean[:, None]
        sq = tl.where(valid_cols, centered * centered, 0.0)
        variance = tl.sum(sq, axis=1) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=cols < hidden, other=0.0).to(tl.float32)
        out = (centered * invstd[:, None]) * weight[None, :] + bias[None, :]

        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _block_m_for_hidden(hidden: int) -> int:
    if hidden <= 128:
        return 4
    return 1


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int], int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm, weight, bias, view_shape, out_shape_value = inputs
    tensor_inputs = (addmm, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")

    if addmm.ndim != 2:
        raise ValueError(f"input 0 must be rank 2, got shape {tuple(addmm.shape)}")
    rows = int(addmm.shape[0])
    hidden = int(addmm.shape[1])
    if hidden <= 0 or rows <= 0:
        raise ValueError(f"unexpected addmm shape {tuple(addmm.shape)}")
    if hidden > 2048:
        raise ValueError(f"{REPRO_ID} supports hidden size up to 2048, got {hidden}")

    expected_vector_shape = (hidden,)
    if tuple(weight.shape) != expected_vector_shape:
        raise ValueError(f"input 1 shape {tuple(weight.shape)} != {expected_vector_shape}")
    if tuple(bias.shape) != expected_vector_shape:
        raise ValueError(f"input 2 shape {tuple(bias.shape)} != {expected_vector_shape}")

    for index, value in enumerate(tensor_inputs):
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    view_shape_tuple = _shape_tuple(view_shape)
    output_shape = _shape_tuple(out_shape_value)
    if len(view_shape_tuple) < 1 or view_shape_tuple[-1] != hidden:
        raise ValueError(f"unexpected input view shape parameter: {view_shape!r}")
    if _numel(view_shape_tuple) != rows * hidden:
        raise ValueError(
            f"input view shape {view_shape_tuple} does not match addmm shape {tuple(addmm.shape)}"
        )
    if output_shape != (rows, hidden):
        raise ValueError(f"unexpected output view shape parameter: {out_shape_value!r}")

    return addmm, weight, bias, output_shape, rows, hidden


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
        raise RuntimeError("Triton is required for oracle_gelu_layernorm.py")

    addmm, weight, bias, output_shape, rows, hidden = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        (hidden, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    block_n = _next_power_of_2(hidden)
    block_m = _block_m_for_hidden(hidden)
    _gelu_layernorm_kernel[(triton.cdiv(rows, block_m),)](
        addmm,
        weight,
        bias,
        output,
        rows=rows,
        hidden=hidden,
        eps=EPS,
        rsqrt2=RSQRT2,
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        num_warps=8 if block_n >= 2048 else 4,
        num_stages=3,
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
