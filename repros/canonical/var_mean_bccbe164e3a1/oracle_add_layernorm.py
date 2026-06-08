"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete add-plus-affine LayerNorm scope in one shape-specialized Triton row kernel, whereas Inductor currently emits one fused generic Welford-style reduction kernel for the same view/add/var_mean/rsqrt/affine/view graph; Inductor cannot do this today because its correction=0 var_mean lowering keeps generic Welford bookkeeping instead of selecting fixed-width LayerNorm algebra that reuses the resident row tile for the affine epilogue; the fix is ALGEBRAIC_ELIMINATION: add a guarded correction=0 LayerNorm lowering that replaces generic Welford state updates with direct mean and centered-variance reductions for static hidden dimensions."""
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


EPS = 1.0e-12


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _next_power_of_2(value: int) -> int:
    if value < 1:
        raise ValueError(f"expected positive hidden dimension, got {value}")
    return 1 << (value - 1).bit_length()


def _num_warps(block_n: int) -> int:
    if block_n >= 2048:
        return 8
    if block_n >= 512:
        return 4
    return 1


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int, int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_70, add_103, arg18_1, arg19_1, shape0, shape1 = inputs
    for index, value in enumerate((addmm_70, add_103, arg18_1, arg19_1)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"input {index} must be a tensor, got {type(value).__name__}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if addmm_70.dim() != 2:
        raise ValueError(f"addmm_70 must be rank-2, got shape {tuple(addmm_70.shape)}")
    if add_103.dim() != 3:
        raise ValueError(f"add_103 must be rank-3, got shape {tuple(add_103.shape)}")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    if tuple(add_103.shape) != shape0_tuple:
        raise ValueError(f"input view shape parameter {shape0_tuple} does not match add_103")

    rows = int(addmm_70.shape[0])
    hidden = int(addmm_70.shape[1])
    if hidden != int(add_103.shape[-1]):
        raise ValueError(f"hidden mismatch: addmm={hidden}, add_103={int(add_103.shape[-1])}")
    if add_103.numel() != addmm_70.numel():
        raise ValueError("addmm_70 and add_103 must have the same number of elements")
    if rows * hidden != addmm_70.numel():
        raise ValueError("addmm_70 must be a contiguous [rows, hidden] tensor")
    if tuple(arg18_1.shape) != (hidden,) or tuple(arg19_1.shape) != (hidden,):
        raise ValueError("LayerNorm weight and bias must both have shape [hidden]")
    if any(value.device != addmm_70.device for value in (add_103, arg18_1, arg19_1)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    view_numel = 1
    inferred_dims = 0
    for dim in shape1_tuple:
        if dim == -1:
            inferred_dims += 1
            continue
        view_numel *= dim
    if inferred_dims > 1 or (inferred_dims == 0 and view_numel != addmm_70.numel()):
        raise ValueError(f"output view shape parameter {shape1_tuple} is incompatible")
    if inferred_dims == 1 and addmm_70.numel() % view_numel != 0:
        raise ValueError(f"output view shape parameter {shape1_tuple} is incompatible")

    return addmm_70, add_103, arg18_1, arg19_1, shape1_tuple, rows, hidden


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def oracle_kernel(
        addmm_ptr,
        add_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = addmm + residual

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        square = centered * centered
        variance = tl.sum(tl.where(mask, square, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        normalized = centered * invstd
        out = normalized * weight + bias
        tl.store(out_ptr + offsets, out, mask=mask)


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

    addmm_70, add_103, arg18_1, arg19_1, output_shape, rows, hidden = _validate_inputs(inputs)
    output = torch.empty_like(addmm_70)
    block_n = _next_power_of_2(hidden)
    oracle_kernel[(rows,)](
        addmm_70,
        add_103,
        arg18_1,
        arg19_1,
        output,
        hidden=hidden,
        eps=EPS,
        BLOCK_N=block_n,
        num_warps=_num_warps(block_n),
        num_stages=3,
    )
    return output.view(output_shape)


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
