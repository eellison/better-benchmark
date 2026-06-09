"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the residual-add, eps=1e-6 fp32 population LayerNorm, and affine epilogue only for the two returned distilled-token rows while returning the same strided select views from one full logical storage, whereas Inductor currently normalizes and stores the full [128,198,768] residual tensor before selecting tokens 0 and 1; Inductor cannot do this today because the normalization scheduler does not sink constant token selects backward through row-independent reshape/add/var_mean/affine graphs to shrink the row domain; the fix is ALGEBRAIC_ELIMINATION: add a guarded select-sinking rewrite for row-local LayerNorm that narrows codegen to the live token rows and preserves select view metadata."""
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
BATCH = 128
TOKENS = 198
HIDDEN = 768
ROWS = BATCH * TOKENS
ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, TOKENS, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
EPS = 1.0e-6
BLOCK_H = 1024
TOKEN_BLOCK = 2

if triton is not None:

    @triton.jit
    def _selected_residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_storage_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        token_block: tl.constexpr,
    ):
        batch = tl.program_id(0)
        token_offsets = tl.arange(0, token_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        mask = cols < hidden

        rows = batch * tokens + token_offsets
        offsets = rows * hidden + cols

        x = (
            tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias
        tl.store(output_storage_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_47, add_80, weight, bias, shape_param = inputs
    tensor_inputs = (addmm_47, add_80, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (ADDMM_SHAPE, RESIDUAL_SHAPE, AFFINE_SHAPE, AFFINE_SHAPE)
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape_param) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected reshape parameter: {shape_param!r}")

    device = addmm_47.device
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm_47, add_80, weight, bias


@oracle_impl(hardware="H100", shapes="(T([25344, 768], f32), T([128, 198, 768], f32), T([768], f32), T([768], f32), S([128, 198, 768]))")
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
        raise RuntimeError("Triton is required for oracle_selected_residual_layernorm.py")

    addmm_47, add_80, weight, bias = _validate_inputs(inputs)
    output_storage = torch.empty_strided(
        RESIDUAL_SHAPE,
        (TOKENS * HIDDEN, HIDDEN, 1),
        device=addmm_47.device,
        dtype=torch.float32,
    )
    _selected_residual_layernorm_kernel[(BATCH,)](
        addmm_47,
        add_80,
        weight,
        bias,
        output_storage,
        tokens=TOKENS,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        token_block=TOKEN_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return output_storage[:, 0], output_storage[:, 1]


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
