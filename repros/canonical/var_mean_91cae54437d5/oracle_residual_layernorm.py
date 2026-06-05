"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GPT-J residual-add LayerNorm scope in one shape-specialized Triton row kernel, including both `[128,4096]` views, the third `[1,128,4096]` residual addend, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 normalization, affine scale/bias, and final `[128,4096]` view store using a sum/sum-square variance identity, whereas Inductor already fuses the full graph into one generic Welford reduction kernel that rereads the three addends for the affine epilogue; Inductor cannot do this today because `var_mean(correction=0)` lowering preserves the generic Welford/template schedule rather than selecting a fixed-width sum/sum-square LayerNorm specialization; the fix is ALGEBRAIC_ELIMINATION: add a guarded correction=0 normalization lowering that uses sum and sum-of-squares variance when its numerical contract is acceptable."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 128
HIDDEN = 4096
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (1, ROWS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
BLOCK_H = 4096
EPS = 1.0e-5
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
        mm_ptr,
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = mm + addmm + residual

        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        sum_x2 = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        mean = sum_x / hidden
        variance = sum_x2 / hidden - mean * mean
        variance = tl.maximum(variance, 0.0)
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    mm_111, addmm_55, add_218, weight, bias, shape0, shape1, shape2 = inputs
    tensor_inputs = (mm_111, addmm_55, add_218, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = [tuple(value.stride()) for value in tensor_inputs]
        raise ValueError(f"all tensor inputs must be contiguous, got strides={strides}")

    expected_shapes = (
        INPUT_SHAPE,
        INPUT_SHAPE,
        RESIDUAL_SHAPE,
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    shape2_tuple = _shape_tuple(shape2)
    if shape0_tuple != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if shape1_tuple != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected second view shape parameter: {shape1!r}")
    if shape2_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape2!r}")

    return mm_111, addmm_55, add_218, weight, bias, shape2_tuple


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward residual-add LayerNorm computation.

    SCOPE INVARIANT: accepts the same 8 inputs as Repro.forward() and returns
    the same single contiguous float32 `[128, 4096]` output tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_layernorm.py")

    mm_111, addmm_55, add_218, weight, bias, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        (HIDDEN, 1),
        device=mm_111.device,
        dtype=torch.float32,
    )
    _residual_layernorm_kernel[(ROWS,)](
        mm_111,
        addmm_55,
        add_218,
        weight,
        bias,
        output,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=8,
        num_stages=3,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
