"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Albert infer residual-layernorm scope in one fixed-hidden Triton row kernel, including the addmm output view, fp16-rounded residual add, fp32 layernorm with eps=1e-12, affine epilogue, fp16 cast, and all three returned [512, 768] view aliases, whereas Inductor currently lowers the graph through its generic norm-template reduction for the residual-add input and returns view aliases around that generic kernel; Inductor cannot do this today because norm-template canonicalization does not recognize this ALBERT residual-layernorm/view-alias pattern as a fixed-K inference epilogue with one shared output buffer and specialized hidden-size-768 codegen; the fix is NEW_PATTERN: add a residual-add layernorm inference template that folds the addmm view and residual add into a fixed-K row layernorm kernel and preserves multi-view alias returns from the single fp16 output."""
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
    oracle_impl,
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

ROWS = 512
HIDDEN = 768
INPUT_VIEW_SHAPE = (1, ROWS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
BLOCK_H = 1024
EPS = 1.0e-12
CLASSIFICATION = "NEW_PATTERN"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
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

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # Repro.forward materializes the residual add as fp16 before promoting
        # to fp32 for var_mean/layernorm.
        x = (addmm + residual).to(tl.float16).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y.to(tl.float16), mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"shape parameter must be iterable, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    addmm_66, residual, weight, bias, shape0, shape1, shape2, shape3 = inputs
    tensor_inputs = (addmm_66, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (OUTPUT_SHAPE, INPUT_VIEW_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"input {index} shape {tuple(value.shape)} != {expected_shape}"
            )
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    if _shape_tuple(shape0) != INPUT_VIEW_SHAPE:
        raise ValueError(f"unexpected addmm view shape parameter: {shape0!r}")
    output_shapes = (_shape_tuple(shape1), _shape_tuple(shape2), _shape_tuple(shape3))
    if any(shape != OUTPUT_SHAPE for shape in output_shapes):
        raise ValueError(f"unexpected output view shape parameters: {shape1!r}, {shape2!r}, {shape3!r}")

    return addmm_66, residual, weight, bias, output_shapes[0], output_shapes[1], output_shapes[2]


@oracle_impl(hardware="H100", shapes="(T([512, 768], f16), T([1, 512, 768], f16), T([768], f16), T([768], f16), S([1, 512, 768]), S([512, 768]), S([512, 768]), S([512, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward computation with one Triton row kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same three contiguous float16[512, 768] views, all aliasing one fp16 result
    buffer produced after the addmm view, residual add, layernorm, affine, and
    final cast.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_layernorm.py")

    addmm_66, residual, weight, bias, shape1, shape2, shape3 = _validate_inputs(inputs)
    addmm_view = addmm_66.view(INPUT_VIEW_SHAPE)
    base = torch.empty_strided(
        INPUT_VIEW_SHAPE,
        (ROWS * HIDDEN, HIDDEN, 1),
        device=addmm_66.device,
        dtype=torch.float16,
    )
    _residual_layernorm_kernel[(ROWS,)](
        addmm_view,
        residual,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=8,
    )
    return base.view(shape1), base.view(shape2), base.view(shape3)


def _check_layout_and_alias(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual[0].is_cuda:
            torch.cuda.synchronize()

    expected_list = list(expected)
    actual_list = list(actual)
    layout_ok = all(
        tuple(actual_value.shape) == tuple(expected_value.shape)
        and actual_value.stride() == expected_value.stride()
        for expected_value, actual_value in zip(expected_list, actual_list)
    )
    expected_alias_ok = len({value.data_ptr() for value in expected_list}) == 1
    actual_alias_ok = len({value.data_ptr() for value in actual_list}) == 1
    ok = layout_ok and expected_alias_ok and actual_alias_ok
    print(
        f"  output layout/alias: {'PASS' if ok else 'FAIL'} "
        f"(expected_strides={[value.stride() for value in expected_list]}, "
        f"oracle_strides={[value.stride() for value in actual_list]}, "
        f"expected_alias={expected_alias_ok}, oracle_alias={actual_alias_ok})"
    )
    return ok


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
        ok = _check_layout_and_alias(instance, inputs) and ok
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
