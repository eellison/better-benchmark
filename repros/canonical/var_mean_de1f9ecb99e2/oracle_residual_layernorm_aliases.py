"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full GPT-J residual add tree, fp32 hidden-size-4096 var_mean LayerNorm, affine epilogue, and four aliased `[128,4096]` view returns in one shape-specialized Triton row kernel, whereas Inductor currently lowers the captured residual producers and normalization/view-alias consumers through generic scheduler fusion and norm-template codegen with avoidable reduction bookkeeping and alias-output handling overhead; Inductor cannot do this today because its scheduler does not form a single fixed-hidden residual-add LayerNorm plan that preserves a multi-output alias contract from one result storage; the fix is SCHEDULER_FUSION: teach the normalization scheduler to fuse fixed-rank residual add producers into one row-reduction/affine kernel and return repeated view aliases from the same output buffer."""
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
EPS = 1.0e-5
BLOCK_H = 4096
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


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
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets).to(tl.float32)
        addmm = tl.load(addmm_ptr + offsets).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        summed = (mm + addmm) + residual

        mean = tl.sum(summed, axis=0) / hidden
        centered = summed - mean
        variance = tl.sum(centered * centered, axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols).to(tl.float32)
        bias = tl.load(bias_ptr + cols).to(tl.float32)
        out = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, out)


def _shape_tuple(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    mm, addmm, residual, weight, bias, shape0, shape1, shape2, shape3, shape4, shape5 = inputs
    tensor_inputs = (mm, addmm, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (ROWS, HIDDEN),
        (ROWS, HIDDEN),
        (1, ROWS, HIDDEN),
        (HIDDEN,),
        (HIDDEN,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if any(value.dtype != torch.float32 for value in tensor_inputs):
        dtypes = ", ".join(str(value.dtype) for value in tensor_inputs)
        raise TypeError(f"all tensor inputs must be f32, got {dtypes}")

    if _shape_tuple(shape0) != (1, ROWS, HIDDEN) or _shape_tuple(shape1) != (1, ROWS, HIDDEN):
        raise ValueError(f"unexpected input view shapes: {shape0!r}, {shape1!r}")

    out_shapes = tuple(_shape_tuple(shape) for shape in (shape2, shape3, shape4, shape5))
    for index, shape in enumerate(out_shapes, start=7):
        if shape != (ROWS, HIDDEN):
            raise ValueError(f"input {index} unexpected output shape parameter: {shape!r}")

    return mm, addmm, residual, weight, bias, *out_shapes


def oracle_residual_layernorm_aliases(
    mm: torch.Tensor,
    addmm: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape2: tuple[int, int],
    shape3: tuple[int, int],
    shape4: tuple[int, int],
    shape5: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the full Repro.forward result and return four aliased views."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    base = torch.empty_strided(
        (1, ROWS, HIDDEN),
        (ROWS * HIDDEN, HIDDEN, 1),
        device=mm.device,
        dtype=mm.dtype,
    )
    _residual_layernorm_kernel[(ROWS,)](
        mm,
        addmm,
        residual,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=8,
        num_stages=4,
    )

    return (
        base.view(shape2),
        base.view(shape3),
        base.view(shape4),
        base.view(shape5),
    )


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full-scope residual-add LayerNorm oracle."""
    return oracle_residual_layernorm_aliases(*_validate_inputs(inputs))


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def _check_layout_and_aliasing(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    eager_outputs = tuple(eager)
    oracle_outputs = tuple(oracle_out)
    layout_ok = all(e.stride() == o.stride() for e, o in zip(eager_outputs, oracle_outputs))
    eager_alias_ok = len({_storage_ptr(out) for out in eager_outputs}) == 1
    oracle_alias_ok = len({_storage_ptr(out) for out in oracle_outputs}) == 1
    ok = layout_ok and eager_alias_ok and oracle_alias_ok
    print(
        f"  output alias/layout: {'PASS' if ok else 'FAIL'} "
        f"(layout={layout_ok}, eager_alias={eager_alias_ok}, oracle_alias={oracle_alias_ok})"
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
        ok = _check_layout_and_aliasing(instance, inputs) and ok
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
