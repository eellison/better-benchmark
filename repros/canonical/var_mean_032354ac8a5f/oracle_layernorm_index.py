"""
Oracle for var_mean_032354ac8a5f

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the
complete Repro.forward result by folding the final fixed `index(..., -1)` into
the producer and running the fp32 layernorm affine only for the live final
sequence row, whereas Inductor currently schedules the add/layernorm work for
all 64 sequence rows before materializing the indexed row; Inductor cannot do
this today because its scheduler and graph simplifier do not propagate a
post-normalization constant row index backward through the view, add, var_mean,
and affine layernorm pattern to prove the other rows are dead; the fix is
ALGEBRAIC_ELIMINATION: add a guarded rewrite that pushes fixed row selects
through row-independent normalization patterns and narrows the scheduled
iteration domain to the rows consumed by the final output.
"""
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

BATCH = 1
SEQ = 64
HIDDEN = 768
SELECTED_ROW = SEQ - 1
OUTPUT_SHAPE = (1, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-5
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _selected_row_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        selected_row: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden
        row_offsets = selected_row * hidden + cols

        x = (
            tl.load(addmm_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        )
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias
        tl.store(out_ptr + cols, y, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_47, add_91, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (addmm_47, add_91, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first four repro inputs must be tensors")

    if tuple(addmm_47.shape) != (SEQ, HIDDEN):
        raise ValueError(f"unexpected addmm_47 shape: {tuple(addmm_47.shape)}")
    if tuple(add_91.shape) != (BATCH, SEQ, HIDDEN):
        raise ValueError(f"unexpected add_91 shape: {tuple(add_91.shape)}")
    if tuple(weight.shape) != (HIDDEN,) or tuple(bias.shape) != (HIDDEN,):
        raise ValueError("weight and bias must have shape (768,)")
    if any(value.dtype is not torch.float32 for value in tensor_inputs):
        raise TypeError("all tensor inputs must be torch.float32")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.is_contiguous() for value in tensor_inputs):
        raise ValueError("all tensor inputs must be contiguous")
    if tuple(shape0) != (BATCH, SEQ, HIDDEN) or tuple(shape1) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected shape parameters: {shape0!r}, {shape1!r}")

    return addmm_47, add_91, weight, bias


@oracle_impl(hardware="H100", shapes="(T([64, 768], f32), T([1, 64, 768], f32), T([768], f32), T([768], f32), S([1, 64, 768]), S([1, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation with a selected-row Triton kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single contiguous float32[1, 768] output tensor. Rows 0-62 are not
    computed because the fixed final index makes them dead before return.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layernorm_index.py")

    addmm_47, add_91, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=addmm_47.device, dtype=addmm_47.dtype)
    _selected_row_layernorm_kernel[(1,)](
        addmm_47,
        add_91,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        selected_row=SELECTED_ROW,
        eps=EPS,
        BLOCK_N=triton.next_power_of_2(HIDDEN),
        num_warps=4,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()
    ok = isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor) and expected.stride() == actual.stride()
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
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
        ok = _check_layout(instance, inputs) and ok
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
