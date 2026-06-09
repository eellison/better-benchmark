"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete row L2 normalization from Repro.forward in one Triton row kernel, reducing each f32[32,128] row's sum of squares, applying sqrt plus clamp_min(1e-12), and writing the full normalized f32[32,128] output, whereas Inductor currently lowers the decomposed pow/sum/sqrt/clamp/expand/div graph as generic reduction and broadcast-divide scheduling that may split across configs and is not specialized as a row-normalization epilogue; Inductor cannot do this today because its scheduler/codegen lacks a row-reduction-to-full-row epilogue template that keeps the norm scalar in registers while storing all columns; the fix is SCHEDULER_FUSION: add scheduler/codegen support for fusing vector-norm reductions with their dependent broadcast division into a single row-normalization kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
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

ROWS = 32
COLS = 128
INPUT_SHAPE = (ROWS, COLS)
OUTPUT_STRIDE = (COLS, 1)
BLOCK_M = 2
BLOCK_N = 128
HISTORICAL_BEST_COMPILE_US = 5.1


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _row_l2_norm_kernel(
        x_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        n_cols: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cols = tl.arange(0, block_n)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        offsets = rows[:, None] * n_cols + cols[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x * x, axis=1)
        denom = tl.sqrt(sum_sq)
        denom = tl.maximum(denom, 1.0e-12)
        out = x / denom[:, None]

        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, list[int] | tuple[int, ...]]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if x.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {x.dtype}")
    if not x.is_cuda:
        raise ValueError("oracle_row_norm.py expects CUDA inputs")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={x.stride()}")
    if not isinstance(shape_param, (list, tuple)) or tuple(shape_param) != INPUT_SHAPE:
        raise ValueError(f"unexpected shape parameter: {shape_param!r}")

    return x, shape_param


@oracle_impl(hardware="H100", shapes="(T([32, 128], f32), S([32, 128]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward row L2 normalization."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_row_norm.py")

    x, _shape_param = _validate_inputs(inputs)
    out = torch.empty_strided(INPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=x.dtype)
    _row_l2_norm_kernel[(triton.cdiv(ROWS, BLOCK_M),)](
        x,
        out,
        n_rows=ROWS,
        n_cols=COLS,
        block_m=BLOCK_M,
        block_n=BLOCK_N,
        num_warps=1,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()
    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and expected.stride() == actual.stride()
    )
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
        ok = ok and _check_layout(instance, inputs)
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
            print(f"historical_best_compile_us={HISTORICAL_BEST_COMPILE_US:.3f}")
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
