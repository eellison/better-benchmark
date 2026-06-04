"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete row L2 normalization from Repro.forward in one Triton row kernel, loading the selected `arg0[:, -1, :]` slice from the original contiguous f32[64,50,256] input, reducing each row's 256-wide sum of squares, applying sqrt plus clamp_min(1e-12), and writing the full normalized contiguous f32[64,256] output, whereas Inductor already lowers this tiny full-scope graph to one fused reduction/divide kernel at the practical launch plus memory-materialization floor; Inductor cannot do materially less work through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the required dynamic input read, row reduction, and full output store dominate this capture; the fix class is BANDWIDTH_BOUND: treat this as diagnosis-only unless the full-scope oracle beats the required compile configs."""
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

BATCH = 64
TIMESTEPS = 50
COLS = 256
SELECTED_TIMESTEP = TIMESTEPS - 1
INPUT_SHAPE = (BATCH, TIMESTEPS, COLS)
OUTPUT_SHAPE = (BATCH, COLS)
OUTPUT_STRIDE = (COLS, 1)
HISTORICAL_BEST_COMPILE_US = 5.696000065654516


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 2, "BLOCK_N": 256}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=4, num_stages=3),
        ],
        key=["n_rows", "n_cols"],
    )
    @triton.jit
    def _row_l2_norm_selected_kernel(
        x_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        n_cols: tl.constexpr,
        input_row_stride: tl.constexpr,
        input_timestep_stride: tl.constexpr,
        selected_timestep: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_N)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)

        input_offsets = (
            rows[:, None] * input_row_stride
            + selected_timestep * input_timestep_stride
            + cols[None, :]
        )
        x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)

        sum_sq = tl.sum(x * x, axis=1)
        denom = tl.maximum(tl.sqrt(sum_sq), 1.0e-12)
        out = x / denom[:, None]

        output_offsets = rows[:, None] * n_cols + cols[None, :]
        tl.store(out_ptr + output_offsets, out, mask=mask)


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
    if not isinstance(shape_param, (list, tuple)) or tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected shape parameter: {shape_param!r}")

    return x, shape_param


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward selected-row L2 normalization."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_row_norm.py")

    x, _shape_param = _validate_inputs(inputs)
    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(BATCH, meta["BLOCK_M"]),)
    _row_l2_norm_selected_kernel[grid](
        x,
        out,
        n_rows=BATCH,
        n_cols=COLS,
        input_row_stride=TIMESTEPS * COLS,
        input_timestep_stride=COLS,
        selected_timestep=SELECTED_TIMESTEP,
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
