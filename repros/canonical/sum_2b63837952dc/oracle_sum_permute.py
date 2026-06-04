"""
Oracle for sum_2b63837952dc.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle collapses the
singleton-dimension permute plus view into the original contiguous
float32[128,12,1,64] storage, then one Triton kernel reads each source element
once to write the required float32[768,128] transposed view backing storage and
the sibling float32[768] reduction, whereas tuned Inductor already handles this
small full-scope multi-output reduction close to the same one-read/one-store
shape through a generic reduction schedule; Inductor cannot materially do less
today because the returned transpose must be materialized and the sum must read
all 128 rows, so the remaining gap is mostly memory traffic plus launch and
generic-schedule overhead rather than a missing scatter-reduce, split-K,
algebraic-elimination, recompute-fusion, or new-pattern optimization.
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
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 128
HEADS = 12
SINGLETON = 1
WIDTH = 64
COLS = HEADS * SINGLETON * WIDTH

INPUT_SHAPE = (ROWS, HEADS, SINGLETON, WIDTH)
INPUT_STRIDE = (COLS, WIDTH, WIDTH, 1)
VIEW_SHAPE = (ROWS, SINGLETON, COLS)
SUM_SHAPE = (COLS,)
SUM_STRIDE = (1,)
PERMUTE_SHAPE = (COLS, ROWS)
PERMUTE_STRIDE = (1, COLS)


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_COLS": 2}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_COLS": 4}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_COLS": 8}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_COLS": 16}, num_warps=8, num_stages=4),
        ],
        key=["N_COLS"],
    )
    @triton.jit
    def _sum_permute_kernel(
        input_ptr,
        sum_out_ptr,
        permute_out_ptr,
        input_s0: tl.constexpr,
        permute_s0: tl.constexpr,
        permute_s1: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        N_COLS: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_ROWS)
        cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        col_mask = cols < N_COLS

        input_offsets = rows[:, None] * input_s0 + cols[None, :]
        values = tl.load(input_ptr + input_offsets, mask=col_mask[None, :], other=0.0)

        permute_offsets = rows[:, None] * permute_s1 + cols[None, :] * permute_s0
        tl.store(permute_out_ptr + permute_offsets, values, mask=col_mask[None, :])

        sums = tl.sum(values, axis=0)
        tl.store(sum_out_ptr + cols, sums, mask=col_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter to be iterable, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    x, shape_param_0, shape_param_1, shape_param_2 = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected input 0 to be a tensor, got {type(x)!r}")
    if x.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if x.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {x.dtype}")
    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != INPUT_STRIDE or not x.is_contiguous():
        raise ValueError(f"unexpected input stride: {x.stride()}")
    if _shape_tuple(shape_param_0) != VIEW_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    if _shape_tuple(shape_param_1) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1!r}")
    if _shape_tuple(shape_param_2) != (ROWS, COLS):
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2!r}")
    return x


def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full permute/view -> sum + materialized transposed-view scope."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    x = _validate_inputs(inputs)
    sum_out = torch.empty_strided(SUM_SHAPE, SUM_STRIDE, device=x.device, dtype=x.dtype)
    permute_out = torch.empty_strided(
        PERMUTE_SHAPE,
        PERMUTE_STRIDE,
        device=x.device,
        dtype=x.dtype,
    )

    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_COLS"]),)
    _sum_permute_kernel[grid](
        x,
        sum_out,
        permute_out,
        input_s0=x.stride(0),
        permute_s0=permute_out.stride(0),
        permute_s1=permute_out.stride(1),
        BLOCK_ROWS=ROWS,
        N_COLS=COLS,
    )
    return sum_out, permute_out


def _normalize_outputs(outputs: Any) -> list[torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (tuple, list)):
        return [item for item in outputs if isinstance(item, torch.Tensor)]
    return []


def _sync_from_inputs(inputs: tuple[Any, ...]) -> None:
    for value in inputs:
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            torch.cuda.synchronize(value.device)
            return


def run_check(rtol: float, atol: float) -> bool:
    inputs = get_inputs()
    instance = get_repro_instance()

    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        _sync_from_inputs(inputs)

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    ok = True
    for idx, (actual_tensor, expected_tensor) in enumerate(zip(actual_list, expected_list)):
        shape_ok = actual_tensor.shape == expected_tensor.shape
        dtype_ok = actual_tensor.dtype == expected_tensor.dtype
        stride_ok = actual_tensor.stride() == expected_tensor.stride()
        value_ok = torch.allclose(
            actual_tensor.float(),
            expected_tensor.float(),
            rtol=rtol,
            atol=atol,
        )
        diff = (actual_tensor.float() - expected_tensor.float()).abs()
        max_diff = diff.max().item() if diff.numel() else 0.0
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"  output {idx}: {'PASS' if output_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} dtype={actual_tensor.dtype} "
            f"stride={actual_tensor.stride()} max_diff={max_diff:.2e} "
            f"allclose={value_ok})"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
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

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = run_check(rtol=args.rtol, atol=args.atol)
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        inputs = get_inputs()
        instance = get_repro_instance()
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
