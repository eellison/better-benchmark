"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Visformer attention softmax region from Repro.forward, including the `[768, 49, 49] -> [128, 6, 49, 49]` view, the no-op multiply, max-subtracted 0.08838834764831845 scaling, exp/sum/div softmax over the 49-wide last dimension, expand, and final `[768, 49, 49]` view, whereas Inductor lowers the decomposed amax/sub/mul/exp/sum/div view graph through its generic reduction scheduler; Inductor cannot do this today because it lacks a dedicated small-row online-softmax template that recognizes the paired max and sum reductions plus view-only epilogue as one register-resident row operation; the fix is NEW_PATTERN: add a softmax pattern lowering for small fixed reduction widths that preserves the captured arithmetic order and output layout while processing multiple rows per Triton program."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


REPRO_ID = "amax_sum_211823fef3b3"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

INPUT_SHAPE = (768, 49, 49)
VIEW_SHAPE = (128, 6, 49, 49)
OUT_SHAPE = INPUT_SHAPE
CONTIGUOUS_STRIDE = (49 * 49, 49, 1)
ROWS = 768 * 49
COLS = 49
SCALE = 0.08838834764831845


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _visformer_softmax49_kernel(
        x_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        SCALE_VALUE: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_COLS)
        active_rows = rows < N_ROWS
        active_cols = cols < N_COLS
        mask = active_rows[:, None] & active_cols[None, :]
        offsets = rows[:, None] * N_COLS + cols[None, :]

        vals = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(vals, axis=1)
        row_max = tl.where(active_rows, row_max, 0.0)

        shifted = (vals - row_max[:, None]) * SCALE_VALUE
        numer = tl.exp(shifted)
        numer = tl.where(active_cols[None, :], numer, 0.0)
        denom = tl.sum(numer, axis=1)
        out = numer / denom[:, None]

        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = tuple(int(dim) for dim in actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_14: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
) -> None:
    if not bmm_14.is_cuda:
        raise RuntimeError("CUDA tensor input is required")
    if bmm_14.dtype != torch.float32:
        raise TypeError(f"expected bmm_14 dtype torch.float32, got {bmm_14.dtype}")
    if tuple(bmm_14.shape) != INPUT_SHAPE:
        raise ValueError(f"expected bmm_14 shape {INPUT_SHAPE}, got {tuple(bmm_14.shape)}")
    if tuple(bmm_14.stride()) != CONTIGUOUS_STRIDE:
        raise ValueError(
            f"expected bmm_14 stride {CONTIGUOUS_STRIDE}, got {tuple(bmm_14.stride())}"
        )
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _launch_oracle(
    bmm_14: torch.Tensor,
    out: torch.Tensor,
    *,
    block_rows: int,
    block_cols: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if block_rows <= 0 or block_rows & (block_rows - 1):
        raise ValueError(f"block_rows must be a positive power of two, got {block_rows}")
    if block_cols < COLS or block_cols & (block_cols - 1):
        raise ValueError(f"block_cols must be a power of two >= {COLS}, got {block_cols}")

    _visformer_softmax49_kernel[(triton.cdiv(ROWS, block_rows),)](
        bmm_14,
        out,
        N_ROWS=ROWS,
        N_COLS=COLS,
        SCALE_VALUE=SCALE,
        BLOCK_ROWS=block_rows,
        BLOCK_COLS=block_cols,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_14: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    *,
    block_rows: int = 8,
    block_cols: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(bmm_14, _shape_param_0, _shape_param_1, _shape_param_2)
    out = torch.empty_strided(
        OUT_SHAPE,
        CONTIGUOUS_STRIDE,
        device=bmm_14.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        bmm_14,
        out,
        block_rows=block_rows,
        block_cols=block_cols,
        num_warps=num_warps,
    )


def oracle_forward(inputs):
    """Run the oracle computation."""
    return oracle_online_softmax(*inputs)


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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
