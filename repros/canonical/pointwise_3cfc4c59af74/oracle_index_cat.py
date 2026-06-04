"""
Oracle for pointwise_3cfc4c59af74

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete advanced-index gather and concatenation directly into the final f32[2048,100] output in one Triton launch, whereas Inductor materializes the f32[2048,36] indexed tensor and then runs a separate cat/copy into the final output; Inductor cannot do this today because its scheduler does not fuse an indirect-indexed gather producer through aten.cat's multi-input output layout; the fix is SCHEDULER_FUSION: teach concat scheduling to inline gather producers into the cat store region and emit per-region index expressions for the final output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


REPRO_ID = "pointwise_3cfc4c59af74"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

M = 2048
BMM_N = 9
RELU_COLS = 64
INDEX_COLS = 36
OUT_COLS = 100
BLOCK_COLS = 128
BLOCK_M = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _index_cat_kernel(
        idx0_ptr,
        idx1_ptr,
        bmm_ptr,
        relu_ptr,
        out_ptr,
        M_: tl.constexpr,
        BLOCK_COLS_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BMM_N_: tl.constexpr,
        RELU_COLS_: tl.constexpr,
        OUT_COLS_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        cols = tl.arange(0, BLOCK_COLS_)
        row_cols = rows[:, None] * OUT_COLS_ + cols[None, :]
        relu_cols = rows[:, None] * RELU_COLS_ + cols[None, :]
        row_mask = rows[:, None] < M_

        relu_mask = cols < RELU_COLS_
        relu_vals = tl.load(
            relu_ptr + relu_cols,
            mask=row_mask & relu_mask[None, :],
            other=0.0,
        )

        gather_cols = cols - RELU_COLS_
        gather_mask = (cols >= RELU_COLS_) & (cols < OUT_COLS_)
        idx0 = tl.load(idx0_ptr + gather_cols, mask=gather_mask, other=0)
        idx1 = tl.load(idx1_ptr + gather_cols, mask=gather_mask, other=0)
        gather_vals = tl.load(
            bmm_ptr + rows[:, None] * BMM_N_ * BMM_N_ + idx0[None, :] * BMM_N_ + idx1[None, :],
            mask=row_mask & gather_mask[None, :],
            other=0.0,
        )

        vals = tl.where(relu_mask[None, :], relu_vals, gather_vals)
        tl.store(out_ptr + row_cols, vals, mask=row_mask & (cols[None, :] < OUT_COLS_))


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if len(inputs) != 4:
        raise ValueError(f"expected 4 inputs, got {len(inputs)}")

    idx0, idx1, bmm, relu_1 = inputs
    if not all(isinstance(x, torch.Tensor) for x in (idx0, idx1, bmm, relu_1)):
        raise TypeError("all inputs must be tensors")
    if idx0.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if idx0.dtype != torch.int64 or idx1.dtype != torch.int64:
        raise TypeError("expected int64 index tensors")
    if bmm.dtype != torch.float32 or relu_1.dtype != torch.float32:
        raise TypeError("expected float32 data tensors")
    if tuple(idx0.shape) != (INDEX_COLS,) or tuple(idx1.shape) != (INDEX_COLS,):
        raise ValueError("unexpected index tensor shape")
    if tuple(bmm.shape) != (M, BMM_N, BMM_N):
        raise ValueError(f"unexpected bmm shape: {tuple(bmm.shape)}")
    if tuple(relu_1.shape) != (M, RELU_COLS):
        raise ValueError(f"unexpected relu_1 shape: {tuple(relu_1.shape)}")
    if not (idx0.is_contiguous() and idx1.is_contiguous() and bmm.is_contiguous() and relu_1.is_contiguous()):
        raise ValueError("oracle expects contiguous captured inputs")
    return idx0, idx1, bmm, relu_1


def oracle_forward(inputs):
    """Compute the full Repro.forward output: cat([relu_1, bmm[:, idx0, idx1]], dim=1)."""
    idx0, idx1, bmm, relu_1 = _validate_inputs(inputs)
    out = torch.empty((M, OUT_COLS), device=bmm.device, dtype=torch.float32)
    _index_cat_kernel[(triton.cdiv(M, BLOCK_M),)](
        idx0,
        idx1,
        bmm,
        relu_1,
        out,
        M_=M,
        BLOCK_COLS_=BLOCK_COLS,
        BLOCK_M_=BLOCK_M,
        BMM_N_=BMM_N,
        RELU_COLS_=RELU_COLS,
        OUT_COLS_=OUT_COLS,
        num_warps=4,
    )
    return out


def main():
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
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Disable auto-detection and skipping of stochastic outputs")
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
