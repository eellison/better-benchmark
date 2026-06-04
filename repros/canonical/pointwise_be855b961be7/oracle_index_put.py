"""
Full-scope oracle for pointwise_be855b961be7.

Gap diagnosis (classification: BANDWIDTH_BOUND): this diagnosis oracle computes
the `mm_5[:, 64:100]` producer, zero materialization, and
duplicate-preserving `index_put(accumulate=True)` into the returned contiguous
`float32[2048, 9, 9]` tensor with a Triton gather-reduce over the 36 indexed
source columns, whereas Inductor's tuned path uses the natural zero-fill plus
indexed accumulate decomposition. Inductor does not currently lower this as a
scatter-reduce gather per output cell, but on this small full scope that
alternate lowering is not a real floor: the best required coordinate-descent
compile already matches or beats the handwritten Triton path, so the remaining
cost is launch/allocation plus required 0.6 MB output/input traffic rather than
a material scatter-reduce opportunity; the fix is BANDWIDTH_BOUND: do not add a
new lowering for this repro unless a larger shape shows the scatter-reduce
rewrite winning against tuned compile.
"""
from __future__ import annotations

import argparse
import os
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "BANDWIDTH_BOUND"

N_COLS = 100
SLICE_START = 64
N_SCATTER = 36
OUT_H = 9
OUT_W = 9
OUT_CELLS = OUT_H * OUT_W


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _row_scatter_index_put_kernel(
        mm_ptr,
        idx0_ptr,
        idx1_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        mm_stride0: tl.constexpr,
        mm_stride1: tl.constexpr,
        idx0_stride0: tl.constexpr,
        idx1_stride0: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        out_stride2: tl.constexpr,
        block_m: tl.constexpr,
        block_cells: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        row_mask = rows < n_rows
        cells = tl.arange(0, block_cells)
        out_i = cells // 9
        out_j = cells - out_i * 9

        zero_offsets = (
            rows[:, None] * out_stride0
            + out_i[None, :] * out_stride1
            + out_j[None, :] * out_stride2
        )
        zero_mask = row_mask[:, None] & (cells[None, :] < 81)
        tl.store(out_ptr + zero_offsets, tl.zeros((block_m, block_cells), tl.float32), mask=zero_mask)

        for source_col in tl.static_range(0, 36):
            values = tl.load(
                mm_ptr + rows * mm_stride0 + (64 + source_col) * mm_stride1,
                mask=row_mask,
                other=0.0,
            )
            idx_i = tl.load(idx0_ptr + source_col * idx0_stride0).to(tl.int64)
            idx_j = tl.load(idx1_ptr + source_col * idx1_stride0).to(tl.int64)
            dst_offsets = rows * out_stride0 + idx_i * out_stride1 + idx_j * out_stride2
            tl.atomic_add(out_ptr + dst_offsets, values, sem="relaxed", mask=row_mask)

    @triton.jit
    def _row_gather_reduce_index_put_kernel(
        mm_ptr,
        idx0_ptr,
        idx1_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        mm_stride0: tl.constexpr,
        mm_stride1: tl.constexpr,
        idx0_stride0: tl.constexpr,
        idx1_stride0: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        out_stride2: tl.constexpr,
        block_m: tl.constexpr,
        block_cells: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        cells = tl.program_id(1) * block_cells + tl.arange(0, block_cells)
        row_mask = rows < n_rows
        cell_mask = cells < 81
        acc = tl.zeros((block_m, block_cells), tl.float32)

        for source_col in tl.static_range(0, 36):
            values = tl.load(
                mm_ptr + rows * mm_stride0 + (64 + source_col) * mm_stride1,
                mask=row_mask,
                other=0.0,
            )
            idx_i = tl.load(idx0_ptr + source_col * idx0_stride0).to(tl.int64)
            idx_j = tl.load(idx1_ptr + source_col * idx1_stride0).to(tl.int64)
            target_cell = idx_i * 9 + idx_j
            acc += tl.where(cells[None, :] == target_cell, values[:, None], 0.0)

        out_i = cells // 9
        out_j = cells - out_i * 9
        offsets = (
            rows[:, None] * out_stride0
            + out_i[None, :] * out_stride1
            + out_j[None, :] * out_stride2
        )
        tl.store(out_ptr + offsets, acc, mask=row_mask[:, None] & cell_mask[None, :])

    @triton.jit
    def _scatter_only_index_put_kernel(
        mm_ptr,
        idx0_ptr,
        idx1_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        mm_stride0: tl.constexpr,
        mm_stride1: tl.constexpr,
        idx0_stride0: tl.constexpr,
        idx1_stride0: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        out_stride2: tl.constexpr,
        block_m: tl.constexpr,
    ):
        rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
        row_mask = rows < n_rows
        for source_col in tl.static_range(0, 36):
            values = tl.load(
                mm_ptr + rows * mm_stride0 + (64 + source_col) * mm_stride1,
                mask=row_mask,
                other=0.0,
            )
            idx_i = tl.load(idx0_ptr + source_col * idx0_stride0).to(tl.int64)
            idx_j = tl.load(idx1_ptr + source_col * idx1_stride0).to(tl.int64)
            dst_offsets = rows * out_stride0 + idx_i * out_stride1 + idx_j * out_stride2
            tl.atomic_add(out_ptr + dst_offsets, values, sem="relaxed", mask=row_mask)


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _validate_inputs(inputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise AssertionError(f"expected 3 inputs, got {len(inputs)}")
    mm_5, idx0, idx1 = inputs
    if not isinstance(mm_5, torch.Tensor):
        raise TypeError(f"mm_5 must be a tensor, got {type(mm_5)!r}")
    if not isinstance(idx0, torch.Tensor) or not isinstance(idx1, torch.Tensor):
        raise TypeError("idx0 and idx1 must be tensors")
    if not mm_5.is_cuda or not idx0.is_cuda or not idx1.is_cuda:
        raise ValueError("oracle_index_put.py expects CUDA tensor inputs")
    if mm_5.dtype is not torch.float32:
        raise TypeError(f"mm_5 must be float32, got {mm_5.dtype}")
    if idx0.dtype is not torch.int64 or idx1.dtype is not torch.int64:
        raise TypeError(f"indices must be int64, got {idx0.dtype} and {idx1.dtype}")
    if mm_5.dim() != 2 or int(mm_5.shape[1]) != N_COLS:
        raise ValueError(f"mm_5 must have shape [N, {N_COLS}], got {tuple(mm_5.shape)}")
    if tuple(idx0.shape) != (N_SCATTER,) or tuple(idx1.shape) != (N_SCATTER,):
        raise ValueError(
            f"indices must both have shape ({N_SCATTER},), got {tuple(idx0.shape)} "
            f"and {tuple(idx1.shape)}"
        )
    return mm_5, idx0, idx1


def oracle_forward(inputs):
    """Run the full slice + duplicate-preserving index_put accumulate scope."""
    _require_triton_cuda()
    mm_5, idx0, idx1 = _validate_inputs(inputs)

    n_rows = int(mm_5.shape[0])
    out = torch.empty_strided(
        (n_rows, OUT_H, OUT_W),
        (OUT_CELLS, OUT_W, 1),
        device=mm_5.device,
        dtype=torch.float32,
    )
    variant = os.environ.get("POINTWISE_BE855_ORACLE_VARIANT", "gather")
    common_kwargs = dict(
        n_rows=n_rows,
        mm_stride0=int(mm_5.stride(0)),
        mm_stride1=int(mm_5.stride(1)),
        idx0_stride0=int(idx0.stride(0)),
        idx1_stride0=int(idx1.stride(0)),
        out_stride0=int(out.stride(0)),
        out_stride1=int(out.stride(1)),
        out_stride2=int(out.stride(2)),
    )
    if variant == "gather":
        block_m = int(os.environ.get("POINTWISE_BE855_GATHER_BLOCK_M", "4"))
        block_cells = int(os.environ.get("POINTWISE_BE855_GATHER_BLOCK_CELLS", "32"))
        num_warps = int(os.environ.get("POINTWISE_BE855_GATHER_WARPS", "1"))
        _row_gather_reduce_index_put_kernel[
            (triton.cdiv(n_rows, block_m), triton.cdiv(OUT_CELLS, block_cells))
        ](
            mm_5,
            idx0,
            idx1,
            out,
            **common_kwargs,
            block_m=block_m,
            block_cells=block_cells,
            num_warps=num_warps,
        )
    elif variant == "two_kernel_scatter":
        out.zero_()
        block_m = int(os.environ.get("POINTWISE_BE855_TWO_KERNEL_BLOCK_M", "128"))
        num_warps = int(os.environ.get("POINTWISE_BE855_TWO_KERNEL_WARPS", "4"))
        _scatter_only_index_put_kernel[(triton.cdiv(n_rows, block_m),)](
            mm_5,
            idx0,
            idx1,
            out,
            **common_kwargs,
            block_m=block_m,
            num_warps=num_warps,
        )
    elif variant == "scatter":
        block_m = int(os.environ.get("POINTWISE_BE855_SCATTER_BLOCK_M", "8"))
        num_warps = int(os.environ.get("POINTWISE_BE855_SCATTER_WARPS", "8"))
        _row_scatter_index_put_kernel[(triton.cdiv(n_rows, block_m),)](
            mm_5,
            idx0,
            idx1,
            out,
            **common_kwargs,
            block_m=block_m,
            block_cells=128,
            num_warps=num_warps,
        )
    else:
        raise ValueError(f"unknown POINTWISE_BE855_ORACLE_VARIANT={variant!r}")
    return out


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
