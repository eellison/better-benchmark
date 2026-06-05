"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MT5 residual-add RMSNorm alias scope in one Triton row kernel, including the metadata-only `[4096,512] -> [32,128,512]` input view, f32 residual add, f32 mean-square reduction over hidden size 512, eps=1e-6 rsqrt, affine weight multiply, and the three returned `[4096,512]` alias views of one final buffer, whereas Inductor already lowers this fixed-row normalization region to the same CUDAGraph-measured memory-traffic floor; Inductor cannot materially improve this today with a local scheduler-fusion, scatter-reduce, split-K, algebraic, or recompute change because the required reads of two activation inputs and one weight vector, one small row reduction, rsqrt, and single output write dominate; the fix is BANDWIDTH_BOUND: record this as an at-floor RMSNorm alias variant unless broader normalization-template, launch-overhead, or memory-traffic work moves both paths."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 32
SEQ_LEN = 128
ROWS = BATCH * SEQ_LEN
HIDDEN = 512
BASE_SHAPE = (BATCH, SEQ_LEN, HIDDEN)
VIEW_SHAPE = (ROWS, HIDDEN)
BASE_STRIDE = (SEQ_LEN * HIDDEN, HIDDEN, 1)
EPS = 1.0e-6
OUTPUT_COUNT = 3

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _residual_rmsnorm_aliases_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        rows_total: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_H)
        row_mask = rows < rows_total
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * hidden + cols[None, :]

        mm = tl.load(
            mm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        added = mm + residual
        square_sum = tl.sum(tl.where(mask, added * added, 0.0), axis=1)
        inv_rms = tl.rsqrt(square_sum * (1.0 / hidden) + eps)
        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        normalized = added * inv_rms[:, None]
        out = weight[None, :] * normalized
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a shape sequence, got {value!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mm = _require_tensor("mm_132", inputs[0], VIEW_SHAPE, torch.float32)
    residual = _require_tensor("add_129", inputs[1], BASE_SHAPE, torch.float32)
    weight = _require_tensor("arg176_1", inputs[2], (HIDDEN,), torch.float32)
    _require_shape("_shape_param_0", inputs[3], BASE_SHAPE)
    output_shapes = tuple(
        _require_shape(f"_shape_param_{index}", inputs[index + 3], VIEW_SHAPE)
        for index in range(1, OUTPUT_COUNT + 1)
    )

    if residual.device != mm.device or weight.device != mm.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mm, residual, weight, output_shapes


def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the complete residual-add RMSNorm affine alias repro scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same three f32 [4096,512] view outputs. The returned tensors are views
    of one contiguous [32,128,512] base buffer, matching eager repeated-view
    layout and alias behavior.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_rmsnorm_aliases.py")

    mm, residual, weight, output_shapes = _validate_inputs(inputs)
    out_base = torch.empty_strided(
        BASE_SHAPE,
        BASE_STRIDE,
        device=mm.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_M"]),)
    _residual_rmsnorm_aliases_kernel[grid](
        mm,
        residual,
        weight,
        out_base,
        rows_total=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=HIDDEN,
    )
    return tuple(out_base.view(shape) for shape in output_shapes)


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
