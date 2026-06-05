"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete LayerNorm-backward-style scope by fusing the two per-row hidden reductions, masked gradient side-output store, returned transpose view, and sibling feature reduction over that side output into one row-block partial kernel plus one finalizer, whereas Inductor currently schedules the row reductions, mask/pointwise output, transpose-producing layout work, and final feature reduction as separate generic regions; Inductor cannot do this today because its scheduler/codegen does not represent dependent multi-output reductions with row-local scalar intermediates and a materialized side output as a cooperative split-K column reduction; the fix is COOPERATIVE_SPLIT_K: add a scheduler lowering that lets row-reduction templates accumulate compatible column partials while writing the side output and finalizing the sibling reductions together."""
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

BATCH = 8
SEQ = 1024
ROWS = BATCH * SEQ
C = 768
ROW_SPLIT = 4
BLOCK_C = 1024
FINAL_BLOCK_C = 8


if triton is not None:

    @triton.jit
    def _row_partial_kernel(
        grad_ptr,
        weight_ptr,
        rhs_ptr,
        scale_ptr,
        keep_ptr,
        out_base_ptr,
        partial_sum_ptr,
        C_: tl.constexpr,
        ROW_SPLIT_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C_)
        col_mask = cols < C_
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        acc_col = tl.zeros([BLOCK_C_], dtype=tl.float32)

        for row_i in tl.static_range(0, ROW_SPLIT_):
            row = pid * ROW_SPLIT_ + row_i
            offsets = row * C_ + cols

            grad = tl.load(grad_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
            rhs = tl.load(rhs_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
            weighted = grad * weight
            row_sum = tl.sum(tl.where(col_mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(col_mask, weighted * rhs, 0.0), axis=0)
            scale = tl.load(scale_ptr + row).to(tl.float32)
            keep = tl.load(keep_ptr + offsets, mask=col_mask, other=0).to(tl.float32)

            out = scale * (weighted * C_ - row_sum - rhs * row_dot) * keep
            tl.store(out_base_ptr + offsets, out, mask=col_mask)
            acc_col += tl.where(col_mask, out, 0.0)

        partial_offsets = pid * C_ + cols
        tl.store(partial_sum_ptr + partial_offsets, acc_col, mask=col_mask)

    @triton.jit
    def _finalize_column_sum_kernel(
        partial_sum_ptr,
        out_sum_ptr,
        C_: tl.constexpr,
        BLOCK_ROW_BLOCKS: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
        offsets = blocks[:, None] * C_ + cols[None, :]
        vals = tl.load(partial_sum_ptr + offsets).to(tl.float32)
        tl.store(out_sum_ptr + cols, tl.sum(vals, axis=0))


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _expect_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    actual = tuple(int(dim) for dim in value)
    if actual != expected:
        raise ValueError(f"{name} is {actual}, expected {expected}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    grad, weight, rhs, scale, keep, shape0, shape1 = inputs
    grad_t = _expect_tensor(
        "arg303_1",
        grad,
        (BATCH, SEQ, C),
        torch.float32,
        (SEQ * C, C, 1),
    )
    weight_t = _expect_tensor("arg96_1", weight, (C,), torch.float32, (1,))
    rhs_t = _expect_tensor(
        "arg230_1",
        rhs,
        (BATCH, SEQ, C),
        torch.float32,
        (SEQ * C, C, 1),
    )
    scale_t = _expect_tensor(
        "arg231_1",
        scale,
        (BATCH, SEQ, 1),
        torch.float32,
        (SEQ, 1, 1),
    )
    keep_t = _expect_tensor(
        "arg229_1",
        keep,
        (BATCH, SEQ, C),
        torch.bool,
        (SEQ * C, C, 1),
    )
    _expect_shape("_shape_param_0", shape0, (ROWS, C))
    _expect_shape("_shape_param_1", shape1, (C,))

    device = grad_t.device
    if any(t.device != device for t in (weight_t, rhs_t, scale_t, keep_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return grad_t, weight_t, rhs_t, scale_t, keep_t


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    grad, weight, rhs, scale, keep = _validate_inputs(inputs)
    num_row_blocks = triton.cdiv(ROWS, ROW_SPLIT)
    out_base = torch.empty((ROWS, C), device=grad.device, dtype=torch.float32)
    out_sum = torch.empty((C,), device=grad.device, dtype=torch.float32)
    partial_sum = torch.empty((num_row_blocks, C), device=grad.device, dtype=torch.float32)

    _row_partial_kernel[(num_row_blocks,)](
        grad,
        weight,
        rhs,
        scale,
        keep,
        out_base,
        partial_sum,
        C_=C,
        ROW_SPLIT_=ROW_SPLIT,
        BLOCK_C_=BLOCK_C,
        num_warps=2,
        num_stages=1,
    )
    _finalize_column_sum_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_sum,
        out_sum,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=1,
    )

    return out_base.permute(1, 0), out_sum


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
