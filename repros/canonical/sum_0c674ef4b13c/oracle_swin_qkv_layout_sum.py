"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full Swin QKV layout-and-sum scope by streaming the q/k/v bmm outputs directly into the final contiguous `[401408,384]` backing storage, applying the required f32 q scale before both the store and reduction, returning the eager-like `[384,401408]` permute view, and accumulating the sibling `[384]` column sum from the same f32 values; Inductor currently schedules the Q/K/V reshape-permute-cat-clone layout materialization and the reduction as separate work, so it cannot share the layout-copy tile with the compatible reduction epilogue; the fix is SCHEDULER_FUSION: group QKV split layout materialization with sibling column reductions and sink simple per-branch pointwise work into that fused schedule."""
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
    oracle_impl,
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
BATCH = 8192
HEADS = 4
TOKENS = 49
HEAD_DIM = 32
ROWS = BATCH * TOKENS
QKV_COLS = HEADS * HEAD_DIM
COLS = 3 * QKV_COLS
SCALE = 0.1767766952966369
BLOCK_M = 256

INPUT_V_SHAPE = (BATCH * HEADS, TOKENS, HEAD_DIM)
INPUT_K_SHAPE = (BATCH * HEADS, HEAD_DIM, TOKENS)
INPUT_Q_SHAPE = (BATCH * HEADS, TOKENS, HEAD_DIM)
RESHAPE_V = (BATCH, HEADS, TOKENS, HEAD_DIM)
RESHAPE_K = (BATCH, HEADS, HEAD_DIM, TOKENS)
RESHAPE_Q = (BATCH, HEADS, TOKENS, HEAD_DIM)
CAT_RESHAPE = (3, BATCH, HEADS, TOKENS, HEAD_DIM)
FINAL_VIEW0 = (BATCH, TOKENS, COLS)
FINAL_VIEW1 = (ROWS, COLS)
SUM_SHAPE = (COLS,)

if triton is not None:

    @triton.jit
    def _swin_qkv_layout_sum_kernel(
        v_ptr,
        k_ptr,
        q_ptr,
        layout_ptr,
        sum_ptr,
        SOURCE: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        ROWS_: tl.constexpr,
        TOKENS_: tl.constexpr,
        HEADS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        QKV_COLS_: tl.constexpr,
        COLS_: tl.constexpr,
        SCALE_: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        head = tl.program_id(1)

        rows = row_block * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        dims = tl.arange(0, HEAD_DIM_)
        row_mask = rows < ROWS_

        batch = rows // TOKENS_
        token = rows - batch * TOKENS_
        col_base = SOURCE * QKV_COLS_ + head * HEAD_DIM_

        if SOURCE == 0:
            input_offsets = ((batch[:, None] * HEADS_ + head) * TOKENS_ + token[:, None]) * HEAD_DIM_ + dims[None, :]
            values = tl.load(q_ptr + input_offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
            values = values * SCALE_
        elif SOURCE == 1:
            input_offsets = ((batch[:, None] * HEADS_ + head) * HEAD_DIM_ + dims[None, :]) * TOKENS_ + token[:, None]
            values = tl.load(k_ptr + input_offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)
        else:
            input_offsets = ((batch[:, None] * HEADS_ + head) * TOKENS_ + token[:, None]) * HEAD_DIM_ + dims[None, :]
            values = tl.load(v_ptr + input_offsets, mask=row_mask[:, None], other=0.0).to(tl.float32)

        out_offsets = rows[:, None] * COLS_ + col_base + dims[None, :]
        tl.store(layout_ptr + out_offsets, values, mask=row_mask[:, None])

        partial = tl.sum(tl.where(row_mask[:, None], values, 0.0), axis=0)
        tl.atomic_add(sum_ptr + col_base + dims, partial, sem="relaxed")


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        bmm_140,
        bmm_142,
        bmm_143,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
        shape6,
    ) = inputs
    v = _expect_tensor("bmm_140", bmm_140, INPUT_V_SHAPE, torch.float32)
    k = _expect_tensor("bmm_142", bmm_142, INPUT_K_SHAPE, torch.float32)
    q = _expect_tensor("bmm_143", bmm_143, INPUT_Q_SHAPE, torch.float32)

    expected_shapes = (
        (shape0, RESHAPE_V, "shape0"),
        (shape1, RESHAPE_K, "shape1"),
        (shape2, RESHAPE_Q, "shape2"),
        (shape3, CAT_RESHAPE, "shape3"),
        (shape4, FINAL_VIEW0, "shape4"),
        (shape5, FINAL_VIEW1, "shape5"),
        (shape6, SUM_SHAPE, "shape6"),
    )
    for actual, expected, name in expected_shapes:
        actual_tuple = _shape_tuple(actual)
        if actual_tuple != expected:
            raise ValueError(f"{name} is {actual_tuple}, expected {expected}")

    return v, k, q


@oracle_impl(hardware="H100", shapes="(T([32768, 49, 32], f32), T([32768, 32, 49], f32), T([32768, 49, 32], f32), S([8192, 4, 49, 32]), S([8192, 4, 32, 49]), S([8192, 4, 49, 32]), S([3, 8192, 4, 49, 32]), S([8192, 49, 384]), S([401408, 384]), S([384]))")
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
        raise RuntimeError("Triton is required for oracle_swin_qkv_layout_sum.py")

    v, k, q = _validate_inputs(inputs)

    layout_base = torch.empty((BATCH, TOKENS, 3, HEADS, HEAD_DIM), device=q.device, dtype=torch.float32)
    layout_matrix = layout_base.reshape(FINAL_VIEW1)
    sum_base = torch.empty((1, COLS), device=q.device, dtype=torch.float32)
    sum_base.zero_()

    grid = (triton.cdiv(ROWS, BLOCK_M), HEADS)
    _swin_qkv_layout_sum_kernel[grid](
        v,
        k,
        q,
        layout_matrix,
        sum_base,
        SOURCE=0,
        BLOCK_M_=BLOCK_M,
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        QKV_COLS_=QKV_COLS,
        COLS_=COLS,
        SCALE_=SCALE,
        num_warps=8,
        num_stages=2,
    )
    _swin_qkv_layout_sum_kernel[grid](
        v,
        k,
        q,
        layout_matrix,
        sum_base,
        SOURCE=1,
        BLOCK_M_=BLOCK_M,
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        QKV_COLS_=QKV_COLS,
        COLS_=COLS,
        SCALE_=SCALE,
        num_warps=8,
        num_stages=2,
    )
    _swin_qkv_layout_sum_kernel[grid](
        v,
        k,
        q,
        layout_matrix,
        sum_base,
        SOURCE=2,
        BLOCK_M_=BLOCK_M,
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        QKV_COLS_=QKV_COLS,
        COLS_=COLS,
        SCALE_=SCALE,
        num_warps=8,
        num_stages=2,
    )

    return layout_matrix.permute(1, 0), sum_base.reshape(SUM_SHAPE)


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
