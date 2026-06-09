"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete bool-to-fp32 scale/multiply, sentinel `arg0 == -1` zeroing, fresh zero-filled `f32[320,256]`, and duplicate-preserving `index_put(accumulate=True)` as direct Triton zero-fill plus flat masked atomic scatter-add kernels, whereas Inductor already emits the same practical two-kernel flat atomic scatter plan for this captured shape and benchmarks at parity; Inductor cannot materially improve it today because the remaining work is the mandatory 8,388,608 duplicate-index scatter updates with no layout/stencil materialization left to remove; the fix is SCATTER_REDUCE: only a future collision-aware grouped scatter-reduce that combines duplicate row updates before atomics could beat this floor."""
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

OUT_ROWS = 320
COLS = 256
SCALE = 1.0526315789473684
ZERO_BLOCK = 1024
SCATTER_BLOCK = 1024
NUM_WARPS = 4


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.jit
    def _zero_kernel(
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(out_ptr + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < n_elements)

    @triton.jit
    def _masked_scale_scatter_kernel(
        bool_ptr,
        values_ptr,
        index_ptr,
        out_ptr,
        bool_stride0: tl.constexpr,
        bool_stride1: tl.constexpr,
        bool_stride2: tl.constexpr,
        values_stride0: tl.constexpr,
        values_stride1: tl.constexpr,
        values_stride2: tl.constexpr,
        index_stride0: tl.constexpr,
        index_stride1: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        seq_len: tl.constexpr,
        total_elements: tl.constexpr,
        OUT_ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        element_mask = offsets < total_elements

        source = offsets // COLS_
        cols = offsets - source * COLS_
        batch = source // seq_len
        pos = source - batch * seq_len
        raw_index = tl.load(
            index_ptr + batch * index_stride0 + pos * index_stride1,
            mask=element_mask,
            other=0,
        ).to(tl.int64)
        wrapped_index = tl.where(raw_index < 0, raw_index + OUT_ROWS_, raw_index)

        valid_index = (wrapped_index >= 0) & (wrapped_index < OUT_ROWS_)
        tl.device_assert(valid_index, "index out of bounds: 0 <= index < 320")
        active = element_mask & valid_index

        bool_values = tl.load(
            bool_ptr
            + batch * bool_stride0
            + pos * bool_stride1
            + cols * bool_stride2,
            mask=active,
            other=0,
        ).to(tl.float32)
        fp_values = tl.load(
            values_ptr
            + batch * values_stride0
            + pos * values_stride1
            + cols * values_stride2,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        updates = tl.where(raw_index == -1, 0.0, bool_values * fp_values * SCALE_)

        tl.atomic_add(
            out_ptr + wrapped_index * out_stride0 + cols * out_stride1,
            updates,
            sem="relaxed",
            mask=active,
        )


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"expected 3 inputs, got {len(inputs)}")

    arg1_1, arg3_1, arg0_1 = inputs
    if not isinstance(arg1_1, torch.Tensor):
        raise TypeError("arg1_1 must be a tensor")
    if not isinstance(arg3_1, torch.Tensor):
        raise TypeError("arg3_1 must be a tensor")
    if not isinstance(arg0_1, torch.Tensor):
        raise TypeError("arg0_1 must be a tensor")
    if arg1_1.device.type != "cuda" or arg3_1.device.type != "cuda" or arg0_1.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if arg1_1.dtype != torch.bool:
        raise TypeError(f"arg1_1 must be bool, got {arg1_1.dtype}")
    if arg3_1.dtype != torch.float32:
        raise TypeError(f"arg3_1 must be float32, got {arg3_1.dtype}")
    if arg0_1.dtype != torch.int64:
        raise TypeError(f"arg0_1 must be int64, got {arg0_1.dtype}")
    if arg1_1.dim() != 3 or arg3_1.dim() != 3:
        raise ValueError(
            f"arg1_1 and arg3_1 must be rank 3, got {tuple(arg1_1.shape)} and {tuple(arg3_1.shape)}"
        )
    if arg0_1.dim() != 2:
        raise ValueError(f"arg0_1 must be rank 2, got shape {tuple(arg0_1.shape)}")
    if tuple(arg1_1.shape) != tuple(arg3_1.shape):
        raise ValueError(f"arg1_1 shape {tuple(arg1_1.shape)} must match arg3_1 {tuple(arg3_1.shape)}")
    if tuple(arg1_1.shape[:2]) != tuple(arg0_1.shape):
        raise ValueError(
            f"arg1_1 leading shape {tuple(arg1_1.shape[:2])} must match arg0_1 {tuple(arg0_1.shape)}"
        )
    if int(arg1_1.shape[2]) != COLS:
        raise ValueError(f"expected last dimension {COLS}, got {int(arg1_1.shape[2])}")
    return arg1_1, arg3_1, arg0_1


@oracle_impl(hardware="H100", shapes="(T([8, 4096, 256], b8), T([8, 4096, 256], f32), T([8, 4096], i64, gen=Index(320)))")
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
        raise RuntimeError("triton is required for this oracle")

    arg1_1, arg3_1, arg0_1 = _validate_inputs(inputs)
    batch = int(arg0_1.shape[0])
    seq_len = int(arg0_1.shape[1])
    total_sources = batch * seq_len
    n_elements = OUT_ROWS * COLS
    total_updates = total_sources * COLS

    out = torch.empty_strided(
        (OUT_ROWS, COLS),
        (COLS, 1),
        device=arg3_1.device,
        dtype=torch.float32,
    )
    _zero_kernel[(triton.cdiv(n_elements, ZERO_BLOCK),)](
        out,
        n_elements=n_elements,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _masked_scale_scatter_kernel[(triton.cdiv(total_updates, SCATTER_BLOCK),)](
        arg1_1,
        arg3_1,
        arg0_1,
        out,
        bool_stride0=int(arg1_1.stride(0)),
        bool_stride1=int(arg1_1.stride(1)),
        bool_stride2=int(arg1_1.stride(2)),
        values_stride0=int(arg3_1.stride(0)),
        values_stride1=int(arg3_1.stride(1)),
        values_stride2=int(arg3_1.stride(2)),
        index_stride0=int(arg0_1.stride(0)),
        index_stride1=int(arg0_1.stride(1)),
        out_stride0=int(out.stride(0)),
        out_stride1=int(out.stride(1)),
        seq_len=seq_len,
        total_elements=total_updates,
        OUT_ROWS_=OUT_ROWS,
        COLS_=COLS,
        SCALE_=SCALE,
        BLOCK=SCATTER_BLOCK,
        num_warps=NUM_WARPS,
    )
    return out


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
