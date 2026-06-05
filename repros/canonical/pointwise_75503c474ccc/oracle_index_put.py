"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured `mul(arg1, 1)`, `eq(arg0, 1)` row mask, broadcasted source zeroing, fresh zero `f32[50265, 768]`, and duplicate-preserving `index_put(accumulate=True)` as a Triton zero-fill plus masked atomic scatter-add, whereas Inductor's tuned compile already matches that full scope on the captured shape; Inductor cannot materially improve this today through a narrower indexed-accumulation rewrite because the required dense output allocation/zeroing, 154 MB output write, and duplicate-safe `2048 x 768` update dominate, and the handwritten Triton path is within measurement noise of compiled Inductor; the fix is BANDWIDTH_BOUND: record this repro as at floor unless larger shapes show the direct masked scatter-add beating tuned compile."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "BANDWIDTH_BOUND"
OUT_ROWS = 50265
BATCH = 4
SEQ_LEN = 512
COLS = 768
ZERO_BLOCK = 1024
BLOCK_COLS = 1024

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
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

if triton is not None:

    @triton.jit
    def _zero_kernel(
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            out_ptr + offsets,
            tl.zeros((BLOCK,), tl.float32),
            mask=offsets < n_elements,
        )

    @triton.jit
    def _masked_index_put_scatter_kernel(
        values_ptr,
        index_ptr,
        out_ptr,
        values_stride0: tl.constexpr,
        values_stride1: tl.constexpr,
        values_stride2: tl.constexpr,
        index_stride0: tl.constexpr,
        index_stride1: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        seq_len: tl.constexpr,
        total_sources: tl.constexpr,
        out_rows: tl.constexpr,
        cols: tl.constexpr,
        block_cols: tl.constexpr,
    ):
        source = tl.program_id(0)
        col_offsets = tl.program_id(1) * block_cols + tl.arange(0, block_cols)
        col_mask = col_offsets < cols

        batch = source // seq_len
        token = source - batch * seq_len
        raw_index = tl.load(index_ptr + batch * index_stride0 + token * index_stride1).to(tl.int64)
        wrapped_index = tl.where(raw_index < 0, raw_index + out_rows, raw_index)
        active = (
            (source < total_sources)
            & (raw_index != 1)
            & (wrapped_index >= 0)
            & (wrapped_index < out_rows)
        )

        values = tl.load(
            values_ptr
            + batch * values_stride0
            + token * values_stride1
            + col_offsets * values_stride2,
            mask=active & col_mask,
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            out_ptr + wrapped_index * out_stride0 + col_offsets * out_stride1,
            values,
            sem="relaxed",
            mask=active & col_mask,
        )


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"expected 2 inputs, got {len(inputs)}")

    arg1_1, arg0_1 = inputs
    if not isinstance(arg1_1, torch.Tensor) or not isinstance(arg0_1, torch.Tensor):
        raise TypeError("expected tensor inputs")
    if arg1_1.device.type != "cuda" or arg0_1.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if arg1_1.dtype is not torch.float32:
        raise TypeError(f"arg1_1 must be float32, got {arg1_1.dtype}")
    if arg0_1.dtype is not torch.int64:
        raise TypeError(f"arg0_1 must be int64, got {arg0_1.dtype}")
    if tuple(arg1_1.shape) != (BATCH, SEQ_LEN, COLS):
        raise ValueError(
            f"arg1_1 expected shape {(BATCH, SEQ_LEN, COLS)}, got {tuple(arg1_1.shape)}"
        )
    if tuple(arg0_1.shape) != (BATCH, SEQ_LEN):
        raise ValueError(f"arg0_1 expected shape {(BATCH, SEQ_LEN)}, got {tuple(arg0_1.shape)}")
    return arg1_1, arg0_1


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

    arg1_1, arg0_1 = _validate_inputs(inputs)
    out = torch.empty_strided(
        (OUT_ROWS, COLS),
        (COLS, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    _zero_kernel[(triton.cdiv(OUT_ROWS * COLS, ZERO_BLOCK),)](
        out,
        n_elements=OUT_ROWS * COLS,
        BLOCK=ZERO_BLOCK,
        num_warps=4,
    )
    _masked_index_put_scatter_kernel[(BATCH * SEQ_LEN, triton.cdiv(COLS, BLOCK_COLS))](
        arg1_1,
        arg0_1,
        out,
        values_stride0=int(arg1_1.stride(0)),
        values_stride1=int(arg1_1.stride(1)),
        values_stride2=int(arg1_1.stride(2)),
        index_stride0=int(arg0_1.stride(0)),
        index_stride1=int(arg0_1.stride(1)),
        out_stride0=int(out.stride(0)),
        out_stride1=int(out.stride(1)),
        seq_len=SEQ_LEN,
        total_sources=BATCH * SEQ_LEN,
        out_rows=OUT_ROWS,
        cols=COLS,
        block_cols=BLOCK_COLS,
        num_warps=4,
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
