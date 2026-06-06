"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete layout-only permute-clone-view repro as one direct Triton copy from `arg21_1[b, h, s, d]` to the fresh contiguous `[b * 128 + s, h * 64 + d]` output, preserving the exact clone/view output contract, whereas Inductor already lowers this fixed materialization to the same mandatory one-read/one-write traffic envelope; Inductor cannot materially improve it with a local scheduler-fusion, scatter-reduce, split-K, algebraic, recompute, or new-pattern change because the captured scope is only the required non-aliasing layout copy; the fix is BANDWIDTH_BOUND: record this repro as at floor unless a broader layout-copy or memory-traffic improvement moves both implementations together."""
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

BATCH = 64
HEADS = 16
SEQ = 128
DIM = 64
OUT_ROWS = BATCH * SEQ
OUT_COLS = HEADS * DIM
OUT_SHAPE = (OUT_ROWS, OUT_COLS)
OUT_STRIDE = (OUT_COLS, 1)
IN_SHAPE = (BATCH, HEADS, SEQ, DIM)
IN_STRIDE = (HEADS * SEQ * DIM, SEQ * DIM, DIM, 1)
BLOCK_M = 4

if triton is not None:

    @triton.jit
    def _layout_copy_kernel(
        x,
        out,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < 8192
        b = rows // 128
        s = rows - b * 128

        cols0 = tl.arange(0, 512)
        h0 = cols0 // 64
        d0 = cols0 - h0 * 64
        in0 = (
            b[:, None] * 131072
            + h0[None, :] * 8192
            + s[:, None] * 64
            + d0[None, :]
        )
        out0 = rows[:, None] * 1024 + cols0[None, :]
        vals0 = tl.load(x + in0, mask=row_mask[:, None], other=0.0)
        tl.store(out + out0, vals0, mask=row_mask[:, None])

        cols1 = 512 + tl.arange(0, 512)
        h1 = cols1 // 64
        d1 = cols1 - h1 * 64
        in1 = (
            b[:, None] * 131072
            + h1[None, :] * 8192
            + s[:, None] * 64
            + d1[None, :]
        )
        out1 = rows[:, None] * 1024 + cols1[None, :]
        vals1 = tl.load(x + in1, mask=row_mask[:, None], other=0.0)
        tl.store(out + out1, vals1, mask=row_mask[:, None])


def oracle_forward(inputs):
    """Run the complete permute-clone-view layout materialization."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(x)!r}")
    if x.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {x.dtype}")
    if tuple(x.shape) != IN_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != IN_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(x.stride())}")
    if tuple(shape_param) != OUT_SHAPE:
        raise ValueError(f"unexpected shape input: {shape_param!r}")

    out = torch.empty(OUT_SHAPE, device=x.device, dtype=x.dtype)
    _layout_copy_kernel[(triton.cdiv(OUT_ROWS, BLOCK_M),)](
        x,
        out,
        BLOCK_ROWS=BLOCK_M,
        num_warps=8,
    )
    if tuple(out.stride()) != OUT_STRIDE:
        raise RuntimeError(f"unexpected output stride: {tuple(out.stride())}")
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
