"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured layout-cat-div scope by writing the fresh contiguous `float32[32, 32001]` output directly from `bmm[:,0,0]` and `mm[:, :]` with the required `1 / 0.07` scale in one row-tiled Triton kernel, whereas Inductor already fuses the same view/permute/unsqueeze/cat/div graph into a single pointwise materialization kernel over the full output; Inductor cannot materially improve this today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining work is one launch plus required input reads and output stores for the materialized cat result; the fix is BANDWIDTH_BOUND: treat this as a full-scope pointwise/layout floor unless broader pointwise launch or indexing overhead reductions move the baseline."""
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
ROWS = 32
BMM_SHAPE = (ROWS, 1, 1)
BMM_STRIDE = (1, 1, 1)
MM_COLS = 32000
MM_SHAPE = (ROWS, MM_COLS)
MM_STRIDE = (MM_COLS, 1)
OUT_COLS = MM_COLS + 1
OUTPUT_SHAPE = (ROWS, OUT_COLS)
OUTPUT_STRIDE = (OUT_COLS, 1)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_COLS": 512}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_COLS": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_COLS": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_COLS": 4096}, num_warps=8, num_stages=4),
        ],
        key=["N_COLS"],
    )
    @triton.jit
    def _cat_div_kernel(
        bmm_ptr,
        mm_ptr,
        out_ptr,
        N_COLS: tl.constexpr,
        MM_N_COLS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        cols = block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = cols < N_COLS

        from_bmm = cols == 0
        bmm_value = tl.load(bmm_ptr + row)
        mm_cols = cols - 1
        mm_value = tl.load(
            mm_ptr + row * MM_N_COLS + mm_cols,
            mask=(cols > 0) & mask,
            other=0.0,
        )
        value = tl.where(from_bmm, bmm_value, mm_value)
        tl.store(out_ptr + row * N_COLS + cols, value * 14.285714285714285, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat_div.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    bmm, mm, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(bmm, torch.Tensor) or not isinstance(mm, torch.Tensor):
        raise TypeError("expected tensor inputs for bmm and mm")
    if bmm.device.type != "cuda" or mm.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if bmm.dtype is not torch.float32 or mm.dtype is not torch.float32:
        raise TypeError(f"expected f32 inputs, got bmm={bmm.dtype}, mm={mm.dtype}")
    if tuple(bmm.shape) != BMM_SHAPE:
        raise ValueError(f"expected bmm shape {BMM_SHAPE}, got {tuple(bmm.shape)}")
    if tuple(mm.shape) != MM_SHAPE:
        raise ValueError(f"expected mm shape {MM_SHAPE}, got {tuple(mm.shape)}")
    if tuple(bmm.stride()) != BMM_STRIDE:
        raise ValueError(f"expected bmm stride {BMM_STRIDE}, got {tuple(bmm.stride())}")
    if tuple(mm.stride()) != MM_STRIDE:
        raise ValueError(f"expected mm stride {MM_STRIDE}, got {tuple(mm.stride())}")
    if tuple(shape0) != (ROWS, 1):
        raise ValueError(f"unexpected first view shape: {shape0}")
    if tuple(shape1) != (ROWS,):
        raise ValueError(f"unexpected second view shape: {shape1}")
    if tuple(shape2) != (ROWS, 1, MM_COLS):
        raise ValueError(f"unexpected third view shape: {shape2}")
    if tuple(shape3) != MM_SHAPE:
        raise ValueError(f"unexpected fourth view shape: {shape3}")
    return bmm, mm


def oracle_forward(inputs):
    """Run the full-scope Triton oracle for Repro.forward()."""
    bmm, mm = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm.device,
        dtype=mm.dtype,
    )
    grid = lambda meta: (ROWS, triton.cdiv(OUT_COLS, meta["BLOCK_COLS"]))
    _cat_div_kernel[grid](
        bmm,
        mm,
        output,
        N_COLS=OUT_COLS,
        MM_N_COLS=MM_COLS,
    )
    return output


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
