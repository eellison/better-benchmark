"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete batch-axis advanced-index materialization as a row-tiled Triton gather-copy that loads one selected batch index per contiguous output tile and writes the fresh contiguous `[32,3,224,224]` result directly, whereas Inductor lowers the `view -> select -> aten.index.Tensor` graph through generic advanced-index pointwise code; Inductor cannot materially improve this repro today with a local scheduler-fusion, scatter-reduce, split-K, algebraic, recompute, or new-pattern change because the measured full-scope time is already at the required input-read plus output-write memory-traffic floor; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader gather-copy bandwidth or launch-overhead work moves both implementations together."""
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


BATCH = 32
CHANNELS = 3
HEIGHT = 224
WIDTH = 224
OUTPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUTPUT_STRIDE = (CHANNELS * HEIGHT * WIDTH, HEIGHT * WIDTH, WIDTH, 1)
ROW_ELEMS = CHANNELS * HEIGHT * WIDTH


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ELEMS": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ELEMS": 2048}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ELEMS": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ELEMS": 8192}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _batch_index_gather_kernel(
        index_ptr,
        input_ptr,
        output_ptr,
        ROW_ELEMS_CONST: tl.constexpr,
        BATCH_CONST: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        batch = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        mask = offsets < ROW_ELEMS_CONST

        source_batch = tl.load(index_ptr + batch)
        source_batch = tl.where(source_batch < 0, source_batch + BATCH_CONST, source_batch)
        values = tl.load(input_ptr + source_batch * ROW_ELEMS_CONST + offsets, mask=mask)
        tl.store(output_ptr + batch * ROW_ELEMS_CONST + offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([32], i64, gen=Index(32)), T([32, 3, 224, 224], f32))")
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
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    indices, x = inputs
    if not isinstance(indices, torch.Tensor) or not isinstance(x, torch.Tensor):
        raise TypeError("expected both repro inputs to be tensors")
    if tuple(indices.shape) != (BATCH,):
        raise ValueError(f"unexpected index shape: {tuple(indices.shape)}")
    if indices.dtype != torch.int64:
        raise ValueError(f"unexpected index dtype: {indices.dtype}")
    if tuple(x.shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected input shape: {tuple(x.shape)}")
    if tuple(x.stride()) != OUTPUT_STRIDE:
        raise ValueError(f"unexpected input stride: {tuple(x.stride())}")

    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=x.dtype)
    grid = lambda meta: (BATCH, triton.cdiv(ROW_ELEMS, meta["BLOCK_ELEMS"]))
    _batch_index_gather_kernel[grid](
        indices,
        x,
        out,
        ROW_ELEMS_CONST=ROW_ELEMS,
        BATCH_CONST=BATCH,
    )
    if tuple(out.stride()) != OUTPUT_STRIDE:
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
