"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Mistral same-segment causal attention-mask tuple in one Triton pointwise kernel, including the cumsum equality predicate, key-position <= query-position predicate, bf16 0/-inf where epilogue, and all eight distinct contiguous output tensors returned by Repro.forward, whereas Inductor lowers the decomposed iota/unsqueeze/index/eq/where sibling graph as generic pointwise work in one compiled kernel; Inductor cannot materially do less local work because the required stores for eight materialized bf16 [4,1,512,512] masks dominate the scope once the cheap structured predicates are fused; the fix is BANDWIDTH_BOUND: record this repro as a full-scope materialized-mask floor unless a larger graph can eliminate or alias the sibling outputs."""
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

BATCH = 4
ROWS = 512
COLS = 512
NUM_OUTPUTS = 8
OUT_SHAPE = (BATCH, 1, ROWS, COLS)
OUT_STRIDE = (ROWS * COLS, ROWS * COLS, COLS, 1)
NUMEL = BATCH * ROWS * COLS
BLOCK_SIZE = 256

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
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.jit
    def _segment_mask_multi_output_kernel(
        cumsum_ptr,
        out0,
        out1,
        out2,
        out3,
        out4,
        out5,
        out6,
        out7,
        cumsum_stride0: tl.constexpr,
        cumsum_stride1: tl.constexpr,
        n_cols: tl.constexpr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements

        batch = offsets // (n_cols * n_cols)
        within_batch = offsets - batch * n_cols * n_cols
        query_pos = within_batch // n_cols
        key_pos = within_batch - query_pos * n_cols

        query_segment = tl.load(
            cumsum_ptr + batch * cumsum_stride0 + query_pos * cumsum_stride1,
            mask=mask,
            other=-1,
        )
        key_segment = tl.load(
            cumsum_ptr + batch * cumsum_stride0 + key_pos * cumsum_stride1,
            mask=mask,
            other=-2,
        )
        keep = (key_pos <= query_pos) & (query_segment == key_segment) & mask
        values = tl.where(keep, 0.0, -float("inf"))

        tl.store(out0 + offsets, values, mask=mask)
        tl.store(out1 + offsets, values, mask=mask)
        tl.store(out2 + offsets, values, mask=mask)
        tl.store(out3 + offsets, values, mask=mask)
        tl.store(out4 + offsets, values, mask=mask)
        tl.store(out5 + offsets, values, mask=mask)
        tl.store(out6 + offsets, values, mask=mask)
        tl.store(out7 + offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([4, 512], i64, gen=Index(4)), S([4, -1, 512, 512]))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with Triton compute and eight outputs."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_segment_mask_multi_output.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    cumsum, shape_param = inputs
    if not isinstance(cumsum, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expected tensor input 0, got {type(cumsum)!r}")
    if tuple(cumsum.shape) != (BATCH, COLS) or cumsum.dtype is not torch.int64:
        raise ValueError(
            f"{REPRO_ID} expected cumsum i64[{BATCH},{COLS}], "
            f"got shape={tuple(cumsum.shape)} dtype={cumsum.dtype}"
        )
    if not cumsum.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")
    if list(shape_param) != [BATCH, -1, ROWS, COLS]:
        raise ValueError(f"unexpected expand shape parameter: {shape_param}")

    outputs = tuple(
        torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=cumsum.device,
            dtype=torch.bfloat16,
        )
        for _ in range(NUM_OUTPUTS)
    )
    grid = (triton.cdiv(NUMEL, BLOCK_SIZE),)
    _segment_mask_multi_output_kernel[grid](
        cumsum,
        outputs[0],
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
        cumsum_stride0=cumsum.stride(0),
        cumsum_stride1=cumsum.stride(1),
        n_cols=COLS,
        n_elements=NUMEL,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
    )
    return outputs


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
