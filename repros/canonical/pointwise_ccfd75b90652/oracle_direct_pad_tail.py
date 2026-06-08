"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete double-permute plus three-row `constant_pad_nd` scope by stripping the canceling metadata-only permutes and using a linear Triton copy for the input prefix plus a zero fill for the final pad rows, whereas Inductor lowers the same scope as a generic pointwise pad materialization after view canonicalization; Inductor cannot materially improve this local repro with scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new narrow pattern because the output contract requires reading every input element and writing a fresh contiguous padded output; the fix is BANDWIDTH_BOUND: record this as a pad-copy floor unless broader pointwise memory-codegen, memset/copy lowering, or launch-overhead work moves the shared bandwidth envelope."""
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


CLASSIFICATION = "BANDWIDTH_BOUND"
PAD_ROWS = 3
BLOCK_SIZE = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _direct_pad_tail_kernel(
        input_ptr,
        output_ptr,
        INPUT_ELEMENTS: tl.constexpr,
        TOTAL_ELEMENTS: tl.constexpr,
        COPY_BLOCKS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        if pid < COPY_BLOCKS:
            mask = offsets < INPUT_ELEMENTS
            values = tl.load(input_ptr + offsets, mask=mask)
            tl.store(output_ptr + offsets, values, mask=mask)
        else:
            tail_offsets = INPUT_ELEMENTS + (pid - COPY_BLOCKS) * BLOCK + tl.arange(0, BLOCK)
            mask = tail_offsets < TOTAL_ELEMENTS
            zeros = tl.zeros((BLOCK,), tl.float32)
            tl.store(output_ptr + tail_offsets, zeros, mask=mask)


def _expect_input(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor input 0, got {type(value)!r}")
    if value.ndim != 2:
        raise ValueError(f"expected a rank-2 tensor, got shape {tuple(value.shape)}")
    if value.dtype is not torch.float32:
        raise TypeError(f"expected torch.float32 input, got {value.dtype}")
    expected_stride = (value.shape[1], 1)
    if tuple(value.stride()) != expected_stride:
        raise ValueError(f"expected contiguous input stride {expected_stride}, got {tuple(value.stride())}")
    if value.storage_offset() != 0:
        raise ValueError(f"expected zero storage offset, got {value.storage_offset()}")
    return value


def oracle_forward(inputs):
    """Run the complete inverse-permute and bottom-pad scope."""
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    arg2_1 = _expect_input(inputs[0])
    rows, cols = arg2_1.shape
    output = torch.empty_strided(
        (rows + PAD_ROWS, cols),
        (cols, 1),
        device=arg2_1.device,
        dtype=arg2_1.dtype,
    )

    if not arg2_1.is_cuda or triton is None:
        output[:-PAD_ROWS, :].copy_(arg2_1)
        output[-PAD_ROWS:, :].zero_()
        return output

    input_elements = arg2_1.numel()
    total_elements = output.numel()
    copy_blocks = triton.cdiv(input_elements, BLOCK_SIZE)
    zero_blocks = triton.cdiv(total_elements - input_elements, BLOCK_SIZE)
    _direct_pad_tail_kernel[(copy_blocks + zero_blocks,)](
        arg2_1,
        output,
        INPUT_ELEMENTS=input_elements,
        TOTAL_ELEMENTS=total_elements,
        COPY_BLOCKS=copy_blocks,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
    )
    if output.stride() != (cols, 1) or output.storage_offset() != 0:
        raise RuntimeError("oracle produced an unexpected output layout")
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
