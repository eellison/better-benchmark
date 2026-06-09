"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete bottom-only `constant_pad_nd` scope as one contiguous Triton final-output materialization, copying the `[30522,512]` input prefix linearly and zero-filling only the two-row tail, whereas Inductor's generic pad lowering measures within the same required read/write memory-traffic envelope for this isolated repro; Inductor cannot materially improve this local case today because the dominant work is the mandatory 59.6 MiB input read, 59.6 MiB output write, allocation, and one launch rather than avoidable producer/consumer traffic; the fix is BANDWIDTH_BOUND: record this as at floor unless broader pad-copy, allocation, or memory-codegen improvements move both implementations."""
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


IN_ROWS = 30522
COLS = 512
PAD_ROWS = 2
OUT_ROWS = IN_ROWS + PAD_ROWS
IN_NUMEL = IN_ROWS * COLS
OUT_NUMEL = OUT_ROWS * COLS
OUT_SHAPE = (OUT_ROWS, COLS)
OUT_STRIDE = (COLS, 1)


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
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
            triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        ],
        key=["TOTAL"],
    )
    @triton.jit
    def _bottom_pad_copy_kernel(
        input_ptr,
        output_ptr,
        TOTAL: tl.constexpr,
        INPUT_TOTAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        copy_mask = offsets < INPUT_TOTAL
        values = tl.load(input_ptr + offsets, mask=copy_mask, other=0.0)
        tl.store(output_ptr + offsets, values, mask=copy_mask)

        tail_offsets = INPUT_TOTAL + tl.arange(0, BLOCK_SIZE)
        tail_mask = (tl.program_id(0) == 0) & (tail_offsets < TOTAL)
        tl.store(output_ptr + tail_offsets, 0.0, mask=tail_mask)


def _expect_input(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (IN_ROWS, COLS):
        raise ValueError(f"{REPRO_ID} expects input shape {(IN_ROWS, COLS)}, got {tuple(value.shape)}")
    if tuple(value.stride()) != (COLS, 1):
        raise ValueError(f"{REPRO_ID} expects contiguous input stride {(COLS, 1)}, got {tuple(value.stride())}")
    if value.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects f32 input, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA input")
    return value


@oracle_impl(hardware="H100", shapes="(T([30522, 512], f32))")
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
        raise RuntimeError("Triton is required for oracle_bottom_pad_copy.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")

    x = _expect_input(inputs[0])
    output = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(OUT_NUMEL, meta["BLOCK_SIZE"]),)
    _bottom_pad_copy_kernel[grid](
        x,
        output,
        TOTAL=OUT_NUMEL,
        INPUT_TOTAL=IN_NUMEL,
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
