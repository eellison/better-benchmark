"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete view plus float32-to-complex64 cast as one direct Triton materialization, streaming the contiguous `[16384, 768]` input storage and writing each `complex64[32, 512, 768]` element with one packed 64-bit real-plus-zero-imaginary store, whereas Inductor lowers the isolated metadata view and dtype conversion through generic complex-cast pointwise code; Inductor cannot do this today because its pointwise codegen has no real-to-complex64 cast pattern that preserves the virtual view while emitting packed complex stores; the fix is NEW_PATTERN: add a guarded real-to-complex materialization lowering that packs the exact float32 real bits with a zero imaginary lane in one output store."""
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


INPUT_SHAPE = (16384, 768)
INPUT_STRIDE = (768, 1)
OUT_SHAPE = (32, 512, 768)
OUT_STRIDE = (512 * 768, 768, 1)
NUMEL = 16384 * 768


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
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 8192}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _complex_cast_kernel(
        input_ptr,
        output_u64_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        packed = values.to(tl.uint32, bitcast=True).to(tl.uint64)
        tl.store(output_u64_ptr + offsets, packed, mask=mask)


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    addmm, shape_param = inputs
    if not isinstance(addmm, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(addmm)!r}")
    if addmm.dtype is not torch.float32:
        raise TypeError(f"input 0 must be torch.float32, got {addmm.dtype}")
    if tuple(addmm.shape) != INPUT_SHAPE:
        raise ValueError(f"input 0 shape must be {INPUT_SHAPE}, got {tuple(addmm.shape)}")
    if tuple(addmm.stride()) != INPUT_STRIDE:
        raise ValueError(f"input 0 stride must be {INPUT_STRIDE}, got {tuple(addmm.stride())}")
    if tuple(int(dim) for dim in shape_param) != OUT_SHAPE:
        raise ValueError(f"shape parameter must be {OUT_SHAPE}, got {shape_param!r}")
    return addmm


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

    addmm = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=addmm.device,
        dtype=torch.complex64,
    )
    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK_N"]),)
    output_u64 = torch.view_as_real(output).view(torch.uint64).reshape(OUT_SHAPE)
    _complex_cast_kernel[grid](
        addmm,
        output_u64,
        N=NUMEL,
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
