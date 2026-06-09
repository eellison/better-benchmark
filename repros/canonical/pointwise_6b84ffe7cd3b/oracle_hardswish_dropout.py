"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full hardswish add/clamp/mul/div and Inductor-style keep_prob=0.8 dropout in one Triton pass while folding the `/6` hardswish divisor and `/0.8` dropout scale into a single masked multiply, whereas Inductor currently emits a fused pointwise RNG kernel but preserves the decomposed div-by-6, mask-to-float, div-by-0.8, and final multiply chain; Inductor cannot do this today because its pointwise algebraic simplifier does not combine scalar factors across a stochastic bool-to-float mask expression produced by `prims.inductor_random`; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to fold scalar factors through random-mask conversions and emit one scaled masked multiply in pointwise codegen."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

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


N_ELEMENTS = 256 * 1280
KEEP_PROB = 0.8
HARDSWISH_DROPOUT_SCALE = (1.0 / 6.0) / KEEP_PROB
BLOCK_N = 512


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def oracle_kernel(
        input_ptr,
        seed_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        xindex = offsets.to(tl.int32)

        x = tl.load(input_ptr + xindex).to(tl.float32)
        shifted = x + 3.0
        clipped = tl.minimum(tl.maximum(shifted, 0.0), 6.0)

        seed = tl.load(seed_ptr)
        rand = tl.rand(seed, xindex.to(tl.uint32))
        scale = tl.where(rand < 0.8, 0.20833333333333331, 0.0)
        out = x * clipped * scale

        tl.store(output_ptr + xindex, out)


def _validate_input(addmm: torch.Tensor) -> None:
    if addmm.device.type != "cuda":
        raise RuntimeError("oracle_hardswish_dropout requires a CUDA input")
    if addmm.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {addmm.dtype}")
    if tuple(addmm.shape) != (256, 1280):
        raise ValueError(f"expected input shape (256, 1280), got {tuple(addmm.shape)}")
    if tuple(addmm.stride()) != (1280, 1):
        raise ValueError(f"expected contiguous stride (1280, 1), got {tuple(addmm.stride())}")


@oracle_impl(hardware="H100", shapes="(T([256, 1280], f32))")
def oracle_forward(inputs):
    """Run the full hard-swish plus stochastic dropout oracle."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 1:
        raise ValueError(f"expected one input tensor, got {len(inputs)}")

    addmm = inputs[0]
    if not isinstance(addmm, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(addmm)!r}")
    _validate_input(addmm)

    seeds = torch.ops.prims.inductor_seeds.default(1, addmm.device)
    out = torch.empty_like(addmm)
    grid = (triton.cdiv(N_ELEMENTS, BLOCK_N),)
    oracle_kernel[grid](
        addmm,
        seeds,
        out,
        N=N_ELEMENTS,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
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
