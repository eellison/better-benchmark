"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete NFNet scaled exact-GELU pointwise scope as one Triton kernel while folding the trailing `* 1.0` identity and scalar scale chain into the store expression, whereas Inductor fuses the graph but preserves the decomposed scalar multiplies from the captured ATen program; Inductor cannot do this today because pointwise simplification does not canonicalize exact-GELU scalar chains across the `erf` producer under its default floating-point legality rules, so the fix is ALGEBRAIC_ELIMINATION: add a guarded scaled-exact-GELU canonicalization that removes identity multiplies and folds scalar constants before pointwise codegen."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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


SHAPE = (128, 128, 48, 48)
CONTIGUOUS_STRIDE = (294912, 2304, 48, 1)
CHANNELS_LAST_STRIDE = (294912, 1, 6144, 128)
N_ELEMENTS = 128 * 128 * 48 * 48


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
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _scaled_exact_gelu_kernel(
        input_ptr,
        output_ptr,
        total_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_elements

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erf_term = libdevice.erf(x * 0.7071067811865476) + 1.0
        y = (x * 0.8507521748542786) * erf_term
        tl.store(output_ptr + offsets, y, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")

    x = inputs[0]
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input must be a tensor, got {type(x)!r}")
    if not x.is_cuda:
        raise RuntimeError(f"{REPRO_ID} requires a CUDA tensor input")
    if x.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects torch.float32 input, got {x.dtype}")
    if tuple(x.shape) != SHAPE:
        raise ValueError(f"{REPRO_ID} expects shape {SHAPE}, got {tuple(x.shape)}")
    if tuple(x.stride()) not in (CONTIGUOUS_STRIDE, CHANNELS_LAST_STRIDE):
        raise ValueError(
            f"{REPRO_ID} expects stride {CONTIGUOUS_STRIDE} or "
            f"{CHANNELS_LAST_STRIDE}, got {tuple(x.stride())}"
        )
    return x


@oracle_impl(hardware="H100", shapes="(T([128, 128, 48, 48], f32))")
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
    if triton is None or libdevice is None:
        raise RuntimeError("Triton with libdevice is required for this oracle")

    x = _validate_inputs(inputs)
    output = torch.empty_strided(
        SHAPE,
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda meta: (triton.cdiv(N_ELEMENTS, meta["BLOCK_SIZE"]),)
    _scaled_exact_gelu_kernel[grid](x, output, total_elements=N_ELEMENTS)
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
