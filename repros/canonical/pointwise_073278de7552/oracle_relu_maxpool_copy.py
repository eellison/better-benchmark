"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full captured ReLU -> low-memory 3x3 stride-2 pad-1 maxpool-with-offsets -> copy_ scope in one Triton launch, returning f32 pooled values, i8 pool offsets, and the mutated arg0 ReLU copy, whereas Inductor currently cannot share the ReLU producer between the stencil maxpool consumer and the copy_ side-effect without scheduling extra materialization or generic pointwise work; Inductor cannot do this today because the scheduler does not fuse a stencil reduction consumer and an aliasing in-place copy side-effect around one common pointwise producer; the fix is SCHEDULER_FUSION: teach Inductor to sink ReLU into maxpool while also routing the same value into the copy_ mutation when aliasing permits."""
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


BATCH = 64
CHANNELS = 64
H_IN = 16
W_IN = 16
H_OUT = 8
W_OUT = 8
PLANES = BATCH * CHANNELS
INPUT_PLANE = H_IN * W_IN
OUTPUT_PLANE = H_OUT * W_OUT


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _relu_maxpool_copy_kernel(
        input_ptr,
        values_ptr,
        offsets_ptr,
        BLOCK_COPY: tl.constexpr,
        BLOCK_POOL: tl.constexpr,
    ):
        plane = tl.program_id(0)
        input_base = plane * BLOCK_COPY
        output_base = plane * BLOCK_POOL

        pool_offsets = tl.arange(0, BLOCK_POOL)
        oh = pool_offsets // 8
        ow = pool_offsets - oh * 8
        ih0 = oh * 2 - 1
        iw0 = ow * 2 - 1

        best_val = tl.full([BLOCK_POOL], -float("inf"), dtype=tl.float32)
        best_idx = tl.full([BLOCK_POOL], 0, dtype=tl.int8)

        for kh in tl.static_range(3):
            ih = ih0 + kh
            ih_valid = (0 <= ih) & (ih < 16)
            for kw in tl.static_range(3):
                iw = iw0 + kw
                valid = ih_valid & (0 <= iw) & (iw < 16)
                raw = tl.load(input_ptr + input_base + ih * 16 + iw, mask=valid, other=-float("inf"))
                relu = tl.where(raw != raw, raw, tl.maximum(raw, 0.0))
                is_better = valid & ((relu > best_val) | (relu != relu))
                best_val = tl.where(is_better, relu, best_val)
                best_idx = tl.where(is_better, tl.full([BLOCK_POOL], kh * 3 + kw, dtype=tl.int8), best_idx)

        copy_offsets = tl.arange(0, BLOCK_COPY)
        raw_copy = tl.load(input_ptr + input_base + copy_offsets)
        relu_copy = tl.where(raw_copy != raw_copy, raw_copy, tl.maximum(raw_copy, 0.0))
        tl.store(input_ptr + input_base + copy_offsets, relu_copy)

        tl.store(values_ptr + output_base + pool_offsets, best_val)
        tl.store(offsets_ptr + output_base + pool_offsets, best_idx)


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
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (arg0_1,) = inputs
    if not isinstance(arg0_1, torch.Tensor):
        raise TypeError("oracle expects the input to be a tensor")
    if tuple(arg0_1.shape) != (BATCH, CHANNELS, H_IN, W_IN):
        raise ValueError(f"unexpected input shape: {tuple(arg0_1.shape)}")
    if arg0_1.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {arg0_1.dtype}")
    if not arg0_1.is_cuda:
        raise ValueError("oracle requires a CUDA input")
    if not arg0_1.is_contiguous():
        raise ValueError(f"expected contiguous input, got strides {arg0_1.stride()}")

    values = torch.empty(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    offsets = torch.empty(
        (BATCH, CHANNELS, H_OUT, W_OUT),
        device=arg0_1.device,
        dtype=torch.int8,
    )

    _relu_maxpool_copy_kernel[(PLANES,)](
        arg0_1,
        values,
        offsets,
        BLOCK_COPY=INPUT_PLANE,
        BLOCK_POOL=OUTPUT_PLANE,
        num_warps=4,
    )
    return (values, offsets, arg0_1)


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
