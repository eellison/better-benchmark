"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete ViT train select_scatter/reduction scope returned by Repro.forward, including both [384] reductions over the scattered tensor, the required [384,6304] transposed side view, and the [384] reduction from that side tensor, by proving the zero-filled select_scatter leaves only token 0 nonzero and reducing/writing only those token rows while zeroing the side storage, whereas Inductor currently lowers the decomposed full/select_scatter/mul/sum/sub/view/permute graph as generic dense work over all 32*197 tokens; Inductor cannot do this today because its simplifier and reduction scheduler do not propagate the single-token sparsity created by zero-fill select_scatter through sibling reductions and a required layout-only side output; the fix is ALGEBRAIC_ELIMINATION: add select_scatter-from-zero canonicalization that rewrites downstream reductions and side-output materialization to operate on the provably nonzero token slice while preserving the transposed view contract."""
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


BATCH = 32
TOKENS = 197
CHANNELS = 384
ROWS = BATCH * TOKENS
SIDE_NUMEL = ROWS * CHANNELS
OUTPUT_NUMEL = 3 * CHANNELS
BLOCK_ZERO = 1024
BLOCK_C = 512

EXPECTED_SHAPE_PARAMS = (
    [ROWS, CHANNELS],
    [CHANNELS],
)


def _check_inputs(inputs):
    if len(inputs) != 6:
        raise ValueError(f"expected 6 repro inputs, got {len(inputs)}")
    mm, arg75_1, arg235_1, arg237_1, shape0, shape1 = inputs
    if mm.shape != (BATCH, CHANNELS) or mm.dtype != torch.float32:
        raise ValueError(f"unexpected mm: shape={tuple(mm.shape)} dtype={mm.dtype}")
    if arg75_1.shape != (CHANNELS,) or arg75_1.dtype != torch.float32:
        raise ValueError(
            f"unexpected arg75_1: shape={tuple(arg75_1.shape)} dtype={arg75_1.dtype}"
        )
    if arg235_1.shape != (BATCH, TOKENS, CHANNELS) or arg235_1.dtype != torch.float32:
        raise ValueError(
            f"unexpected arg235_1: shape={tuple(arg235_1.shape)} dtype={arg235_1.dtype}"
        )
    if arg237_1.shape != (BATCH, TOKENS, 1) or arg237_1.dtype != torch.float32:
        raise ValueError(
            f"unexpected arg237_1: shape={tuple(arg237_1.shape)} dtype={arg237_1.dtype}"
        )
    for idx, (actual, expected) in enumerate(
        zip((shape0, shape1), EXPECTED_SHAPE_PARAMS),
        start=4,
    ):
        if list(actual) != expected:
            raise ValueError(f"unexpected shape param {idx}: {actual} != {expected}")


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.jit
    def _zero_buffers_kernel(
        side_ptr,
        out0_ptr,
        out1_ptr,
        out3_ptr,
        total: tl.constexpr,
        side_numel: tl.constexpr,
        channels: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        zero = tl.zeros((BLOCK,), tl.float32)

        side_mask = offsets < side_numel
        tl.store(side_ptr + offsets, zero, mask=side_mask)

        tail = offsets - side_numel
        out0_mask = (tail >= 0) & (tail < channels)
        out1_mask = (tail >= channels) & (tail < 2 * channels)
        out3_mask = (tail >= 2 * channels) & (tail < 3 * channels)
        out0_offsets = tl.where(out0_mask, tail, 0)
        out1_offsets = tl.where(out1_mask, tail - channels, 0)
        out3_offsets = tl.where(out3_mask, tail - 2 * channels, 0)
        tl.store(out0_ptr + out0_offsets, zero, mask=out0_mask)
        tl.store(out1_ptr + out1_offsets, zero, mask=out1_mask)
        tl.store(out3_ptr + out3_offsets, zero, mask=out3_mask)

    @triton.jit
    def _token0_kernel(
        mm_ptr,
        arg75_ptr,
        arg235_ptr,
        arg237_ptr,
        side_ptr,
        out0_ptr,
        out1_ptr,
        out3_ptr,
        channels: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < channels

        mm = tl.load(mm_ptr + batch * channels + cols, mask=mask, other=0.0).to(tl.float32)
        arg75 = tl.load(arg75_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        arg235 = tl.load(
            arg235_ptr + (batch * tokens * channels) + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        arg237 = tl.load(arg237_ptr + batch * tokens).to(tl.float32)

        weighted = mm * arg75
        dot_arg75 = tl.sum(weighted, axis=0)
        dot_arg235 = tl.sum(weighted * arg235, axis=0)
        side_token = arg237 * (384.0 * weighted - dot_arg75 - arg235 * dot_arg235)

        token_row_offsets = (batch * tokens * channels) + cols
        tl.store(side_ptr + token_row_offsets, side_token, mask=mask)

        tl.atomic_add(out0_ptr + cols, mm * arg235, sem="relaxed", mask=mask)
        tl.atomic_add(out1_ptr + cols, mm, sem="relaxed", mask=mask)
        tl.atomic_add(out3_ptr + cols, side_token, sem="relaxed", mask=mask)


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
        raise RuntimeError("triton is required for this oracle")
    _check_inputs(inputs)
    mm, arg75_1, arg235_1, arg237_1, _shape_param_0, _shape_param_1 = inputs
    if mm.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    sum_with_arg235 = torch.empty((CHANNELS,), device=mm.device, dtype=torch.float32)
    sum_scatter = torch.empty((CHANNELS,), device=mm.device, dtype=torch.float32)
    side_base = torch.empty((ROWS, CHANNELS), device=mm.device, dtype=torch.float32)
    side_reduction = torch.empty((CHANNELS,), device=mm.device, dtype=torch.float32)

    total = SIDE_NUMEL + OUTPUT_NUMEL
    _zero_buffers_kernel[(triton.cdiv(total, BLOCK_ZERO),)](
        side_base,
        sum_with_arg235,
        sum_scatter,
        side_reduction,
        total,
        SIDE_NUMEL,
        CHANNELS,
        BLOCK=BLOCK_ZERO,
    )
    _token0_kernel[(BATCH,)](
        mm,
        arg75_1,
        arg235_1,
        arg237_1,
        side_base,
        sum_with_arg235,
        sum_scatter,
        side_reduction,
        CHANNELS,
        TOKENS,
        BLOCK=BLOCK_C,
    )

    return (sum_with_arg235, sum_scatter, side_base.t(), side_reduction)


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
