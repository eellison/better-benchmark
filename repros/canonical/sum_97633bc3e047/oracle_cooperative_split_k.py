"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full scaled GELU-gradient pointwise producer and `[0, 2, 3]` channel reduction by splitting the 128x96x96 reduction domain into channel-local Triton partial sums and a 16-value finalizer, whereas Inductor currently lowers the same full scope with its generic two-stage reduction schedule for the expensive erf/exp body; Inductor cannot do this today because the reduction scheduler does not have a shape-specialized split-K template for very small output channels with large contiguous N/H/W reductions and an expensive pointwise producer; the fix is COOPERATIVE_SPLIT_K: add a guarded split-reduction schedule that tiles the reduction domain per channel, fuses the GELU-gradient producer into partial accumulation, and finalizes the small channel vector directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 128
CHANNELS = 16
HEIGHT = 96
WIDTH = 96
HW = HEIGHT * WIDTH
REDUCTION = BATCH * HW
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
NORMAL_PDF_SCALE = 0.3989422804014327
BATCHES_PER_SPLIT = 4
SPLITS = BATCH // BATCHES_PER_SPLIT
SPLIT_REDUCTION = BATCHES_PER_SPLIT * HW
BLOCK_R = 1024
FINAL_BLOCK_TILES = 32
CLASSIFICATION = "COOPERATIVE_SPLIT_K"

if triton is not None:

    @triton.jit
    def _gelu_grad_partial_reduce_kernel(
        getitem_ptr,
        arg_ptr,
        partial_ptr,
        BLOCK_R_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        split = tl.program_id(1)
        r_base = tl.arange(0, BLOCK_R_)
        total = tl.zeros((BLOCK_R_,), dtype=tl.float32)

        for r_offset in tl.range(0, 36864, BLOCK_R_):
            r = r_offset + r_base
            local_n = r // 9216
            hw = r - local_n * 9216
            n = split * 4 + local_n
            offsets = n * 147456 + channel * 9216 + hw

            getitem = tl.load(
                getitem_ptr + offsets,
                eviction_policy="evict_first",
            ).to(tl.float32)
            arg = tl.load(
                arg_ptr + offsets,
                eviction_policy="evict_first",
            ).to(tl.float32)

            cdf = (tl.math.erf(arg * 0.7071067811865476) + 1.0) * 0.5
            pdf_term = arg * tl.exp(arg * arg * -0.5) * 0.3989422804014327
            total += (getitem * 1.7015043497085571) * (cdf + pdf_term)

        tl.store(partial_ptr + split * 16 + channel, tl.sum(total, axis=0))

    @triton.jit
    def _gelu_grad_finalize_kernel(
        partial_ptr,
        out_ptr,
        BLOCK_TILES_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES_)
        vals = tl.load(
            partial_ptr + tiles * 16 + channel,
            eviction_policy="evict_first",
        ).to(tl.float32)
        total = tl.sum(vals, axis=0)
        tl.store(out_ptr + channel, total)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    getitem_237, arg154_1 = inputs
    for idx, tensor in enumerate((getitem_237, arg154_1)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{REPRO_ID} input {idx} must be a tensor, got {type(tensor)!r}")
        if tensor.dtype is not torch.float32:
            raise TypeError(f"{REPRO_ID} input {idx} must be float32, got {tensor.dtype}")
        if tuple(tensor.shape) != (BATCH, CHANNELS, HEIGHT, WIDTH):
            raise ValueError(
                f"{REPRO_ID} input {idx} shape {tuple(tensor.shape)} != "
                f"{(BATCH, CHANNELS, HEIGHT, WIDTH)}"
            )
        if not tensor.is_cuda:
            raise RuntimeError(f"{REPRO_ID} expects CUDA inputs")
        if not tensor.is_contiguous():
            raise ValueError(f"{REPRO_ID} input {idx} must be contiguous, got stride={tensor.stride()}")

    return getitem_237, arg154_1


@oracle_impl(hardware="H100", shapes="(T([128, 16, 96, 96], f32), T([128, 16, 96, 96], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
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
        raise RuntimeError("Triton is required for oracle_cooperative_split_k.py")

    getitem_237, arg154_1 = _validate_inputs(inputs)
    partial = torch.empty((SPLITS, CHANNELS), device=getitem_237.device, dtype=torch.float32)
    output = torch.empty_strided((CHANNELS,), (1,), device=getitem_237.device, dtype=torch.float32)

    _gelu_grad_partial_reduce_kernel[(CHANNELS, SPLITS)](
        getitem_237,
        arg154_1,
        partial,
        BLOCK_R_=BLOCK_R,
        num_warps=8,
        num_stages=3,
    )
    _gelu_grad_finalize_kernel[(CHANNELS,)](
        partial,
        output,
        BLOCK_TILES_=FINAL_BLOCK_TILES,
        num_warps=8,
        num_stages=3,
    )
    return output


# --- CLI entry point ---
def main() -> None:
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
