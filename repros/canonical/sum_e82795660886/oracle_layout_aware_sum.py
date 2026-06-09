"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full view-plus-sum scope with Inductor's 182-way fp32 partial split and final partial order, while using equivalent affine chunk-to-layout addressing; Inductor already reads the required 760 MB once and uses the same two-stage reduction shape, so the measured gap is at floor for this memory-streaming reduction; the fix is BANDWIDTH_BOUND: keep this as a floor case unless a future layout-aware reduction also improves the required full-input read."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 64
CHANNELS = 8
TIME = 371372
BATCH_STRIDE = CHANNELS * TIME
ELEMS_PER_CHANNEL = BATCH * TIME
NUM_PARTIALS = 182
PARTIAL_SIZE = 130593


if triton is not None:

    @triton.jit
    def _partial_sum_kernel(
        in_ptr,
        partial_ptr,
        TIME_: tl.constexpr,
        BATCH_STRIDE_: tl.constexpr,
        ELEMS_PER_CHANNEL_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        NUM_PARTIALS_: tl.constexpr,
        PARTIAL_SIZE_: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < (CHANNELS_ * NUM_PARTIALS_)
        rbase = tl.arange(0, RBLOCK)[None, :]

        partial = xindex % NUM_PARTIALS_
        channel = xindex // NUM_PARTIALS_
        chunk_start = partial * PARTIAL_SIZE_
        batch0 = chunk_start // TIME_
        time0 = chunk_start - batch0 * TIME_

        acc = tl.full([XBLOCK, RBLOCK], 0.0, tl.float32)
        for roffset in tl.range(0, PARTIAL_SIZE_, RBLOCK):
            rindex = roffset + rbase
            rmask = rindex < PARTIAL_SIZE_
            logical = chunk_start + rindex
            active = logical < ELEMS_PER_CHANNEL_

            time_unwrapped = time0 + rindex
            crosses_batch = time_unwrapped >= TIME_
            batch = batch0 + crosses_batch.to(tl.int32)
            time_index = time_unwrapped - crosses_batch.to(tl.int32) * TIME_
            offset = batch * BATCH_STRIDE_ + channel * TIME_ + time_index

            values = tl.load(
                in_ptr + offset,
                mask=rmask & active & xmask,
                eviction_policy="evict_last",
                other=0.0,
            )
            updated = acc + values
            acc = tl.where(rmask & xmask, updated, acc)

        sums = tl.sum(acc, axis=1)[:, None]
        tl.store(partial_ptr + xindex, sums, mask=xmask)

    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        out_ptr,
        CHANNELS_: tl.constexpr,
        NUM_PARTIALS_: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        xoffset = tl.program_id(0) * XBLOCK
        channel = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = channel < CHANNELS_
        rindex = tl.arange(0, RBLOCK)[None, :]
        rmask = rindex < NUM_PARTIALS_
        values = tl.load(
            partial_ptr + channel * NUM_PARTIALS_ + rindex,
            mask=rmask & xmask,
            eviction_policy="evict_first",
            other=0.0,
        )
        values = tl.where(rmask & xmask, values, 0.0)
        sums = tl.sum(values, axis=1)[:, None].to(tl.float32)
        tl.store(out_ptr + channel, sums, mask=xmask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"expected 2 inputs, got {len(inputs)}")

    arg35_1, shape = inputs
    if not isinstance(arg35_1, torch.Tensor):
        raise TypeError("first input must be a tensor")
    if arg35_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if arg35_1.dtype != torch.float32:
        raise ValueError(f"expected float32 input, got {arg35_1.dtype}")
    if tuple(arg35_1.shape) != (64, 4, 2, 371372):
        raise ValueError(f"unexpected input shape: {tuple(arg35_1.shape)}")
    if tuple(arg35_1.stride()) != (2970976, 742744, 371372, 1):
        raise ValueError(f"unexpected input stride: {tuple(arg35_1.stride())}")
    if tuple(shape) != (64, 8, 371372):
        raise ValueError(f"unexpected view shape parameter: {shape}")
    return arg35_1


@oracle_impl(hardware="H100", shapes="(T([64, 4, 2, 371372], f32), S([64, 8, 371372]))")
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
    arg35_1 = _validate_inputs(inputs)
    partials = torch.empty_strided(
        (CHANNELS, NUM_PARTIALS),
        (NUM_PARTIALS, 1),
        device=arg35_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided((CHANNELS,), (1,), device=arg35_1.device, dtype=torch.float32)

    xblock = 1
    _partial_sum_kernel[((CHANNELS * NUM_PARTIALS + xblock - 1) // xblock,)](
        arg35_1,
        partials,
        TIME_=TIME,
        BATCH_STRIDE_=BATCH_STRIDE,
        ELEMS_PER_CHANNEL_=ELEMS_PER_CHANNEL,
        CHANNELS_=CHANNELS,
        NUM_PARTIALS_=NUM_PARTIALS,
        PARTIAL_SIZE_=PARTIAL_SIZE,
        XBLOCK=xblock,
        RBLOCK=2048,
        num_warps=16,
        num_stages=1,
    )
    _final_sum_kernel[(CHANNELS,)](
        partials,
        out,
        CHANNELS_=CHANNELS,
        NUM_PARTIALS_=NUM_PARTIALS,
        XBLOCK=1,
        RBLOCK=256,
        num_warps=2,
        num_stages=1,
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
