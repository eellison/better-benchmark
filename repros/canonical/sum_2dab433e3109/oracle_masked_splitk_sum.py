"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete scalar-where f32 channel reduction with a shape-specialized two-stage Triton split-K reduction and predicates the `getitem_30` load to lanes that select the non-scalar arm, whereas Inductor's fused compare/select/reduction split still evaluates both `where` arms while producing partial sums plus finalization for this large contiguous input; Inductor cannot turn that into a meaningful win today because the required f32 predicate input, random lane mask, scalar-selected values, and final f32 accumulation leave the workload dominated by memory traffic rather than a missing semantic rewrite; the fix is BANDWIDTH_BOUND: record this full-scope floor and revisit only if broader memory-bandwidth or reduction-template improvements move both paths."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 64
CHANNELS = 64
TIME = 95696
INPUT_SHAPE = (BATCH, CHANNELS, TIME)
INPUT_STRIDE = (CHANNELS * TIME, TIME, 1)
OUTPUT_SHAPE = (CHANNELS,)
OUTPUT_STRIDE = (1,)
REDUCTION_NUMEL = BATCH * TIME
NUM_PARTS = 47
PART_SIZE = triton.cdiv(REDUCTION_NUMEL, NUM_PARTS) if triton is not None else 130310
PARTIAL_SHAPE = (CHANNELS, NUM_PARTS)
PARTIAL_STRIDE = (1, CHANNELS)
BLOCK_R = 2048
PARTIAL_BLOCK = 64


if triton is not None:

    @triton.jit
    def _masked_splitk_partial_kernel(
        arg_ptr,
        full_ptr,
        getitem_ptr,
        partial_ptr,
        REDUCTION_NUMEL_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        TIME_: tl.constexpr,
        PART_SIZE_: tl.constexpr,
        BLOCK_R_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        part = tl.program_id(1)
        lanes = tl.arange(0, BLOCK_R_)
        full_value = tl.load(full_ptr).to(tl.float32)
        acc = tl.zeros([BLOCK_R_], tl.float32)

        for local_base in tl.range(0, PART_SIZE_, BLOCK_R_):
            local_offsets = local_base + lanes
            reduction_offsets = part * PART_SIZE_ + local_offsets
            active = (local_offsets < PART_SIZE_) & (reduction_offsets < REDUCTION_NUMEL_)
            batch = reduction_offsets // TIME_
            time = reduction_offsets - batch * TIME_
            offsets = batch * CHANNELS_ * TIME_ + channel * TIME_ + time

            predicate = tl.load(
                arg_ptr + offsets,
                mask=active,
                eviction_policy="evict_last",
                other=0.0,
            ).to(tl.float32)
            take_full = predicate <= 0.0
            source = tl.load(
                getitem_ptr + offsets,
                mask=active & ~take_full,
                eviction_policy="evict_last",
                other=0.0,
            ).to(tl.float32)
            selected = tl.where(take_full, full_value, source)
            acc += tl.where(active, selected, 0.0)

        partial = tl.sum(acc, axis=0)
        tl.store(partial_ptr + channel + CHANNELS_ * part, partial)

    @triton.jit
    def _masked_splitk_finalize_kernel(
        partial_ptr,
        out_ptr,
        CHANNELS_: tl.constexpr,
        NUM_PARTS_: tl.constexpr,
        PARTIAL_BLOCK_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        parts = tl.arange(0, PARTIAL_BLOCK_)
        active = parts < NUM_PARTS_
        partials = tl.load(
            partial_ptr + channel + CHANNELS_ * parts,
            mask=active,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(partials, axis=0)
        tl.store(out_ptr + channel, total)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    arg13_1, full, getitem_30 = inputs
    arg13_1 = _require_f32_tensor("arg13_1", arg13_1, INPUT_SHAPE, INPUT_STRIDE)
    full = _require_f32_tensor("full", full, (), ())
    getitem_30 = _require_f32_tensor("getitem_30", getitem_30, INPUT_SHAPE, INPUT_STRIDE)
    if full.device != arg13_1.device or getitem_30.device != arg13_1.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return arg13_1, full, getitem_30


@oracle_impl(hardware="H100", shapes="(T([64, 64, 95696], f32), T([], f32), T([64, 64, 95696], f32))")
def oracle_forward(inputs):
    """Run the complete Repro.forward scalar-where channel reduction."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    arg13_1, full, getitem_30 = _validate_inputs(inputs)
    partials = torch.empty_strided(
        PARTIAL_SHAPE,
        PARTIAL_STRIDE,
        device=arg13_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=arg13_1.device,
        dtype=torch.float32,
    )
    _masked_splitk_partial_kernel[(CHANNELS, NUM_PARTS)](
        arg13_1,
        full,
        getitem_30,
        partials,
        REDUCTION_NUMEL_=REDUCTION_NUMEL,
        CHANNELS_=CHANNELS,
        TIME_=TIME,
        PART_SIZE_=PART_SIZE,
        BLOCK_R_=BLOCK_R,
        num_warps=16,
        num_stages=1,
    )
    _masked_splitk_finalize_kernel[(CHANNELS,)](
        partials,
        out,
        CHANNELS_=CHANNELS,
        NUM_PARTS_=NUM_PARTS,
        PARTIAL_BLOCK_=PARTIAL_BLOCK,
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
