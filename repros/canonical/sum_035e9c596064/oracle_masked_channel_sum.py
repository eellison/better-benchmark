"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete `Repro.forward` masked channel reduction as one direct Triton reduction over the batch and time axes for each channel, but Inductor already lowers the compare, literal-zero select, and f32 `[0, 2]` sum into a single fused reduction with the same memory traffic; the oracle only reaches the same measured floor rather than a meaningful speedup, so there is no proven scheduler-fusion or new-pattern gap here; the fix is BANDWIDTH_BOUND: treat this repro as at floor unless broader memory-bandwidth or reduction-codegen improvements move both implementations."""
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
CHANNELS = 2048
TIME = 92
INPUT_SHAPE = (BATCH, CHANNELS, TIME)
INPUT_STRIDE = (CHANNELS * TIME, TIME, 1)
OUTPUT_SHAPE = (CHANNELS,)
OUTPUT_STRIDE = (1,)


if triton is not None:

    @triton.jit
    def _masked_channel_block_sum_kernel(
        mask_input_ptr,
        source_ptr,
        out_ptr,
        BATCH_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        TIME_: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        k_offsets = tl.arange(0, BLOCK_K)
        c_mask = c_offsets < CHANNELS_
        k_mask = k_offsets < (BATCH_ * TIME_)
        b_offsets = k_offsets // TIME_
        t_offsets = k_offsets - b_offsets * TIME_
        offsets = (
            b_offsets[None, :] * CHANNELS_ * TIME_
            + c_offsets[:, None] * TIME_
            + t_offsets[None, :]
        )
        active = c_mask[:, None] & k_mask[None, :]
        mask_input = tl.load(
            mask_input_ptr + offsets,
            mask=active,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        source = tl.load(
            source_ptr + offsets,
            mask=active,
            eviction_policy="evict_first",
            other=0.0,
        ).to(tl.float32)
        selected = tl.where(mask_input <= 0.0, 0.0, source)
        acc = tl.sum(tl.where(active, selected, 0.0), axis=1)
        tl.store(out_ptr + c_offsets, acc, mask=c_mask)


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
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    arg28_1, getitem = inputs
    arg28_1 = _require_f32_tensor("arg28_1", arg28_1, INPUT_SHAPE, INPUT_STRIDE)
    getitem = _require_f32_tensor("getitem", getitem, INPUT_SHAPE, INPUT_STRIDE)
    if getitem.device != arg28_1.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return arg28_1, getitem


def oracle_forward(inputs):
    """Run the exact Repro.forward scope with a direct masked channel reduction."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    arg28_1, getitem = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=arg28_1.device,
        dtype=torch.float32,
    )
    _masked_channel_block_sum_kernel[(triton.cdiv(CHANNELS, 8),)](
        arg28_1,
        getitem,
        out,
        BATCH_=BATCH,
        CHANNELS_=CHANNELS,
        TIME_=TIME,
        BLOCK_C=8,
        BLOCK_K=8192,
        num_warps=8,
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
