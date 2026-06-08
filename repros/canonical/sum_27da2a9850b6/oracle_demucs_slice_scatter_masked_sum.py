"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Demucs slice-scatter plus sibling masked-sum scope by materializing the required zero-padded `[64,64,95696]` output and accumulating the `where(arg30_1, 0.0, getitem_3).sum(dim=[0,2])` result while reading the central f32 input once in one Triton copy/reduce pass, whereas Inductor currently treats the slice-scatter materialization and masked reduction as separate scheduled consumers of `getitem_3`; Inductor cannot do this today because its scheduler does not fuse a full materializing pointwise/scatter-style output with a sibling reduction over the same producer into a multi-output copy-plus-reduction schedule; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to emit a fused pad-copy plus reduction producer when the scatter destination is zero-filled and the reduction is over the same source tensor."""
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


BATCH = 64
CHANNELS = 64
IN_T = 92844
PAD = 1426
OUT_T = 95696
INPUT_SHAPE = (BATCH, CHANNELS, IN_T)
INPUT_STRIDE = (CHANNELS * IN_T, IN_T, 1)
OUTPUT_SHAPE = (BATCH, CHANNELS, OUT_T)
OUTPUT_STRIDE = (CHANNELS * OUT_T, OUT_T, 1)
REDUCE_SHAPE = (CHANNELS,)
REDUCE_STRIDE = (1,)
COPY_REDUCE_BLOCK_T = 4096
TILES_T = triton.cdiv(IN_T, COPY_REDUCE_BLOCK_T) if triton is not None else 23
PARTIALS_PER_CHANNEL = BATCH * TILES_T
FINAL_BLOCK = 2048
PAD_BLOCK = 2048


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_pad_kernel(
        out_ptr,
        pad: tl.constexpr,
        out_t: tl.constexpr,
        BLOCK_PAD: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_PAD)
        mask = offsets < pad
        base = row * out_t
        zero = tl.full((BLOCK_PAD,), 0.0, tl.float32)
        tl.store(out_ptr + base + offsets, zero, mask=mask)
        tl.store(out_ptr + base + (out_t - pad) + offsets, zero, mask=mask)

    @triton.jit
    def _copy_reduce_kernel(
        x_ptr,
        mask_ptr,
        out_ptr,
        partial_ptr,
        in_t: tl.constexpr,
        out_t: tl.constexpr,
        channels: tl.constexpr,
        pad: tl.constexpr,
        tiles_t: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        channel = tl.program_id(0)
        batch = tl.program_id(1)
        tile = tl.program_id(2)
        offsets_t = tile * BLOCK_T + tl.arange(0, BLOCK_T)
        valid = offsets_t < in_t

        input_base = (batch * channels + channel) * in_t
        output_base = (batch * channels + channel) * out_t + pad
        input_offsets = input_base + offsets_t
        output_offsets = output_base + offsets_t

        values = tl.load(x_ptr + input_offsets, mask=valid, other=0.0).to(tl.float32)
        predicate = tl.load(mask_ptr + input_offsets, mask=valid, other=1)
        masked_values = tl.where(predicate, 0.0, values)
        partial_sum = tl.sum(masked_values, axis=0)

        tl.store(out_ptr + output_offsets, values, mask=valid)
        partial_offset = (channel * 64 + batch) * tiles_t + tile
        tl.store(partial_ptr + partial_offset, partial_sum)

    @triton.jit
    def _final_sum_kernel(
        partial_ptr,
        sum_ptr,
        partials_per_channel: tl.constexpr,
        BLOCK_PARTIALS: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_PARTIALS)
        valid = offsets < partials_per_channel
        values = tl.load(
            partial_ptr + channel * partials_per_channel + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(values, axis=0)
        tl.store(sum_ptr + channel, total)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA, got {value.device}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")
    getitem_3 = _expect_tensor(
        "getitem_3",
        inputs[0],
        INPUT_SHAPE,
        INPUT_STRIDE,
        torch.float32,
    )
    arg30_1 = _expect_tensor(
        "arg30_1",
        inputs[1],
        INPUT_SHAPE,
        INPUT_STRIDE,
        torch.bool,
    )
    if arg30_1.device != getitem_3.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return getitem_3, arg30_1


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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

    getitem_3, arg30_1 = _validate_inputs(inputs)
    padded = torch.empty(OUTPUT_SHAPE, device=getitem_3.device, dtype=torch.float32)
    partials = torch.empty(
        (CHANNELS, BATCH, TILES_T),
        device=getitem_3.device,
        dtype=torch.float32,
    )
    masked_sum = torch.empty(REDUCE_SHAPE, device=getitem_3.device, dtype=torch.float32)

    _zero_pad_kernel[(BATCH * CHANNELS,)](
        padded,
        PAD,
        OUT_T,
        BLOCK_PAD=PAD_BLOCK,
        num_warps=4,
    )
    _copy_reduce_kernel[(CHANNELS, BATCH, TILES_T)](
        getitem_3,
        arg30_1,
        padded,
        partials,
        IN_T,
        OUT_T,
        CHANNELS,
        PAD,
        TILES_T,
        BLOCK_T=COPY_REDUCE_BLOCK_T,
        num_warps=8,
    )
    _final_sum_kernel[(CHANNELS,)](
        partials,
        masked_sum,
        PARTIALS_PER_CHANNEL,
        BLOCK_PARTIALS=FINAL_BLOCK,
        num_warps=8,
    )

    if tuple(padded.stride()) != OUTPUT_STRIDE:
        raise RuntimeError(f"unexpected padded stride: {tuple(padded.stride())}")
    if tuple(masked_sum.stride()) != REDUCE_STRIDE:
        raise RuntimeError(f"unexpected masked_sum stride: {tuple(masked_sum.stride())}")
    return padded, masked_sum


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
