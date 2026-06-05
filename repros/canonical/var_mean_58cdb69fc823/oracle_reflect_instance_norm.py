"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete per-channel population var_mean, eps=1e-5 normalization, ReLU, and two-axis reflected unsafe-index return for f32 `[1, 64, 256, 256]` by reducing channel statistics and writing the final contiguous `[1, 64, 262, 262]` layout directly, whereas Inductor currently schedules the norm-template reduction/epilogue and downstream reflected indexing as separate materialized regions; Inductor cannot do this today because its normalization scheduler does not treat small static reflection-index consumers as a direct output-layout epilogue of the normalized activation; the fix is SCHEDULER_FUSION: allow fixed reflected-index layout consumers to fuse into the normalization epilogue or be generated as a direct final-layout store."""
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
    has_stochastic_ops,
)


BATCH = 1
CHANNELS = 64
HEIGHT = 256
WIDTH = 256
PAD = 3
PADDED = 262
HW = HEIGHT * WIDTH
OUT_HW = PADDED * PADDED
TOTAL_OUT = BATCH * CHANNELS * OUT_HW
EPS = 1.0e-5
STAT_BLOCK = 1024
STAT_BLOCKS_PER_CHANNEL = HW // STAT_BLOCK
OUT_BLOCK = 512
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_channel_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        hw: tl.constexpr,
        blocks_per_channel: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        channel = pid // blocks_per_channel
        block = pid - channel * blocks_per_channel
        offsets = block * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < hw

        vals = tl.load(x_ptr + channel * hw + offsets, mask=mask, other=0.0).to(tl.float32)
        vals = tl.where(mask, vals, 0.0)
        partial_sum = tl.sum(vals, axis=0)
        partial_sumsq = tl.sum(vals * vals, axis=0)
        partial_offset = channel * blocks_per_channel + block
        tl.store(partial_sum_ptr + partial_offset, partial_sum)
        tl.store(partial_sumsq_ptr + partial_offset, partial_sumsq)

    @triton.jit
    def _final_channel_stats_kernel(
        partial_sum_ptr,
        partial_sumsq_ptr,
        mean_ptr,
        invstd_ptr,
        hw: tl.constexpr,
        blocks_per_channel: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_B)
        mask = offsets < blocks_per_channel
        partial_offsets = channel * blocks_per_channel + offsets

        partial_sum = tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        partial_sumsq = tl.load(partial_sumsq_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        total_sum = tl.sum(partial_sum, axis=0)
        total_sumsq = tl.sum(partial_sumsq, axis=0)
        mean = total_sum / hw
        var = total_sumsq / hw - mean * mean
        invstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)
        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

    @triton.jit
    def _reflect_norm_relu_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        total_out: tl.constexpr,
        hw: tl.constexpr,
        out_hw: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        pad: tl.constexpr,
        padded: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < total_out

        channel = offsets // out_hw
        spatial = offsets - channel * out_hw
        oh = spatial // padded
        ow = spatial - oh * padded

        ih = tl.where(
            oh < pad,
            pad - oh,
            tl.where(oh < height + pad, oh - pad, height * 2 + pad - 2 - oh),
        )
        iw = tl.where(
            ow < pad,
            pad - ow,
            tl.where(ow < width + pad, ow - pad, width * 2 + pad - 2 - ow),
        )
        input_offsets = channel * hw + ih * width + iw

        x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum((x - mean) * invstd, 0.0)
        tl.store(out_ptr + offsets, y, mask=mask)


def _expect_input(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects a tensor input, got {type(value)!r}")
    if tuple(value.shape) != (BATCH, CHANNELS, HEIGHT, WIDTH):
        raise ValueError(
            f"input shape {tuple(value.shape)} != {(BATCH, CHANNELS, HEIGHT, WIDTH)}"
        )
    if value.dtype != torch.float32:
        raise TypeError(f"input dtype {value.dtype} != torch.float32")
    if not value.is_contiguous():
        raise ValueError(f"input stride {tuple(value.stride())} must be contiguous")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects 1 input, got {len(inputs)}")
    return _expect_input(inputs[0])


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    x = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    normalized = torch.relu((x - mean) * torch.rsqrt(var + EPS))
    arange = torch.arange(PADDED, device=x.device, dtype=torch.int64) - PAD
    idx = WIDTH - 1 - torch.abs(WIDTH - 1 - torch.abs(arange))
    return normalized[:, :, idx, :][:, :, :, idx]


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: accepts the same single input as Repro.forward() and
    returns the same contiguous f32 `[1, 64, 262, 262]` tensor. The reflected
    row/column unsafe-index operations are represented by direct final stores.
    """
    x = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    partial_sum = torch.empty(
        (CHANNELS, STAT_BLOCKS_PER_CHANNEL),
        device=x.device,
        dtype=torch.float32,
    )
    partial_sumsq = torch.empty_like(partial_sum)
    mean = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    invstd = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (BATCH, CHANNELS, PADDED, PADDED),
        (CHANNELS * OUT_HW, OUT_HW, PADDED, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _partial_channel_stats_kernel[(CHANNELS * STAT_BLOCKS_PER_CHANNEL,)](
        x,
        partial_sum,
        partial_sumsq,
        hw=HW,
        blocks_per_channel=STAT_BLOCKS_PER_CHANNEL,
        BLOCK_M=STAT_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _final_channel_stats_kernel[(CHANNELS,)](
        partial_sum,
        partial_sumsq,
        mean,
        invstd,
        hw=HW,
        blocks_per_channel=STAT_BLOCKS_PER_CHANNEL,
        eps=EPS,
        BLOCK_B=triton.next_power_of_2(STAT_BLOCKS_PER_CHANNEL),
        num_warps=1,
        num_stages=3,
    )
    _reflect_norm_relu_kernel[(triton.cdiv(TOTAL_OUT, OUT_BLOCK),)](
        x,
        mean,
        invstd,
        out,
        total_out=TOTAL_OUT,
        hw=HW,
        out_hw=OUT_HW,
        height=HEIGHT,
        width=WIDTH,
        pad=PAD,
        padded=PADDED,
        BLOCK_M=OUT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return out


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
