"""
Oracle for sum_sum_2a2b5ec72fea

Gap diagnosis:
  Classification: SCATTER_REDUCE
  What oracle does differently: Computes the full cropped sigmoid-BN backward tuple with Triton channel reductions, gathering the structured 113x113->112x112 crop at the point of use instead of materializing the pad/crop and reduction producers.
  What Inductor change would fix: Add a structured crop/pool-backward scatter-reduce lowering that can feed channel reductions and the dependent full-tensor BN epilogue from the same producer without generic intermediate tensors.
"""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 128
CHANNELS = 96
OUT_H = 112
OUT_W = 112
IN_H = 113
IN_W = 113
REDUCE_SIZE = BATCH * OUT_H * OUT_W
TOTAL_OUT = BATCH * CHANNELS * OUT_H * OUT_W
BN_SCALE = 6.228077168367346e-07

REDUCE_BLOCK = 4096
REDUCE_FINAL_BLOCK = 512
EPILOGUE_BLOCK = 512


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_channel_reduce_kernel(
        crop_src_ptr,
        arg180_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        partials_ptr,
        NUM_BLOCKS: tl.constexpr,
        REDUCE_SIZE_: tl.constexpr,
        OUT_H_: tl.constexpr,
        OUT_W_: tl.constexpr,
        IN_H_: tl.constexpr,
        IN_W_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)

        k_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = k_offsets < REDUCE_SIZE_

        hw = OUT_H_ * OUT_W_
        n = k_offsets // hw
        rem = k_offsets - n * hw
        h = rem // OUT_W_
        w = rem - h * OUT_W_

        out_offsets = ((n * CHANNELS_ + channel) * OUT_H_ + h) * OUT_W_ + w
        crop_offsets = ((n * CHANNELS_ + channel) * IN_H_ + h) * IN_W_ + w

        y = tl.load(arg180_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
        crop = tl.load(crop_src_ptr + crop_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)

        centered = y - mean
        affine = centered * invstd * weight + bias
        sigmoid = 1.0 / (tl.exp(-affine) + 1.0)
        producer = crop * sigmoid * (affine * (1.0 - sigmoid) + 1.0)
        producer = tl.where(mask, producer, 0.0)

        sum0 = tl.sum(producer, axis=0)
        sum1 = tl.sum(producer * centered, axis=0)
        partial_offset = channel * NUM_BLOCKS + block_id
        tl.store(partials_ptr + partial_offset, sum0)
        tl.store(partials_ptr + CHANNELS_ * NUM_BLOCKS + partial_offset, sum1)

    @triton.jit
    def _finalize_channel_reduce_kernel(
        partials_ptr,
        stats_ptr,
        out1_ptr,
        invstd_ptr,
        NUM_BLOCKS: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < NUM_BLOCKS
        base = channel * NUM_BLOCKS + offsets

        sum0 = tl.sum(tl.load(partials_ptr + base, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum1 = tl.sum(
            tl.load(partials_ptr + CHANNELS_ * NUM_BLOCKS + base, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        tl.store(stats_ptr + channel, sum0)
        tl.store(stats_ptr + CHANNELS_ + channel, sum1)
        tl.store(out1_ptr + channel, sum1 * invstd)

    @triton.jit
    def _epilogue_kernel(
        crop_src_ptr,
        arg180_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out0_ptr,
        REDUCE_SIZE_: tl.constexpr,
        OUT_H_: tl.constexpr,
        OUT_W_: tl.constexpr,
        IN_H_: tl.constexpr,
        IN_W_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BN_SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        k_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = k_offsets < REDUCE_SIZE_

        hw = OUT_H_ * OUT_W_
        n = k_offsets // hw
        rem = k_offsets - n * hw
        h = rem // OUT_W_
        w = rem - h * OUT_W_

        out_offsets = ((n * CHANNELS_ + channel) * OUT_H_ + h) * OUT_W_ + w
        crop_offsets = ((n * CHANNELS_ + channel) * IN_H_ + h) * IN_W_ + w

        y = tl.load(arg180_ptr + out_offsets, mask=mask, other=0.0).to(tl.float32)
        crop = tl.load(crop_src_ptr + crop_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        sum0 = tl.load(stats_ptr + channel).to(tl.float32)
        sum1 = tl.load(stats_ptr + CHANNELS_ + channel).to(tl.float32)

        centered = y - mean
        affine = centered * invstd * weight + bias
        sigmoid = 1.0 / (tl.exp(-affine) + 1.0)
        producer = crop * sigmoid * (affine * (1.0 - sigmoid) + 1.0)

        mean0 = sum0 * BN_SCALE_
        mean1_inv2 = sum1 * BN_SCALE_ * invstd * invstd
        out = (producer - centered * mean1_inv2 - mean0) * (invstd * weight)
        tl.store(out0_ptr + out_offsets, out, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"expected 6 inputs, got {len(inputs)}")

    crop_src, arg180, mean, invstd, weight, bias = inputs
    expected = (
        (BATCH, CHANNELS, IN_H, IN_W),
        (BATCH, CHANNELS, OUT_H, OUT_W),
        (1, CHANNELS, 1, 1),
        (1, CHANNELS, 1, 1),
        (CHANNELS,),
        (CHANNELS,),
    )
    actual = tuple(tuple(inp.shape) for inp in inputs)
    if actual != expected:
        raise ValueError(f"unsupported input shapes: expected {expected}, got {actual}")
    if any(inp.dtype != torch.float32 for inp in inputs):
        raise ValueError("this oracle expects float32 inputs")
    if not all(inp.is_cuda for inp in inputs):
        raise ValueError("this oracle expects CUDA inputs")
    if not all(inp.is_contiguous() for inp in inputs):
        raise ValueError("this oracle expects contiguous inputs")
    return crop_src, arg180, mean, invstd, weight, bias


@oracle_impl(hardware="H100", shapes="(T([128, 96, 113, 113], f32), T([128, 96, 112, 112], f32), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation with structured Triton kernels."""
    crop_src, arg180, mean, invstd, weight, bias = _validate_inputs(inputs)

    num_blocks = triton.cdiv(REDUCE_SIZE, REDUCE_BLOCK)
    out0 = torch.empty_like(arg180)
    out1 = torch.empty_like(weight)
    partials = torch.empty((2, CHANNELS, num_blocks), device=arg180.device, dtype=torch.float32)
    stats = torch.empty((2, CHANNELS), device=arg180.device, dtype=torch.float32)

    _partial_channel_reduce_kernel[(CHANNELS, num_blocks)](
        crop_src,
        arg180,
        mean,
        invstd,
        weight,
        bias,
        partials,
        NUM_BLOCKS=num_blocks,
        REDUCE_SIZE_=REDUCE_SIZE,
        OUT_H_=OUT_H,
        OUT_W_=OUT_W,
        IN_H_=IN_H,
        IN_W_=IN_W,
        CHANNELS_=CHANNELS,
        BLOCK=REDUCE_BLOCK,
        num_warps=8,
    )
    _finalize_channel_reduce_kernel[(CHANNELS,)](
        partials,
        stats,
        out1,
        invstd,
        NUM_BLOCKS=num_blocks,
        CHANNELS_=CHANNELS,
        BLOCK=REDUCE_FINAL_BLOCK,
        num_warps=4,
    )
    _epilogue_kernel[(CHANNELS, triton.cdiv(REDUCE_SIZE, EPILOGUE_BLOCK))](
        crop_src,
        arg180,
        mean,
        invstd,
        weight,
        bias,
        stats,
        out0,
        REDUCE_SIZE_=REDUCE_SIZE,
        OUT_H_=OUT_H,
        OUT_W_=OUT_W,
        IN_H_=IN_H,
        IN_W_=IN_W,
        CHANNELS_=CHANNELS,
        BN_SCALE_=BN_SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )
    return out0, out1


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
