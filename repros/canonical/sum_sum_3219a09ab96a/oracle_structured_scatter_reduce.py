"""
Oracle for sum_sum_3219a09ab96a

Gap diagnosis:
  Classification: SCATTER_REDUCE
  What oracle does differently: Computes the full max-pool-backward scatter_add plus two masked channel sums as a direct gather-mask-reduce over the pool offsets, including the nonzero full-value mask contribution.
  What Inductor change would fix: Recognize low-memory max-pool offset scatter-add feeding channel slices, where masks, and sibling reductions as one structured scatter-reduce producer instead of materializing the dense scatter buffer.
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 512
IN_CHANNELS = 256
OUT_CHANNELS = 128
SRC_H = 13
SRC_W = 13
DST_H = 27
DST_W = 27
KERNEL_W = 3
STRIDE = 2

SRC_HW = SRC_H * SRC_W
DST_HW = DST_H * DST_W
SRC_ELEMS = BATCH * SRC_HW
DST_ELEMS = BATCH * DST_HW

REDUCE_BLOCK = 2048
FINAL_BLOCK = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_scatter_reduce_kernel(
        src_ptr,
        offsets_ptr,
        mask_hi_ptr,
        full_ptr,
        mask_lo_ptr,
        partials_ptr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK: tl.constexpr,
        BATCH_: tl.constexpr,
        IN_CHANNELS_: tl.constexpr,
        OUT_CHANNELS_: tl.constexpr,
        SRC_W_: tl.constexpr,
        DST_W_: tl.constexpr,
        KERNEL_W_: tl.constexpr,
        STRIDE_: tl.constexpr,
        SRC_HW_: tl.constexpr,
        DST_HW_: tl.constexpr,
        SRC_ELEMS_: tl.constexpr,
        DST_ELEMS_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        elems = block_id * BLOCK + tl.arange(0, BLOCK)

        full_value = tl.load(full_ptr).to(tl.float32)

        # Part A from where(mask, full, scatter): every true mask position
        # contributes full once, independent of how many source values scatter there.
        dst_valid = elems < DST_ELEMS_
        dst_n = elems // DST_HW_
        dst_s = elems - dst_n * DST_HW_
        mask_offsets = dst_n * OUT_CHANNELS_ * DST_HW_ + channel * DST_HW_ + dst_s

        mask_hi = tl.load(mask_hi_ptr + mask_offsets, mask=dst_valid, other=0) != 0
        mask_lo = tl.load(mask_lo_ptr + mask_offsets, mask=dst_valid, other=0) != 0
        sum_hi = tl.sum(mask_hi.to(tl.float32), axis=0) * full_value
        sum_lo = tl.sum(mask_lo.to(tl.float32), axis=0) * full_value

        # Part B: source values contribute only when the mask at their decoded
        # scatter destination is false.  Output 0 consumes channels 128:256 and
        # output 1 consumes channels 0:128.
        src_valid = elems < SRC_ELEMS_
        src_n = elems // SRC_HW_
        src_s = elems - src_n * SRC_HW_
        src_h = src_s // SRC_W_
        src_w = src_s - src_h * SRC_W_

        channel_lo = channel
        channel_hi = channel + OUT_CHANNELS_
        src_base = src_n * IN_CHANNELS_ * SRC_HW_ + src_s
        off_lo = src_base + channel_lo * SRC_HW_
        off_hi = src_base + channel_hi * SRC_HW_

        pool_lo = tl.load(offsets_ptr + off_lo, mask=src_valid, other=0).to(tl.int32)
        pool_hi = tl.load(offsets_ptr + off_hi, mask=src_valid, other=0).to(tl.int32)

        pool_lo_h = pool_lo // KERNEL_W_
        pool_lo_w = pool_lo - pool_lo_h * KERNEL_W_
        pool_hi_h = pool_hi // KERNEL_W_
        pool_hi_w = pool_hi - pool_hi_h * KERNEL_W_

        dst_lo_s = (src_h * STRIDE_ + pool_lo_h) * DST_W_ + (src_w * STRIDE_ + pool_lo_w)
        dst_hi_s = (src_h * STRIDE_ + pool_hi_h) * DST_W_ + (src_w * STRIDE_ + pool_hi_w)
        dst_lo_off = src_n * OUT_CHANNELS_ * DST_HW_ + channel * DST_HW_ + dst_lo_s
        dst_hi_off = src_n * OUT_CHANNELS_ * DST_HW_ + channel * DST_HW_ + dst_hi_s

        dst_mask_lo = tl.load(mask_lo_ptr + dst_lo_off, mask=src_valid, other=1) != 0
        dst_mask_hi = tl.load(mask_hi_ptr + dst_hi_off, mask=src_valid, other=1) != 0
        src_lo = tl.load(src_ptr + off_lo, mask=src_valid & ~dst_mask_lo, other=0.0).to(tl.float32)
        src_hi = tl.load(src_ptr + off_hi, mask=src_valid & ~dst_mask_hi, other=0.0).to(tl.float32)
        sum_lo += tl.sum(src_lo, axis=0)
        sum_hi += tl.sum(src_hi, axis=0)

        partial_base = channel * NUM_BLOCKS + block_id
        tl.store(partials_ptr + partial_base, sum_hi)
        tl.store(partials_ptr + OUT_CHANNELS_ * NUM_BLOCKS + partial_base, sum_lo)

    @triton.jit
    def _finalize_reduce_kernel(
        partials_ptr,
        out_hi_ptr,
        out_lo_ptr,
        NUM_BLOCKS: tl.constexpr,
        OUT_CHANNELS_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < NUM_BLOCKS
        partial_base = channel * NUM_BLOCKS + offsets

        vals_hi = tl.load(partials_ptr + partial_base, mask=mask, other=0.0).to(tl.float32)
        vals_lo = tl.load(
            partials_ptr + OUT_CHANNELS_ * NUM_BLOCKS + partial_base,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(out_hi_ptr + channel, tl.sum(vals_hi, axis=0))
        tl.store(out_lo_ptr + channel, tl.sum(vals_lo, axis=0))


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")

    (
        getitem_36,
        arg38_1,
        arg57_1,
        full,
        arg58_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs

    if getitem_36.shape != (BATCH, IN_CHANNELS, SRC_H, SRC_W):
        raise ValueError(f"unexpected source shape: {tuple(getitem_36.shape)}")
    if arg38_1.shape != (BATCH, IN_CHANNELS, SRC_H, SRC_W):
        raise ValueError(f"unexpected offset shape: {tuple(arg38_1.shape)}")
    if arg57_1.shape != (BATCH, OUT_CHANNELS, DST_H, DST_W):
        raise ValueError(f"unexpected high-channel mask shape: {tuple(arg57_1.shape)}")
    if arg58_1.shape != (BATCH, OUT_CHANNELS, DST_H, DST_W):
        raise ValueError(f"unexpected low-channel mask shape: {tuple(arg58_1.shape)}")
    if full.shape != ():
        raise ValueError(f"unexpected full scalar shape: {tuple(full.shape)}")
    if getitem_36.dtype != torch.float32:
        raise ValueError(f"expected f32 source, got {getitem_36.dtype}")
    if arg38_1.dtype != torch.int8:
        raise ValueError(f"expected i8 pool offsets, got {arg38_1.dtype}")
    if arg57_1.dtype != torch.bool or arg58_1.dtype != torch.bool:
        raise ValueError("expected bool masks")
    if full.dtype != torch.float32:
        raise ValueError(f"expected f32 scalar full input, got {full.dtype}")
    if getitem_36.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if not all(t.is_contiguous() for t in (getitem_36, arg38_1, arg57_1, arg58_1)):
        raise ValueError("oracle expects contiguous captured tensor inputs")

    return getitem_36, arg38_1, arg57_1, full, arg58_1


def oracle_forward(inputs):
    """Run the full Repro.forward computation with a structured gather-mask-reduce."""
    src, offsets, mask_hi, full, mask_lo = _validate_inputs(inputs)

    out_hi = torch.empty((OUT_CHANNELS,), device=src.device, dtype=torch.float32)
    out_lo = torch.empty((OUT_CHANNELS,), device=src.device, dtype=torch.float32)
    num_blocks = triton.cdiv(DST_ELEMS, REDUCE_BLOCK)
    partials = torch.empty((2, OUT_CHANNELS, num_blocks), device=src.device, dtype=torch.float32)

    _partial_scatter_reduce_kernel[(OUT_CHANNELS, num_blocks)](
        src,
        offsets,
        mask_hi,
        full,
        mask_lo,
        partials,
        NUM_BLOCKS=num_blocks,
        BLOCK=REDUCE_BLOCK,
        BATCH_=BATCH,
        IN_CHANNELS_=IN_CHANNELS,
        OUT_CHANNELS_=OUT_CHANNELS,
        SRC_W_=SRC_W,
        DST_W_=DST_W,
        KERNEL_W_=KERNEL_W,
        STRIDE_=STRIDE,
        SRC_HW_=SRC_HW,
        DST_HW_=DST_HW,
        SRC_ELEMS_=SRC_ELEMS,
        DST_ELEMS_=DST_ELEMS,
        num_warps=8,
    )
    _finalize_reduce_kernel[(OUT_CHANNELS,)](
        partials,
        out_hi,
        out_lo,
        NUM_BLOCKS=num_blocks,
        OUT_CHANNELS_=OUT_CHANNELS,
        BLOCK=FINAL_BLOCK,
        num_warps=4,
    )
    return out_hi, out_lo


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
                print(f"WARNING: oracle is slower than compile " f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
