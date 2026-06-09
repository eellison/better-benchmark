"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet batchnorm-backward-like captured scope, including the add/copy/clone value producer, both channel-reduction groups, the dependent vector outputs, and the full tensor epilogues with the sliced channels-last output layout, using shared multi-accumulator split-K reductions where the two channel groups overlap, whereas Inductor already lands within noise of this full-scope Triton floor despite scheduling the reductions and surrounding pointwise/layout work generically; Inductor cannot materially improve this case through a local fusion rewrite because the required tensor reads, coefficient-dependent rereads, and output stores dominate the measured runtime; the fix is BANDWIDTH_BOUND: treat this as an at-floor multi-output reduction instance unless broader memory-traffic or launch-overhead improvements move the baseline."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 512
CHANNELS_112 = 112
CHANNELS_56 = 56
SPATIAL = 14 * 14
REDUCTION_K = BATCH * SPATIAL
TOTAL_112 = BATCH * CHANNELS_112 * SPATIAL
TOTAL_56 = BATCH * CHANNELS_56 * SPATIAL
NORM_SCALE = 9.964923469387754e-06
BLOCK_K = 1024
BLOCK_FINAL = 128
BLOCK_POINTWISE = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_reduce_112_kernel(
        clone_5_ptr,
        getitem_108_ptr,
        arg367_ptr,
        arg499_ptr,
        arg362_ptr,
        arg501_ptr,
        partial_x_ptr,
        partial_x_sub_ptr,
        partial_x_56_ptr,
        partial_x_sub_56_ptr,
        NBLOCKS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        k = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = k < 100352
        batch = k // 196
        spatial = k - batch * 196
        dense_offset = batch * 21952 + channel * 196 + spatial

        x = (
            tl.load(clone_5_ptr + dense_offset, mask=mask, other=0.0).to(tl.float32)
            + tl.load(getitem_108_ptr + dense_offset, mask=mask, other=0.0).to(
                tl.float32
            )
        )
        centered = (
            tl.load(arg367_ptr + dense_offset, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg499_ptr + channel).to(tl.float32)
        )

        partial_offset = channel * NBLOCKS + block_id
        tl.store(partial_x_ptr + partial_offset, tl.sum(tl.where(mask, x, 0.0), axis=0))
        tl.store(
            partial_x_sub_ptr + partial_offset,
            tl.sum(tl.where(mask, x * centered, 0.0), axis=0),
        )

        channel56 = channel - 56
        is_second_half = channel >= 56
        dense56_offset = batch * 10976 + channel56 * 196 + spatial
        mask56 = mask & is_second_half
        centered56 = (
            tl.load(arg362_ptr + dense56_offset, mask=mask56, other=0.0).to(tl.float32)
            - tl.load(arg501_ptr + channel56, mask=is_second_half, other=0.0).to(
                tl.float32
            )
        )
        partial56_offset = channel56 * NBLOCKS + block_id
        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        tl.store(partial_x_56_ptr + partial56_offset, sum_x, mask=is_second_half)
        tl.store(
            partial_x_sub_56_ptr + partial56_offset,
            tl.sum(tl.where(mask56, x * centered56, 0.0), axis=0),
            mask=is_second_half,
        )

    @triton.jit
    def _finalize_112_kernel(
        partial_x_ptr,
        partial_x_sub_ptr,
        arg368_ptr,
        arg121_ptr,
        mean_ptr,
        coeff_ptr,
        weight_ptr,
        out1_ptr,
        NBLOCKS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < NBLOCKS
        partial_offsets = channel * NBLOCKS + offsets
        sum_x = tl.sum(
            tl.load(partial_x_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        sum_x_sub = tl.sum(
            tl.load(partial_x_sub_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        arg368 = tl.load(arg368_ptr + channel).to(tl.float32)
        mean = sum_x * 9.964923469387754e-06
        coeff = sum_x_sub * 9.964923469387754e-06 * arg368 * arg368
        weight = arg368 * tl.load(arg121_ptr + channel).to(tl.float32)
        tl.store(mean_ptr + channel, mean)
        tl.store(coeff_ptr + channel, coeff)
        tl.store(weight_ptr + channel, weight)
        tl.store(out1_ptr + channel, sum_x_sub * arg368)

    @triton.jit
    def _finalize_56_kernel(
        partial_x_ptr,
        partial_x_sub_ptr,
        arg363_ptr,
        arg117_ptr,
        mean_ptr,
        coeff_ptr,
        weight_ptr,
        out3_ptr,
        NBLOCKS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < NBLOCKS
        partial_offsets = channel * NBLOCKS + offsets
        sum_x = tl.sum(
            tl.load(partial_x_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        sum_x_sub = tl.sum(
            tl.load(partial_x_sub_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        arg363 = tl.load(arg363_ptr + channel).to(tl.float32)
        mean = sum_x * 9.964923469387754e-06
        coeff = sum_x_sub * 9.964923469387754e-06 * arg363 * arg363
        weight = arg363 * tl.load(arg117_ptr + channel).to(tl.float32)
        tl.store(mean_ptr + channel, mean)
        tl.store(coeff_ptr + channel, coeff)
        tl.store(weight_ptr + channel, weight)
        tl.store(out3_ptr + channel, sum_x_sub * arg363)

    @triton.jit
    def _pointwise_112_kernel(
        clone_5_ptr,
        getitem_108_ptr,
        arg367_ptr,
        arg499_ptr,
        mean_ptr,
        coeff_ptr,
        weight_ptr,
        out0_ptr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < 11239424
        channel = (offsets // 196) % 112
        x = (
            tl.load(clone_5_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(getitem_108_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        centered = (
            tl.load(arg367_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg499_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        )
        value = (
            x
            - centered * tl.load(coeff_ptr + channel, mask=mask, other=0.0).to(
                tl.float32
            )
            - tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        ) * tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tl.store(out0_ptr + offsets, value, mask=mask)

    @triton.jit
    def _pointwise_56_kernel(
        clone_5_ptr,
        getitem_108_ptr,
        arg362_ptr,
        arg501_ptr,
        mean_ptr,
        coeff_ptr,
        weight_ptr,
        out2_ptr,
        BLOCK: tl.constexpr,
    ):
        physical_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = physical_offsets < 5619712
        channel = physical_offsets % 56
        tmp = physical_offsets // 56
        width = tmp % 14
        tmp = tmp // 14
        height = tmp % 14
        batch = tmp // 14
        spatial = height * 14 + width
        source_offset = (
            batch * 21952 + (channel + 56) * 196 + spatial
        )
        dense56_offset = batch * 10976 + channel * 196 + spatial

        x = (
            tl.load(clone_5_ptr + source_offset, mask=mask, other=0.0).to(tl.float32)
            + tl.load(getitem_108_ptr + source_offset, mask=mask, other=0.0).to(
                tl.float32
            )
        )
        centered = (
            tl.load(arg362_ptr + dense56_offset, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg501_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        )
        value = (
            x
            - centered * tl.load(coeff_ptr + channel, mask=mask, other=0.0).to(
                tl.float32
            )
            - tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        ) * tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tl.store(out2_ptr + physical_offsets, value, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([512, 112, 14, 14], f32), T([512, 112, 14, 14], f32), T([512, 112, 14, 14], f32), T([1, 112, 1, 1], f32), T([112], f32), T([112], f32), T([512, 56, 14, 14], f32), T([1, 56, 1, 1], f32), T([56], f32), T([56], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation with Triton kernels."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        clone_5,
        getitem_108,
        arg367_1,
        arg499_1,
        arg368_1,
        arg121_1,
        arg362_1,
        arg501_1,
        arg363_1,
        arg117_1,
    ) = inputs

    nblocks = triton.cdiv(REDUCTION_K, BLOCK_K)
    partial_x_112 = torch.empty((CHANNELS_112, nblocks), device=clone_5.device, dtype=torch.float32)
    partial_x_sub_112 = torch.empty_like(partial_x_112)
    partial_x_56 = torch.empty((CHANNELS_56, nblocks), device=clone_5.device, dtype=torch.float32)
    partial_x_sub_56 = torch.empty_like(partial_x_56)

    mean_112 = torch.empty((CHANNELS_112,), device=clone_5.device, dtype=torch.float32)
    coeff_112 = torch.empty_like(mean_112)
    weight_112 = torch.empty_like(mean_112)
    out1 = torch.empty_like(arg368_1)

    mean_56 = torch.empty((CHANNELS_56,), device=clone_5.device, dtype=torch.float32)
    coeff_56 = torch.empty_like(mean_56)
    weight_56 = torch.empty_like(mean_56)
    out3 = torch.empty_like(arg363_1)

    out0 = torch.empty_like(clone_5)
    out2 = torch.empty_strided(
        (BATCH, CHANNELS_56, 14, 14),
        (CHANNELS_56 * SPATIAL, 1, 14 * CHANNELS_56, CHANNELS_56),
        device=clone_5.device,
        dtype=torch.float32,
    )

    _partial_reduce_112_kernel[(CHANNELS_112, nblocks)](
        clone_5,
        getitem_108,
        arg367_1,
        arg499_1,
        arg362_1,
        arg501_1,
        partial_x_112,
        partial_x_sub_112,
        partial_x_56,
        partial_x_sub_56,
        NBLOCKS=nblocks,
        BLOCK=BLOCK_K,
    )
    _finalize_112_kernel[(CHANNELS_112,)](
        partial_x_112,
        partial_x_sub_112,
        arg368_1,
        arg121_1,
        mean_112,
        coeff_112,
        weight_112,
        out1,
        NBLOCKS=nblocks,
        BLOCK=BLOCK_FINAL,
    )
    _finalize_56_kernel[(CHANNELS_56,)](
        partial_x_56,
        partial_x_sub_56,
        arg363_1,
        arg117_1,
        mean_56,
        coeff_56,
        weight_56,
        out3,
        NBLOCKS=nblocks,
        BLOCK=BLOCK_FINAL,
    )
    _pointwise_112_kernel[(triton.cdiv(TOTAL_112, BLOCK_POINTWISE),)](
        clone_5,
        getitem_108,
        arg367_1,
        arg499_1,
        mean_112,
        coeff_112,
        weight_112,
        out0,
        BLOCK=BLOCK_POINTWISE,
    )
    _pointwise_56_kernel[(triton.cdiv(TOTAL_56, BLOCK_POINTWISE),)](
        clone_5,
        getitem_108,
        arg362_1,
        arg501_1,
        mean_56,
        coeff_56,
        weight_56,
        out2,
        BLOCK=BLOCK_POINTWISE,
    )

    return (out0, out1, out2, out3)


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
