"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete `sum_sum_sum_718a025684ba` GhostNet dual batch-norm-backward return tuple by reducing the shared 80-channel add/copy/clone producer and its high-channel 40-channel slice in one split-K Triton reduction, then using the finalized channel summaries to write the contiguous full-gradient tensor, the channels-last slice-gradient tensor, and both scale-gradient vectors, whereas Inductor currently schedules the memory-format copy/clone/slice path, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogues as separate generic pointwise/reduction regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that shares overlapping channel partials across layout-changing producers and dependent side outputs; the fix is COOPERATIVE_SPLIT_K: teach Inductor to tile the `(N, H, W)` reduction domain for compatible channel reductions, accumulate the full and sliced branch summaries together, and fuse the downstream tensor/vector epilogues with layout-aware stores."""
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


BATCH = 512
CHANNELS_FULL = 80
CHANNELS_SLICE = 40
SLICE_START = 40
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
REDUCTION_K = BATCH * HW
TOTAL_FULL = BATCH * CHANNELS_FULL * HW
TOTAL_SLICE = BATCH * CHANNELS_SLICE * HW
NORM_SCALE = 9.964923469387754e-06
FULL_OUT_STRIDE = (CHANNELS_FULL * HW, HW, WIDTH, 1)
SLICE_OUT_STRIDE = (
    CHANNELS_SLICE * HW,
    1,
    WIDTH * CHANNELS_SLICE,
    CHANNELS_SLICE,
)
REDUCE_BLOCK_K = 1024
FINAL_BLOCK_K = 128
POINTWISE_BLOCK = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _partial_reduce_kernel(
        clone_ptr,
        getitem_ptr,
        rhs_full_ptr,
        mean_full_ptr,
        rhs_slice_ptr,
        mean_slice_ptr,
        partial_x_full_ptr,
        partial_x_rhs_full_ptr,
        partial_x_slice_ptr,
        partial_x_rhs_slice_ptr,
        NBLOCKS: tl.constexpr,
        REDUCTION_K_: tl.constexpr,
        CHANNELS_FULL_: tl.constexpr,
        CHANNELS_SLICE_: tl.constexpr,
        SLICE_START_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        k = block_id * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < REDUCTION_K_
        batch = k // HW_
        spatial = k - batch * HW_
        full_offset = batch * (CHANNELS_FULL_ * HW_) + channel * HW_ + spatial

        x = (
            tl.load(clone_ptr + full_offset, mask=mask, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + full_offset, mask=mask, other=0.0).to(tl.float32)
        )
        centered_full = (
            tl.load(rhs_full_ptr + full_offset, mask=mask, other=0.0).to(tl.float32)
            - tl.load(mean_full_ptr + channel).to(tl.float32)
        )
        partial_full_offset = channel * NBLOCKS + block_id
        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        tl.store(partial_x_full_ptr + partial_full_offset, sum_x)
        tl.store(
            partial_x_rhs_full_ptr + partial_full_offset,
            tl.sum(tl.where(mask, x * centered_full, 0.0), axis=0),
        )

        channel_slice = channel - SLICE_START_
        in_slice = channel >= SLICE_START_
        slice_offset = (
            batch * (CHANNELS_SLICE_ * HW_) + channel_slice * HW_ + spatial
        )
        mask_slice = mask & in_slice
        centered_slice = (
            tl.load(rhs_slice_ptr + slice_offset, mask=mask_slice, other=0.0).to(
                tl.float32
            )
            - tl.load(mean_slice_ptr + channel_slice, mask=in_slice, other=0.0).to(
                tl.float32
            )
        )
        partial_slice_offset = channel_slice * NBLOCKS + block_id
        tl.store(partial_x_slice_ptr + partial_slice_offset, sum_x, mask=in_slice)
        tl.store(
            partial_x_rhs_slice_ptr + partial_slice_offset,
            tl.sum(tl.where(mask_slice, x * centered_slice, 0.0), axis=0),
            mask=in_slice,
        )

    @triton.jit
    def _finalize_full_kernel(
        partial_x_ptr,
        partial_x_rhs_ptr,
        rsqrt_ptr,
        weight_arg_ptr,
        mean_term_ptr,
        coeff_ptr,
        fused_weight_ptr,
        out_vec_ptr,
        NBLOCKS: tl.constexpr,
        NORM_SCALE_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < NBLOCKS
        partial_offsets = channel * NBLOCKS + offsets
        sum_x = tl.sum(
            tl.load(partial_x_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        sum_x_rhs = tl.sum(
            tl.load(partial_x_rhs_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        rsqrt = tl.load(rsqrt_ptr + channel).to(tl.float32)
        weight = tl.load(weight_arg_ptr + channel).to(tl.float32)
        tl.store(mean_term_ptr + channel, sum_x * NORM_SCALE_)
        tl.store(coeff_ptr + channel, sum_x_rhs * NORM_SCALE_ * rsqrt * rsqrt)
        tl.store(fused_weight_ptr + channel, rsqrt * weight)
        tl.store(out_vec_ptr + channel, sum_x_rhs * rsqrt)

    @triton.jit
    def _finalize_slice_kernel(
        partial_x_ptr,
        partial_x_rhs_ptr,
        rsqrt_ptr,
        weight_arg_ptr,
        mean_term_ptr,
        coeff_ptr,
        fused_weight_ptr,
        out_vec_ptr,
        NBLOCKS: tl.constexpr,
        NORM_SCALE_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < NBLOCKS
        partial_offsets = channel * NBLOCKS + offsets
        sum_x = tl.sum(
            tl.load(partial_x_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        sum_x_rhs = tl.sum(
            tl.load(partial_x_rhs_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        rsqrt = tl.load(rsqrt_ptr + channel).to(tl.float32)
        weight = tl.load(weight_arg_ptr + channel).to(tl.float32)
        tl.store(mean_term_ptr + channel, sum_x * NORM_SCALE_)
        tl.store(coeff_ptr + channel, sum_x_rhs * NORM_SCALE_ * rsqrt * rsqrt)
        tl.store(fused_weight_ptr + channel, rsqrt * weight)
        tl.store(out_vec_ptr + channel, sum_x_rhs * rsqrt)

    @triton.jit
    def _pointwise_full_kernel(
        clone_ptr,
        getitem_ptr,
        rhs_ptr,
        mean_ptr,
        mean_term_ptr,
        coeff_ptr,
        fused_weight_ptr,
        out_ptr,
        TOTAL_FULL_: tl.constexpr,
        CHANNELS_FULL_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL_FULL_
        channel = (offsets // HW_) % CHANNELS_FULL_
        x = (
            tl.load(clone_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        centered = (
            tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            - tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        )
        value = (
            x
            - centered * tl.load(coeff_ptr + channel, mask=mask, other=0.0).to(
                tl.float32
            )
            - tl.load(mean_term_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        ) * tl.load(fused_weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, value, mask=mask)

    @triton.jit
    def _pointwise_slice_kernel(
        clone_ptr,
        getitem_ptr,
        rhs_ptr,
        mean_ptr,
        mean_term_ptr,
        coeff_ptr,
        fused_weight_ptr,
        out_ptr,
        TOTAL_SLICE_: tl.constexpr,
        CHANNELS_FULL_: tl.constexpr,
        CHANNELS_SLICE_: tl.constexpr,
        SLICE_START_: tl.constexpr,
        HEIGHT_: tl.constexpr,
        WIDTH_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        physical_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = physical_offsets < TOTAL_SLICE_
        channel = physical_offsets % CHANNELS_SLICE_
        tmp = physical_offsets // CHANNELS_SLICE_
        width = tmp % WIDTH_
        tmp = tmp // WIDTH_
        height = tmp % HEIGHT_
        batch = tmp // HEIGHT_
        spatial = height * WIDTH_ + width
        source_offset = (
            batch * (CHANNELS_FULL_ * HW_) + (channel + SLICE_START_) * HW_ + spatial
        )
        rhs_offset = batch * (CHANNELS_SLICE_ * HW_) + channel * HW_ + spatial

        x = (
            tl.load(clone_ptr + source_offset, mask=mask, other=0.0).to(tl.float32)
            + tl.load(getitem_ptr + source_offset, mask=mask, other=0.0).to(
                tl.float32
            )
        )
        centered = (
            tl.load(rhs_ptr + rhs_offset, mask=mask, other=0.0).to(tl.float32)
            - tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        )
        value = (
            x
            - centered * tl.load(coeff_ptr + channel, mask=mask, other=0.0).to(
                tl.float32
            )
            - tl.load(mean_term_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        ) * tl.load(fused_weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + physical_offsets, value, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([512, 80, 14, 14], f32), T([512, 80, 14, 14], f32), T([512, 80, 14, 14], f32), T([1, 80, 1, 1], f32), T([80], f32), T([80], f32), T([512, 40, 14, 14], f32), T([1, 40, 1, 1], f32), T([40], f32), T([40], f32))")
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
        raise RuntimeError("Triton is required for this oracle")

    (
        clone_9,
        getitem_168,
        arg306_1,
        arg513_1,
        arg307_1,
        arg79_1,
        arg301_1,
        arg515_1,
        arg302_1,
        arg75_1,
    ) = inputs

    assert clone_9.shape == (BATCH, CHANNELS_FULL, HEIGHT, WIDTH)
    assert getitem_168.shape == (BATCH, CHANNELS_FULL, HEIGHT, WIDTH)
    assert arg306_1.shape == (BATCH, CHANNELS_FULL, HEIGHT, WIDTH)
    assert arg513_1.shape == (1, CHANNELS_FULL, 1, 1)
    assert arg307_1.shape == (CHANNELS_FULL,)
    assert arg79_1.shape == (CHANNELS_FULL,)
    assert arg301_1.shape == (BATCH, CHANNELS_SLICE, HEIGHT, WIDTH)
    assert arg515_1.shape == (1, CHANNELS_SLICE, 1, 1)
    assert arg302_1.shape == (CHANNELS_SLICE,)
    assert arg75_1.shape == (CHANNELS_SLICE,)

    device = clone_9.device
    nblocks = triton.cdiv(REDUCTION_K, REDUCE_BLOCK_K)

    partial_x_full = torch.empty(
        (CHANNELS_FULL, nblocks), device=device, dtype=torch.float32
    )
    partial_x_rhs_full = torch.empty_like(partial_x_full)
    partial_x_slice = torch.empty(
        (CHANNELS_SLICE, nblocks), device=device, dtype=torch.float32
    )
    partial_x_rhs_slice = torch.empty_like(partial_x_slice)

    mean_term_full = torch.empty((CHANNELS_FULL,), device=device, dtype=torch.float32)
    coeff_full = torch.empty_like(mean_term_full)
    fused_weight_full = torch.empty_like(mean_term_full)
    out_vec_full = torch.empty_like(arg307_1)

    mean_term_slice = torch.empty(
        (CHANNELS_SLICE,), device=device, dtype=torch.float32
    )
    coeff_slice = torch.empty_like(mean_term_slice)
    fused_weight_slice = torch.empty_like(mean_term_slice)
    out_vec_slice = torch.empty_like(arg302_1)

    out_full = torch.empty_strided(
        (BATCH, CHANNELS_FULL, HEIGHT, WIDTH),
        FULL_OUT_STRIDE,
        device=device,
        dtype=torch.float32,
    )
    out_slice = torch.empty_strided(
        (BATCH, CHANNELS_SLICE, HEIGHT, WIDTH),
        SLICE_OUT_STRIDE,
        device=device,
        dtype=torch.float32,
    )

    _partial_reduce_kernel[(CHANNELS_FULL, nblocks)](
        clone_9,
        getitem_168,
        arg306_1,
        arg513_1,
        arg301_1,
        arg515_1,
        partial_x_full,
        partial_x_rhs_full,
        partial_x_slice,
        partial_x_rhs_slice,
        NBLOCKS=nblocks,
        REDUCTION_K_=REDUCTION_K,
        CHANNELS_FULL_=CHANNELS_FULL,
        CHANNELS_SLICE_=CHANNELS_SLICE,
        SLICE_START_=SLICE_START,
        HW_=HW,
        BLOCK_K=REDUCE_BLOCK_K,
        num_warps=4,
    )
    _finalize_full_kernel[(CHANNELS_FULL,)](
        partial_x_full,
        partial_x_rhs_full,
        arg307_1,
        arg79_1,
        mean_term_full,
        coeff_full,
        fused_weight_full,
        out_vec_full,
        NBLOCKS=nblocks,
        NORM_SCALE_=NORM_SCALE,
        BLOCK_K=FINAL_BLOCK_K,
        num_warps=4,
    )
    _finalize_slice_kernel[(CHANNELS_SLICE,)](
        partial_x_slice,
        partial_x_rhs_slice,
        arg302_1,
        arg75_1,
        mean_term_slice,
        coeff_slice,
        fused_weight_slice,
        out_vec_slice,
        NBLOCKS=nblocks,
        NORM_SCALE_=NORM_SCALE,
        BLOCK_K=FINAL_BLOCK_K,
        num_warps=4,
    )
    _pointwise_full_kernel[(triton.cdiv(TOTAL_FULL, POINTWISE_BLOCK),)](
        clone_9,
        getitem_168,
        arg306_1,
        arg513_1,
        mean_term_full,
        coeff_full,
        fused_weight_full,
        out_full,
        TOTAL_FULL_=TOTAL_FULL,
        CHANNELS_FULL_=CHANNELS_FULL,
        HW_=HW,
        BLOCK=POINTWISE_BLOCK,
        num_warps=4,
    )
    _pointwise_slice_kernel[(triton.cdiv(TOTAL_SLICE, POINTWISE_BLOCK),)](
        clone_9,
        getitem_168,
        arg301_1,
        arg515_1,
        mean_term_slice,
        coeff_slice,
        fused_weight_slice,
        out_slice,
        TOTAL_SLICE_=TOTAL_SLICE,
        CHANNELS_FULL_=CHANNELS_FULL,
        CHANNELS_SLICE_=CHANNELS_SLICE,
        SLICE_START_=SLICE_START,
        HEIGHT_=HEIGHT,
        WIDTH_=WIDTH,
        HW_=HW,
        BLOCK=POINTWISE_BLOCK,
        num_warps=4,
    )

    return out_full, out_vec_full, out_slice, out_vec_slice


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
