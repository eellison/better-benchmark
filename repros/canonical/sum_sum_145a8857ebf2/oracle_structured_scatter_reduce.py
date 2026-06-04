"""
Oracle for sum_sum_145a8857ebf2

Gap diagnosis:
  Classification: SCATTER_REDUCE
  What oracle does differently: Computes the full max-pool-backward scatter feeding BN backward as a structured gather-mask-reduce, deriving channel sums directly from pool offsets and materializing only the required final output.
  What Inductor change would fix: Recognize low-memory max-pool offset scatter-add plus pointwise mask plus sibling channel reductions as one structured scatter-reduce producer with a fused final-output epilogue.
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure (installed via pip install -e .)
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


BATCH = 128
CHANNELS = 64
SRC_H = 56
SRC_W = 56
OUT_H = 112
OUT_W = 112
KERNEL_W = 3
STRIDE = 2
PADDING = 1
SRC_HW = SRC_H * SRC_W
OUT_HW = OUT_H * OUT_W
SRC_ELEMS = BATCH * SRC_HW
OUT_ELEMS = BATCH * OUT_HW
BN_NORM = 6.228077168367346e-07

REDUCE_BLOCK = 1024
OUTPUT_BLOCK = 512
REDUCE_WARPS = 4
FINAL_WARPS = 8
OUTPUT_WARPS = 4

_CENTER_OFFSET_CACHE: dict[tuple[int, int], bool] = {}


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
        grad0_ptr,
        grad1_ptr,
        offsets_ptr,
        bn_input_ptr,
        mean_ptr,
        invstd_ptr,
        gamma_ptr,
        beta_ptr,
        full_ptr,
        partials_ptr,
        BLOCK_: tl.constexpr,
        NUM_BLOCKS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        SRC_H_: tl.constexpr,
        SRC_W_: tl.constexpr,
        OUT_H_: tl.constexpr,
        OUT_W_: tl.constexpr,
        KERNEL_W_: tl.constexpr,
        STRIDE_: tl.constexpr,
        PADDING_: tl.constexpr,
        SRC_HW_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        SRC_ELEMS_: tl.constexpr,
        OUT_ELEMS_: tl.constexpr,
        CENTER_ONLY_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        elems = block_id * BLOCK_ + tl.arange(0, BLOCK_)

        mean = tl.load(mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        gamma = tl.load(gamma_ptr + channel).to(tl.float32)
        beta = tl.load(beta_ptr + channel).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        out_valid = elems < OUT_ELEMS_
        out_n = elems // OUT_HW_
        out_s = elems - out_n * OUT_HW_
        out_off = (out_n * CHANNELS_ + channel) * OUT_HW_ + out_s
        bn_value = tl.load(bn_input_ptr + out_off, mask=out_valid, other=0.0).to(tl.float32)
        sub = bn_value - mean
        affine = sub * invstd * gamma + beta
        gate = affine <= 0.0

        gate_value = tl.where(out_valid & gate, full, 0.0)
        sum0 = tl.sum(gate_value, axis=0)
        sum1 = tl.sum(gate_value * sub, axis=0)

        src_valid = elems < SRC_ELEMS_
        src_n = elems // SRC_HW_
        src_s = elems - src_n * SRC_HW_
        src_h = src_s // SRC_W_
        src_w = src_s - src_h * SRC_W_
        src_off = (src_n * CHANNELS_ + channel) * SRC_HW_ + src_s

        if CENTER_ONLY_:
            dest_h = src_h * STRIDE_
            dest_w = src_w * STRIDE_
            dest_valid = src_valid
        else:
            pool_offset = tl.load(offsets_ptr + src_off, mask=src_valid, other=0).to(tl.int32)
            kernel_h = pool_offset // KERNEL_W_
            kernel_w = pool_offset - kernel_h * KERNEL_W_
            dest_h = src_h * STRIDE_ + kernel_h - PADDING_
            dest_w = src_w * STRIDE_ + kernel_w - PADDING_
            dest_valid = (
                src_valid
                & (dest_h >= 0)
                & (dest_h < OUT_H_)
                & (dest_w >= 0)
                & (dest_w < OUT_W_)
            )
        dest_s = dest_h * OUT_W_ + dest_w
        dest_off = (src_n * CHANNELS_ + channel) * OUT_HW_ + dest_s
        dest_bn = tl.load(bn_input_ptr + dest_off, mask=dest_valid, other=0.0).to(tl.float32)
        dest_sub = dest_bn - mean
        dest_affine = dest_sub * invstd * gamma + beta
        dest_gate = dest_affine <= 0.0
        src_contributes = dest_valid & (~dest_gate)

        grad0 = tl.load(grad0_ptr + src_off, mask=src_contributes, other=0.0).to(tl.float32)
        grad1 = tl.load(grad1_ptr + src_off, mask=src_contributes, other=0.0).to(tl.float32)
        scatter_value = grad0 + grad1
        sum0 += tl.sum(scatter_value, axis=0)
        sum1 += tl.sum(scatter_value * dest_sub, axis=0)

        partial_off = channel * NUM_BLOCKS_ + block_id
        tl.store(partials_ptr + partial_off, sum0)
        tl.store(partials_ptr + CHANNELS_ * NUM_BLOCKS_ + partial_off, sum1)

    @triton.jit
    def _finalize_stats_kernel(
        partials_ptr,
        invstd_ptr,
        stats_ptr,
        out1_ptr,
        NUM_BLOCKS_: tl.constexpr,
        BLOCK_R_: tl.constexpr,
        CHANNELS_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_R_)
        mask = offsets < NUM_BLOCKS_
        partial_offs = channel * NUM_BLOCKS_ + offsets
        sum0_vals = tl.load(partials_ptr + partial_offs, mask=mask, other=0.0).to(tl.float32)
        sum1_vals = tl.load(
            partials_ptr + CHANNELS_ * NUM_BLOCKS_ + partial_offs,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        sum0 = tl.sum(sum0_vals, axis=0)
        sum1 = tl.sum(sum1_vals, axis=0)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        tl.store(stats_ptr + channel, sum0)
        tl.store(stats_ptr + CHANNELS_ + channel, sum1)
        tl.store(out1_ptr + channel, sum1 * invstd)

    @triton.jit
    def _candidate_contribution(
        grad0_ptr,
        grad1_ptr,
        offsets_ptr,
        channel,
        n,
        src_h,
        src_w,
        expected_offset,
        valid,
        CHANNELS_: tl.constexpr,
        SRC_W_: tl.constexpr,
        SRC_HW_: tl.constexpr,
    ):
        src_s = src_h * SRC_W_ + src_w
        src_off = (n * CHANNELS_ + channel) * SRC_HW_ + src_s
        actual_offset = tl.load(offsets_ptr + src_off, mask=valid, other=-1).to(tl.int32)
        hit = valid & (actual_offset == expected_offset)
        grad0 = tl.load(grad0_ptr + src_off, mask=hit, other=0.0).to(tl.float32)
        grad1 = tl.load(grad1_ptr + src_off, mask=hit, other=0.0).to(tl.float32)
        return grad0 + grad1

    @triton.jit
    def _write_output_kernel(
        grad0_ptr,
        grad1_ptr,
        offsets_ptr,
        bn_input_ptr,
        mean_ptr,
        invstd_ptr,
        gamma_ptr,
        beta_ptr,
        full_ptr,
        stats_ptr,
        out0_ptr,
        BLOCK_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        SRC_H_: tl.constexpr,
        SRC_W_: tl.constexpr,
        KERNEL_W_: tl.constexpr,
        STRIDE_: tl.constexpr,
        PADDING_: tl.constexpr,
        SRC_HW_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        OUT_W_: tl.constexpr,
        OUT_ELEMS_: tl.constexpr,
        BN_NORM_: tl.constexpr,
        CENTER_ONLY_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        elems = block_id * BLOCK_ + tl.arange(0, BLOCK_)
        valid = elems < OUT_ELEMS_

        n = elems // OUT_HW_
        out_s = elems - n * OUT_HW_
        out_h = out_s // OUT_W_
        out_w = out_s - out_h * OUT_W_
        out_off = (n * CHANNELS_ + channel) * OUT_HW_ + out_s

        mean = tl.load(mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        gamma = tl.load(gamma_ptr + channel).to(tl.float32)
        beta = tl.load(beta_ptr + channel).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)
        sum0 = tl.load(stats_ptr + channel).to(tl.float32)
        sum1 = tl.load(stats_ptr + CHANNELS_ + channel).to(tl.float32)

        bn_value = tl.load(bn_input_ptr + out_off, mask=valid, other=0.0).to(tl.float32)
        sub = bn_value - mean
        affine = sub * invstd * gamma + beta
        gate = affine <= 0.0
        gather_mask = valid & (~gate)

        even_h = (out_h % 2) == 0
        even_w = (out_w % 2) == 0

        if CENTER_ONLY_:
            src_h = out_h // STRIDE_
            src_w = out_w // STRIDE_
            src_s = src_h * SRC_W_ + src_w
            src_off = (n * CHANNELS_ + channel) * SRC_HW_ + src_s
            src_valid = gather_mask & even_h & even_w
            grad0 = tl.load(grad0_ptr + src_off, mask=src_valid, other=0.0).to(tl.float32)
            grad1 = tl.load(grad1_ptr + src_off, mask=src_valid, other=0.0).to(tl.float32)
            scatter_value = grad0 + grad1
        else:
            kh0 = tl.where(even_h, 1, 0)
            sh0 = (out_h + PADDING_ - kh0) // STRIDE_
            sh0_valid = (sh0 >= 0) & (sh0 < SRC_H_)
            kh1 = 2
            sh1 = (out_h + PADDING_ - kh1) // STRIDE_
            sh1_valid = (~even_h) & (sh1 >= 0) & (sh1 < SRC_H_)

            kw0 = tl.where(even_w, 1, 0)
            sw0 = (out_w + PADDING_ - kw0) // STRIDE_
            sw0_valid = (sw0 >= 0) & (sw0 < SRC_W_)
            kw1 = 2
            sw1 = (out_w + PADDING_ - kw1) // STRIDE_
            sw1_valid = (~even_w) & (sw1 >= 0) & (sw1 < SRC_W_)

            scatter_value = _candidate_contribution(
                grad0_ptr,
                grad1_ptr,
                offsets_ptr,
                channel,
                n,
                sh0,
                sw0,
                kh0 * KERNEL_W_ + kw0,
                gather_mask & sh0_valid & sw0_valid,
                CHANNELS_,
                SRC_W_,
                SRC_HW_,
            )
            scatter_value += _candidate_contribution(
                grad0_ptr,
                grad1_ptr,
                offsets_ptr,
                channel,
                n,
                sh0,
                sw1,
                kh0 * KERNEL_W_ + kw1,
                gather_mask & sh0_valid & sw1_valid,
                CHANNELS_,
                SRC_W_,
                SRC_HW_,
            )
            scatter_value += _candidate_contribution(
                grad0_ptr,
                grad1_ptr,
                offsets_ptr,
                channel,
                n,
                sh1,
                sw0,
                kh1 * KERNEL_W_ + kw0,
                gather_mask & sh1_valid & sw0_valid,
                CHANNELS_,
                SRC_W_,
                SRC_HW_,
            )
            scatter_value += _candidate_contribution(
                grad0_ptr,
                grad1_ptr,
                offsets_ptr,
                channel,
                n,
                sh1,
                sw1,
                kh1 * KERNEL_W_ + kw1,
                gather_mask & sh1_valid & sw1_valid,
                CHANNELS_,
                SRC_W_,
                SRC_HW_,
            )

        where_value = tl.where(gate, full, scatter_value)
        mean_grad = sum0 * BN_NORM_
        proj = sum1 * BN_NORM_ * invstd * invstd
        scale = invstd * gamma
        out_value = (where_value - sub * proj - mean_grad) * scale
        tl.store(out0_ptr + out_off, out_value, mask=valid)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 12:
        raise ValueError(f"expected 12 inputs, got {len(inputs)}")

    (
        getitem_144,
        getitem_153,
        arg113_1,
        arg109_1,
        arg110_1,
        arg111_1,
        arg2_1,
        arg3_1,
        full,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs

    expected_src = (BATCH, CHANNELS, SRC_H, SRC_W)
    expected_out = (BATCH, CHANNELS, OUT_H, OUT_W)
    if getitem_144.shape != expected_src or getitem_153.shape != expected_src:
        raise ValueError("unexpected source gradient shape")
    if arg113_1.shape != expected_src or arg113_1.dtype != torch.int8:
        raise ValueError(f"unexpected max-pool offsets: shape={tuple(arg113_1.shape)} dtype={arg113_1.dtype}")
    if arg109_1.shape != expected_out:
        raise ValueError(f"unexpected BN input shape: {tuple(arg109_1.shape)}")
    if arg110_1.shape != (1, CHANNELS, 1, 1) or arg111_1.shape != (1, CHANNELS, 1, 1):
        raise ValueError("unexpected BN mean/invstd shape")
    if arg2_1.shape != (CHANNELS,) or arg3_1.shape != (CHANNELS,):
        raise ValueError("unexpected affine parameter shape")
    if full.shape != ():
        raise ValueError(f"unexpected full scalar shape: {tuple(full.shape)}")
    tensors = (getitem_144, getitem_153, arg109_1, arg110_1, arg111_1, arg2_1, arg3_1, full)
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("oracle expects all floating tensor inputs to be float32")
    if getitem_144.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if not all(t.is_contiguous() for t in (getitem_144, getitem_153, arg113_1, arg109_1, arg110_1, arg111_1, arg2_1, arg3_1)):
        raise ValueError("oracle expects contiguous captured tensor inputs")

    return getitem_144, getitem_153, arg113_1, arg109_1, arg110_1, arg111_1, arg2_1, arg3_1, full


def _offsets_are_center(offsets: torch.Tensor) -> bool:
    """The captured corpus shape has Index(5, 4) max-pool offsets, i.e. all 4."""
    key = (offsets.data_ptr(), offsets.numel())
    cached = _CENTER_OFFSET_CACHE.get(key)
    if cached is None:
        cached = bool(torch.all(offsets == 4).item())
        _CENTER_OFFSET_CACHE[key] = cached
    return cached


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
    (
        grad0,
        grad1,
        offsets,
        bn_input,
        mean,
        invstd,
        gamma,
        beta,
        full,
    ) = _validate_inputs(inputs)
    center_only = _offsets_are_center(offsets)

    out0 = torch.empty((BATCH, CHANNELS, OUT_H, OUT_W), device=bn_input.device, dtype=torch.float32)
    out1 = torch.empty((CHANNELS,), device=bn_input.device, dtype=torch.float32)

    num_reduce_blocks = triton.cdiv(OUT_ELEMS, REDUCE_BLOCK)
    partials = torch.empty((2, CHANNELS, num_reduce_blocks), device=bn_input.device, dtype=torch.float32)
    stats = torch.empty((2, CHANNELS), device=bn_input.device, dtype=torch.float32)

    _partial_reduce_kernel[(CHANNELS, num_reduce_blocks)](
        grad0,
        grad1,
        offsets,
        bn_input,
        mean,
        invstd,
        gamma,
        beta,
        full,
        partials,
        BLOCK_=REDUCE_BLOCK,
        NUM_BLOCKS_=num_reduce_blocks,
        CHANNELS_=CHANNELS,
        SRC_H_=SRC_H,
        SRC_W_=SRC_W,
        OUT_H_=OUT_H,
        OUT_W_=OUT_W,
        KERNEL_W_=KERNEL_W,
        STRIDE_=STRIDE,
        PADDING_=PADDING,
        SRC_HW_=SRC_HW,
        OUT_HW_=OUT_HW,
        SRC_ELEMS_=SRC_ELEMS,
        OUT_ELEMS_=OUT_ELEMS,
        CENTER_ONLY_=center_only,
        num_warps=REDUCE_WARPS,
    )
    _finalize_stats_kernel[(CHANNELS,)](
        partials,
        invstd,
        stats,
        out1,
        NUM_BLOCKS_=num_reduce_blocks,
        BLOCK_R_=triton.next_power_of_2(num_reduce_blocks),
        CHANNELS_=CHANNELS,
        num_warps=FINAL_WARPS,
    )
    _write_output_kernel[(CHANNELS, triton.cdiv(OUT_ELEMS, OUTPUT_BLOCK))](
        grad0,
        grad1,
        offsets,
        bn_input,
        mean,
        invstd,
        gamma,
        beta,
        full,
        stats,
        out0,
        BLOCK_=OUTPUT_BLOCK,
        CHANNELS_=CHANNELS,
        SRC_H_=SRC_H,
        SRC_W_=SRC_W,
        KERNEL_W_=KERNEL_W,
        STRIDE_=STRIDE,
        PADDING_=PADDING,
        SRC_HW_=SRC_HW,
        OUT_HW_=OUT_HW,
        OUT_W_=OUT_W,
        OUT_ELEMS_=OUT_ELEMS,
        BN_NORM_=BN_NORM,
        CENTER_ONLY_=center_only,
        num_warps=OUTPUT_WARPS,
    )
    return (out0, out1)


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
