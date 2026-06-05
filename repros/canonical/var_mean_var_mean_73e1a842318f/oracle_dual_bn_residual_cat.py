"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet dual training-batchnorm block, including both per-channel var/mean reductions, running-stat copy_ side effects, first BN affine contribution folded through the channel-cat half, second BN affine contribution, residual add, and metadata-equivalent stat outputs, whereas Inductor schedules the two norm-template reductions and the downstream cat/add epilogue as separate generic regions; Inductor cannot do this today because the scheduler does not canonicalize sibling BN-training reductions feeding a channel cat plus residual add into one multi-output norm epilogue over the final layout; the fix is SCHEDULER_FUSION: add a guarded dual-BN training template that batches stats finalization and fuses the cat/residual output writer while preserving running-stat updates and side-output views."""
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


N = 512
C1 = 80
C2 = 160
H = 7
W = 7
HW = H * W
ELEMS_PER_CHANNEL = N * HW
TOTAL2 = N * C2 * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361
STAT_BLOCK = 32768
OUTPUT_BLOCK = 256


def _validate_inputs(inputs):
    if len(inputs) != 11:
        raise ValueError(f"expected 11 inputs, got {len(inputs)}")
    tensors = tuple(inputs)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("all repro inputs must be tensors")
    expected_shapes = (
        (N, C1, H, W),
        (C1,),
        (C1,),
        (C1,),
        (C1,),
        (N, C1, H, W),
        (N, C2, H, W),
        (C2,),
        (C2,),
        (C2,),
        (C2,),
    )
    for idx, (tensor, shape) in enumerate(zip(tensors, expected_shapes)):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"input {idx} shape {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"input {idx} dtype {tensor.dtype} != torch.float32")
        if not tensor.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not tensor.is_contiguous():
            raise ValueError(f"input {idx} must be contiguous")
    return tensors

if triton is not None:

    @triton.jit
    def _dual_bn_stats_update_kernel(
        x1_ptr,
        x2_ptr,
        running_mean1_ptr,
        running_var1_ptr,
        running_mean2_ptr,
        running_var2_ptr,
        mean1_ptr,
        invstd1_ptr,
        mean2_ptr,
        invstd2_ptr,
        c1: tl.constexpr,
        c2: tl.constexpr,
        hw_size: tl.constexpr,
        elems_per_channel: tl.constexpr,
        correction: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        block_size: tl.constexpr,
    ):
        stat_id = tl.program_id(0)
        first = stat_id < c1
        channel1 = tl.where(first, stat_id, 0)
        channel2 = tl.where(first, 0, stat_id - c1)
        offsets = tl.arange(0, block_size)
        mask = offsets < elems_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x1_offsets = n_idx * c1 * hw_size + channel1 * hw_size + hw_idx
        x2_offsets = n_idx * c2 * hw_size + channel2 * hw_size + hw_idx
        values1 = tl.load(x1_ptr + x1_offsets, mask=mask & first, other=0.0).to(tl.float32)
        values2 = tl.load(x2_ptr + x2_offsets, mask=mask & ~first, other=0.0).to(tl.float32)
        values = values1 + values2

        total_sum = tl.sum(values, axis=0)
        total_sum_sq = tl.sum(values * values, axis=0)
        mean = total_sum / elems_per_channel
        var = total_sum_sq / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean1 = tl.load(running_mean1_ptr + channel1, mask=first, other=0.0)
        old_var1 = tl.load(running_var1_ptr + channel1, mask=first, other=0.0)
        old_mean2 = tl.load(running_mean2_ptr + channel2, mask=~first, other=0.0)
        old_var2 = tl.load(running_var2_ptr + channel2, mask=~first, other=0.0)
        new_mean1 = old_mean1 * (1.0 - momentum) + mean * momentum
        new_var1 = old_var1 * (1.0 - momentum) + var * correction * momentum
        new_mean2 = old_mean2 * (1.0 - momentum) + mean * momentum
        new_var2 = old_var2 * (1.0 - momentum) + var * correction * momentum

        tl.store(running_mean1_ptr + channel1, new_mean1, mask=first)
        tl.store(running_var1_ptr + channel1, new_var1, mask=first)
        tl.store(mean1_ptr + channel1, mean, mask=first)
        tl.store(invstd1_ptr + channel1, invstd, mask=first)
        tl.store(running_mean2_ptr + channel2, new_mean2, mask=~first)
        tl.store(
            running_var2_ptr + channel2,
            new_var2,
            mask=~first,
        )
        tl.store(mean2_ptr + channel2, mean, mask=~first)
        tl.store(invstd2_ptr + channel2, invstd, mask=~first)

    @triton.jit
    def _final_residual_cat_kernel(
        x1_ptr,
        add1_ptr,
        x2_ptr,
        weight1_ptr,
        bias1_ptr,
        weight2_ptr,
        bias2_ptr,
        mean1_ptr,
        invstd1_ptr,
        mean2_ptr,
        invstd2_ptr,
        out_ptr,
        total: tl.constexpr,
        c1: tl.constexpr,
        c2: tl.constexpr,
        hw_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total

        hw_idx = offsets % hw_size
        tmp = offsets // hw_size
        channel2 = tmp % c2
        n_idx = tmp // c2

        x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean2 = tl.load(mean2_ptr + channel2, mask=mask, other=0.0)
        invstd2 = tl.load(invstd2_ptr + channel2, mask=mask, other=0.0)
        weight2 = tl.load(weight2_ptr + channel2, mask=mask, other=0.0)
        bias2 = tl.load(bias2_ptr + channel2, mask=mask, other=0.0)
        y2 = (x2 - mean2) * invstd2 * weight2 + bias2

        first_half = channel2 < c1
        channel1 = tl.where(first_half, channel2, channel2 - c1)
        offset1 = n_idx * c1 * hw_size + channel1 * hw_size + hw_idx
        mask1 = mask & first_half
        mask_bn1 = mask & ~first_half

        add1 = tl.load(add1_ptr + offset1, mask=mask1, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offset1, mask=mask_bn1, other=0.0).to(tl.float32)
        mean1 = tl.load(mean1_ptr + channel1, mask=mask_bn1, other=0.0)
        invstd1 = tl.load(invstd1_ptr + channel1, mask=mask_bn1, other=0.0)
        weight1 = tl.load(weight1_ptr + channel1, mask=mask_bn1, other=0.0)
        bias1 = tl.load(bias1_ptr + channel1, mask=mask_bn1, other=0.0)
        y1 = (x1 - mean1) * invstd1 * weight1 + bias1
        cat_value = tl.where(first_half, add1, y1)

        tl.store(out_ptr + offsets, cat_value + y2, mask=mask)


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
        convolution_70,
        arg383_1,
        arg384_1,
        arg385_1,
        arg386_1,
        add_315,
        convolution_72,
        arg395_1,
        arg396_1,
        arg397_1,
        arg398_1,
    ) = _validate_inputs(inputs)

    mean1 = torch.empty((1, C1, 1, 1), device=convolution_70.device, dtype=torch.float32)
    invstd1 = torch.empty((C1,), device=convolution_70.device, dtype=torch.float32)
    mean2 = torch.empty((1, C2, 1, 1), device=convolution_72.device, dtype=torch.float32)
    invstd2 = torch.empty((C2,), device=convolution_72.device, dtype=torch.float32)
    out = torch.empty_like(convolution_72)

    _dual_bn_stats_update_kernel[(C1 + C2,)](
        convolution_70,
        convolution_72,
        arg383_1,
        arg384_1,
        arg395_1,
        arg396_1,
        mean1,
        invstd1,
        mean2,
        invstd2,
        c1=C1,
        c2=C2,
        hw_size=HW,
        elems_per_channel=ELEMS_PER_CHANNEL,
        correction=RUNNING_VAR_CORRECTION,
        eps=EPS,
        momentum=MOMENTUM,
        block_size=STAT_BLOCK,
        num_warps=8,
    )
    _final_residual_cat_kernel[(triton.cdiv(TOTAL2, OUTPUT_BLOCK),)](
        convolution_70,
        add_315,
        convolution_72,
        arg385_1,
        arg386_1,
        arg397_1,
        arg398_1,
        mean1,
        invstd1,
        mean2,
        invstd2,
        out,
        total=TOTAL2,
        c1=C1,
        c2=C2,
        hw_size=HW,
        block_size=OUTPUT_BLOCK,
        num_warps=4,
    )

    return (
        invstd1,
        invstd2,
        out,
        mean2,
        mean1,
        arg383_1,
        arg384_1,
        arg395_1,
        arg396_1,
    )


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
