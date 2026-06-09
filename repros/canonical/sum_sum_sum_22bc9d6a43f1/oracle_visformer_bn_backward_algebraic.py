"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Visformer double-normalization backward scope by keeping the first `sum(x)` and `sum(x * normalized)` channel reductions as reusable summaries, materializing the required `[1, C, H, W]` batch-sum side output, and deriving the final dependent channel reductions from compact block summaries, whereas Inductor schedules the decomposed sibling and dependent reductions as generic producer/consumer regions over large intermediates; Inductor cannot do this today because its algebraic simplifier and reduction scheduler do not recognize this BN-backward-style chain as a reusable-summary pattern once a materialized side output is also returned; the fix is ALGEBRAIC_ELIMINATION: add a guarded multi-output reduction rewrite that preserves the shared channel summaries, lowers the batch-sum side output directly, and finalizes the dependent channel reductions without rematerializing the full intermediate tensor."""
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


N = 128
C = 192
H = 28
W = 28
HW = H * W
REDUCE_SIZE = N * HW
INV_REDUCE_SIZE = 9.964923469387754e-06
BATCH_BLOCK_HW = 64
NUM_HW_BLOCKS = (HW + BATCH_BLOCK_HW - 1) // BATCH_BLOCK_HW


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _spatial_summary_kernel(
        x_ptr,
        arg96_ptr,
        mean1_ptr,
        invstd1_ptr,
        weight1_ptr,
        bias1_ptr,
        spatial_bias_ptr,
        mean2_ptr,
        residual_ptr,
        spatial_summaries_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        c = tl.program_id(0)
        hw_block = tl.program_id(1)
        n = tl.arange(0, BLOCK_N)
        hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw < HW_
        offsets = n[:, None] * C_ * HW_ + c * HW_ + hw[None, :]
        mask = hw_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg96 = tl.load(arg96_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean1 = tl.load(mean1_ptr + c).to(tl.float32)
        invstd1 = tl.load(invstd1_ptr + c).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c).to(tl.float32)
        spatial_bias = tl.load(spatial_bias_ptr + c * HW_ + hw, mask=hw_mask, other=0.0).to(tl.float32)
        mean2 = tl.load(mean2_ptr + c).to(tl.float32)

        sub4 = arg96 - mean1
        normalized = sub4 * invstd1 * weight1 + bias1 + spatial_bias[None, :] - mean2
        sum_x_hw = tl.sum(x, axis=0)
        sum_x_norm_hw = tl.sum(x * normalized, axis=0)
        sum_norm_hw = tl.sum(normalized, axis=0)
        sum_residual_hw = tl.sum(residual, axis=0)
        sum_sub4_hw = tl.sum(sub4, axis=0)
        sum_norm_sub4_hw = tl.sum(normalized * sub4, axis=0)
        sum_x_sub4_hw = tl.sum(x * sub4, axis=0)
        sum_residual_sub4_hw = tl.sum(residual * sub4, axis=0)

        base = c * HW_ + hw
        plane_stride = C_ * HW_
        tl.store(spatial_summaries_ptr + base, sum_x_hw, mask=hw_mask)
        tl.store(spatial_summaries_ptr + plane_stride + base, sum_x_norm_hw, mask=hw_mask)
        tl.store(spatial_summaries_ptr + 2 * plane_stride + base, sum_norm_hw, mask=hw_mask)
        tl.store(spatial_summaries_ptr + 3 * plane_stride + base, sum_residual_hw, mask=hw_mask)
        tl.store(spatial_summaries_ptr + 4 * plane_stride + base, sum_sub4_hw, mask=hw_mask)
        tl.store(spatial_summaries_ptr + 5 * plane_stride + base, sum_norm_sub4_hw, mask=hw_mask)
        tl.store(spatial_summaries_ptr + 6 * plane_stride + base, sum_x_sub4_hw, mask=hw_mask)
        tl.store(spatial_summaries_ptr + 7 * plane_stride + base, sum_residual_sub4_hw, mask=hw_mask)

    @triton.jit
    def _finalize_from_spatial_summaries_kernel(
        spatial_summaries_ptr,
        gamma2_ptr,
        beta2_ptr,
        invstd1_ptr,
        weight1_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        INV_REDUCE_SIZE_: tl.constexpr,
        REDUCE_SIZE_: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        c = tl.program_id(0)
        hw = tl.arange(0, BLOCK_HW)
        mask = hw < HW_
        base = c * HW_ + hw
        plane_stride = C_ * HW_

        sum_x_hw = tl.load(spatial_summaries_ptr + base, mask=mask, other=0.0).to(tl.float32)
        sum_x_norm_hw = tl.load(spatial_summaries_ptr + plane_stride + base, mask=mask, other=0.0).to(tl.float32)
        sum_norm_hw = tl.load(spatial_summaries_ptr + 2 * plane_stride + base, mask=mask, other=0.0).to(tl.float32)
        sum_residual_hw = tl.load(spatial_summaries_ptr + 3 * plane_stride + base, mask=mask, other=0.0).to(tl.float32)
        sum_sub4_hw = tl.load(spatial_summaries_ptr + 4 * plane_stride + base, mask=mask, other=0.0).to(tl.float32)
        sum_norm_sub4_hw = tl.load(spatial_summaries_ptr + 5 * plane_stride + base, mask=mask, other=0.0).to(tl.float32)
        sum_x_sub4_hw = tl.load(spatial_summaries_ptr + 6 * plane_stride + base, mask=mask, other=0.0).to(tl.float32)
        sum_residual_sub4_hw = tl.load(spatial_summaries_ptr + 7 * plane_stride + base, mask=mask, other=0.0).to(tl.float32)

        sum_x = tl.sum(sum_x_hw, axis=0)
        sum_x_norm = tl.sum(sum_x_norm_hw, axis=0)
        sum_sub4 = tl.sum(sum_sub4_hw, axis=0)
        sum_norm_sub4 = tl.sum(sum_norm_sub4_hw, axis=0)
        sum_x_sub4 = tl.sum(sum_x_sub4_hw, axis=0)
        sum_residual_sub4 = tl.sum(sum_residual_sub4_hw, axis=0)

        gamma2 = tl.load(gamma2_ptr + c).to(tl.float32)
        beta2 = tl.load(beta2_ptr + c).to(tl.float32)
        invstd1 = tl.load(invstd1_ptr + c).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c).to(tl.float32)
        mean_x = sum_x * INV_REDUCE_SIZE_
        norm_scale = sum_x_norm * INV_REDUCE_SIZE_ * gamma2 * gamma2
        grad_scale = gamma2 * beta2

        out1 = sum_residual_hw + grad_scale * sum_x_hw - grad_scale * norm_scale * sum_norm_hw - grad_scale * mean_x * 128.0
        tl.store(out1_ptr + base, out1, mask=mask)

        sum_grad_sub4 = (
            sum_residual_sub4
            + grad_scale * sum_x_sub4
            - grad_scale * norm_scale * sum_norm_sub4
            - grad_scale * mean_x * sum_sub4
        )
        correction = sum_grad_sub4 * INV_REDUCE_SIZE_ * invstd1 * invstd1
        out3 = -correction * sum_sub4 * invstd1 * weight1

        tl.store(out0_ptr + c, sum_x_norm * gamma2)
        tl.store(out2_ptr + c, sum_grad_sub4 * invstd1)
        tl.store(out3_ptr + c, out3)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 11:
        raise ValueError(f"expected 11 inputs, got {len(inputs)}")
    tensors = tuple(inputs)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("all repro inputs must be tensors")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("all repro inputs must be torch.float32")
    if any(x.device.type != "cuda" for x in tensors):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")

    (
        x,
        arg96,
        mean1,
        invstd1,
        weight1,
        bias1,
        spatial_bias,
        mean2,
        gamma2,
        beta2,
        residual,
    ) = tensors

    expected_4d = (N, C, H, W)
    if tuple(x.shape) != expected_4d or tuple(arg96.shape) != expected_4d:
        raise ValueError("unexpected activation shape")
    if tuple(residual.shape) != expected_4d:
        raise ValueError("unexpected residual shape")
    if tuple(mean1.shape) != (1, C, 1, 1) or tuple(invstd1.shape) != (1, C, 1, 1):
        raise ValueError("unexpected first normalization parameter shape")
    if tuple(mean2.shape) != (1, C, 1, 1):
        raise ValueError("unexpected second normalization mean shape")
    if tuple(spatial_bias.shape) != (1, C, H, W):
        raise ValueError("unexpected spatial bias shape")
    for name, tensor in (("weight1", weight1), ("bias1", bias1), ("gamma2", gamma2), ("beta2", beta2)):
        if tuple(tensor.shape) != (C,):
            raise ValueError(f"unexpected {name} shape: {tuple(tensor.shape)}")
    if not all(x.is_contiguous() for x in tensors):
        raise ValueError("oracle expects contiguous inputs for this canonical shape")
    return tensors


@oracle_impl(hardware="H100", shapes="(T([128, 192, 28, 28], f32), T([128, 192, 28, 28], f32), T([1, 192, 1, 1], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([1, 192, 28, 28], f32), T([1, 192, 1, 1], f32), T([192], f32), T([192], f32), T([128, 192, 28, 28], f32))")
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
        x,
        arg96,
        mean1,
        invstd1,
        weight1,
        bias1,
        spatial_bias,
        mean2,
        gamma2,
        beta2,
        residual,
    ) = _validate_inputs(tuple(inputs))

    spatial_summaries = torch.empty((8, C, HW), device=x.device, dtype=torch.float32)
    out0 = torch.empty((C,), device=x.device, dtype=torch.float32)
    out1 = torch.empty((1, C, H, W), device=x.device, dtype=torch.float32)
    out2 = torch.empty((C,), device=x.device, dtype=torch.float32)
    out3 = torch.empty((C,), device=x.device, dtype=torch.float32)

    _spatial_summary_kernel[(C, NUM_HW_BLOCKS)](
        x,
        arg96,
        mean1,
        invstd1,
        weight1,
        bias1,
        spatial_bias,
        mean2,
        residual,
        spatial_summaries,
        C_=C,
        HW_=HW,
        BLOCK_N=N,
        BLOCK_HW=BATCH_BLOCK_HW,
        num_warps=4,
        num_stages=4,
    )
    _finalize_from_spatial_summaries_kernel[(C,)](
        spatial_summaries,
        gamma2,
        beta2,
        invstd1,
        weight1,
        out0,
        out1,
        out2,
        out3,
        C_=C,
        HW_=HW,
        INV_REDUCE_SIZE_=INV_REDUCE_SIZE,
        REDUCE_SIZE_=REDUCE_SIZE,
        BLOCK_HW=triton.next_power_of_2(HW),
        num_warps=4,
        num_stages=4,
    )
    return out0, out1, out2, out3


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
