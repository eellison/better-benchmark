"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full DenseNet inference cat-BN-ReLU scope by keeping the channel concat virtual, hoisting the per-channel BN inverse standard deviation, and writing the final fp16 ReLU output directly from the five source tensors without materializing the [64,640,7,7] cat or intermediate affine tensors, whereas the best Inductor configuration is already slightly faster than this full-scope fused Triton implementation on the measured shape; Inductor cannot get a confirmed local win here because the remaining cost is dominated by required source reads, parameter reads, output stores, and launch overhead, and the oracle does not establish a faster floor than compile; the fix is BANDWIDTH_BOUND: mark this repro at floor for now and revisit only if a broader concat-pointwise fusion template beats the current compiled kernel on this exact full scope."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 1024}, num_warps=8, num_stages=3),
        ],
        key=["C_TOTAL"],
    )
    @triton.jit
    def _bn_invstd_kernel(
        var_ptr,
        invstd_ptr,
        C_TOTAL: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = offsets < C_TOTAL
        var = tl.load(var_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        invstd = 1.0 / tl.sqrt(var + 1.0e-5)
        tl.store(invstd_ptr + offsets, invstd, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL", "C0", "C_TOTAL", "HW", "BRANCH_C"],
    )
    @triton.jit
    def oracle_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
        x4_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        C0: tl.constexpr,
        C_TOTAL: tl.constexpr,
        HW: tl.constexpr,
        BRANCH_C: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        safe_offsets = tl.where(mask, offsets, 0)

        spatial = safe_offsets % HW
        channel = (safe_offsets // HW) % C_TOTAL
        batch = safe_offsets // (C_TOTAL * HW)

        rel = channel - C0
        in0 = channel < C0
        in1 = (rel >= 0) & (rel < BRANCH_C)
        in2 = (rel >= BRANCH_C) & (rel < 2 * BRANCH_C)
        in3 = (rel >= 2 * BRANCH_C) & (rel < 3 * BRANCH_C)
        in4 = (rel >= 3 * BRANCH_C) & (rel < 4 * BRANCH_C)

        c0 = tl.where(in0, channel, 0)
        c1 = tl.where(in1, rel, 0)
        c2 = tl.where(in2, rel - BRANCH_C, 0)
        c3 = tl.where(in3, rel - 2 * BRANCH_C, 0)
        c4 = tl.where(in4, rel - 3 * BRANCH_C, 0)

        x0_offsets = (batch * C0 + c0) * HW + spatial
        xb_offsets = (batch * BRANCH_C) * HW + spatial
        x1_offsets = xb_offsets + c1 * HW
        x2_offsets = xb_offsets + c2 * HW
        x3_offsets = xb_offsets + c3 * HW
        x4_offsets = xb_offsets + c4 * HW

        x = tl.load(x0_ptr + x0_offsets, mask=mask & in0, other=0.0).to(tl.float32)
        x += tl.load(x1_ptr + x1_offsets, mask=mask & in1, other=0.0).to(tl.float32)
        x += tl.load(x2_ptr + x2_offsets, mask=mask & in2, other=0.0).to(tl.float32)
        x += tl.load(x3_ptr + x3_offsets, mask=mask & in3, other=0.0).to(tl.float32)
        x += tl.load(x4_ptr + x4_offsets, mask=mask & in4, other=0.0).to(tl.float32)

        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * invstd
        y = y * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        tl.store(out_ptr + offsets, y, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([64, 512, 7, 7], f16), T([64, 32, 7, 7], f16), T([64, 32, 7, 7], f16), T([64, 32, 7, 7], f16), T([64, 32, 7, 7], f16), T([640], f16), T([640], f16), T([640], f16), T([640], f16))")
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
    if len(inputs) != 9:
        raise ValueError(f"expected 9 inputs, got {len(inputs)}")

    (
        avg_pool2d_2,
        convolution_89,
        convolution_91,
        convolution_93,
        convolution_95,
        arg481_1,
        arg482_1,
        arg483_1,
        arg484_1,
    ) = inputs

    if not avg_pool2d_2.is_cuda:
        cat = torch.cat(
            [avg_pool2d_2, convolution_89, convolution_91, convolution_93, convolution_95],
            dim=1,
        )
        invstd = torch.reciprocal(torch.sqrt(arg482_1.float() + 1.0e-5))
        y = (cat - arg481_1.float()[None, :, None, None]) * invstd[None, :, None, None]
        y = y * arg483_1[None, :, None, None] + arg484_1[None, :, None, None]
        return torch.relu(y.to(torch.float16))

    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    tensor_inputs = inputs
    if any(not tensor.is_contiguous() for tensor in tensor_inputs):
        raise ValueError("oracle_cat_bn_relu.py expects contiguous NCHW inputs and contiguous parameters")
    if any(tensor.dtype != torch.float16 for tensor in tensor_inputs):
        raise ValueError("oracle_cat_bn_relu.py expects all inputs to be fp16 tensors")

    n_batches, c0, height, width = avg_pool2d_2.shape
    branch_c = convolution_89.shape[1]
    hw = height * width
    c_total = c0 + 4 * branch_c
    if arg481_1.numel() != c_total or arg482_1.numel() != c_total:
        raise ValueError("BN mean/variance parameter length does not match virtual cat channels")
    if arg483_1.numel() != c_total or arg484_1.numel() != c_total:
        raise ValueError("BN affine parameter length does not match virtual cat channels")
    for branch in (convolution_91, convolution_93, convolution_95):
        if branch.shape != convolution_89.shape:
            raise ValueError("all four convolution branch inputs must have the same shape")
    if avg_pool2d_2.shape[0] != convolution_89.shape[0] or avg_pool2d_2.shape[2:] != convolution_89.shape[2:]:
        raise ValueError("all virtual cat inputs must share batch and spatial dimensions")

    output = torch.empty((n_batches, c_total, height, width), device=avg_pool2d_2.device, dtype=torch.float16)
    invstd = torch.empty((c_total,), device=avg_pool2d_2.device, dtype=torch.float32)
    total = output.numel()

    invstd_grid = lambda meta: (triton.cdiv(c_total, meta["BLOCK_C"]),)
    _bn_invstd_kernel[invstd_grid](
        arg482_1,
        invstd,
        C_TOTAL=c_total,
    )
    pointwise_grid = lambda meta: (triton.cdiv(total, meta["BLOCK_N"]),)
    oracle_kernel[pointwise_grid](
        avg_pool2d_2,
        convolution_89,
        convolution_91,
        convolution_93,
        convolution_95,
        arg481_1,
        invstd,
        arg483_1,
        arg484_1,
        output,
        TOTAL=total,
        C0=c0,
        C_TOTAL=c_total,
        HW=hw,
        BRANCH_C=branch_c,
    )
    return output


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
