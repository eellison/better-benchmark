"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet BN-affine/ReLU plus channel-cat scope with a paired layout kernel that copies `relu_35` into the first output half while computing the normalized `convolution_84` value for the matching source coordinate into the second output half, whereas Inductor lowers the cat as generic output-element layout/indexing work and cannot pair the two channel-half stores or hoist channel parameters at the row granularity; Inductor cannot do this today because its scheduler/codegen does not model fixed channel-cat materialization as a fused producer with ownership of both destination channel ranges; the fix is SCHEDULER_FUSION: teach concat/materialization scheduling to emit paired channel-segment stores with pointwise producers fused and per-channel parameters loaded once per row tile."""
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


EPS = 1.0e-5

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
        ],
        key=["N", "C", "H", "W"],
    )
    @triton.jit
    def _paired_cat_bn_relu_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        relu_ptr,
        out_ptr,
        N: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        conv_s0: tl.constexpr,
        conv_s1: tl.constexpr,
        conv_s2: tl.constexpr,
        conv_s3: tl.constexpr,
        relu_s0: tl.constexpr,
        relu_s1: tl.constexpr,
        relu_s2: tl.constexpr,
        relu_s3: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        out_s2: tl.constexpr,
        out_s3: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw_offsets = tl.program_id(1) * BLOCK_HW + tl.arange(0, BLOCK_HW)

        hw: tl.constexpr = H * W
        n_offsets = row_offsets // C
        c_offsets = row_offsets - n_offsets * C
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets - h_offsets * W

        valid_rows = row_offsets < (N * C)
        valid_hw = hw_offsets < hw
        valid = valid_rows[:, None] & valid_hw[None, :]

        relu_offsets = (
            n_offsets[:, None] * relu_s0
            + c_offsets[:, None] * relu_s1
            + h_offsets[None, :] * relu_s2
            + w_offsets[None, :] * relu_s3
        )
        conv_offsets = (
            n_offsets[:, None] * conv_s0
            + c_offsets[:, None] * conv_s1
            + h_offsets[None, :] * conv_s2
            + w_offsets[None, :] * conv_s3
        )
        out_base = (
            n_offsets[:, None] * out_s0
            + c_offsets[:, None] * out_s1
            + h_offsets[None, :] * out_s2
            + w_offsets[None, :] * out_s3
        )

        relu_values = tl.load(relu_ptr + relu_offsets, mask=valid, other=0.0)
        tl.store(out_ptr + out_base, relu_values, mask=valid)

        conv_values = tl.load(conv_ptr + conv_offsets, mask=valid, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var[:, None] + 1.0e-5)
        normalized = (conv_values - mean[:, None]) * invstd
        transformed = normalized * weight[:, None] + bias[:, None]
        transformed = tl.where(transformed != transformed, transformed, tl.maximum(transformed, 0.0))
        tl.store(out_ptr + out_base + C * out_s1, transformed, mask=valid)


def _validate_inputs(inputs):
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, conv, var, weight, bias, relu = inputs
    tensor_inputs = (mean, conv, var, weight, bias, relu)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensor_inputs):
        raise TypeError("all oracle inputs must be tensors")
    if any(tensor.dtype is not torch.float32 for tensor in tensor_inputs):
        raise TypeError("oracle_paired_cat_bn_relu.py expects all inputs to be torch.float32")
    if conv.ndim != 4 or relu.ndim != 4:
        raise ValueError("convolution_84 and relu_35 must be rank-4 tensors")
    if conv.shape != relu.shape:
        raise ValueError(f"convolution_84 and relu_35 shapes differ: {tuple(conv.shape)} vs {tuple(relu.shape)}")

    n, c, h, w = conv.shape
    expected_param_shape = (c,)
    for name, tensor in (
        ("arg386_1", mean),
        ("arg387_1", var),
        ("arg388_1", weight),
        ("arg389_1", bias),
    ):
        if tuple(tensor.shape) != expected_param_shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {expected_param_shape}")

    device = conv.device
    if any(tensor.device != device for tensor in tensor_inputs):
        raise ValueError("all oracle inputs must be on the same device")

    return mean, conv, var, weight, bias, relu, n, c, h, w


def _output_stride(conv: torch.Tensor, relu: torch.Tensor) -> tuple[int, int, int, int]:
    n, c, h, w = conv.shape
    if (
        conv.is_contiguous(memory_format=torch.channels_last)
        and relu.is_contiguous(memory_format=torch.channels_last)
    ):
        return (2 * c * h * w, 1, 2 * c * w, 2 * c)
    return (2 * c * h * w, h * w, w, 1)


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
    mean, conv, var, weight, bias, relu, n, c, h, w = _validate_inputs(inputs)

    if not conv.is_cuda:
        invstd = torch.reciprocal(torch.sqrt(var + EPS))
        normalized = (conv - mean[None, :, None, None]) * invstd[None, :, None, None]
        transformed = torch.relu(normalized * weight[None, :, None, None] + bias[None, :, None, None])
        return torch.cat([relu, transformed], dim=1)

    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    output = torch.empty_strided(
        (n, 2 * c, h, w),
        _output_stride(conv, relu),
        device=conv.device,
        dtype=torch.float32,
    )
    hw = h * w
    grid = lambda meta: (
        triton.cdiv(n * c, meta["BLOCK_ROWS"]),
        triton.cdiv(hw, meta["BLOCK_HW"]),
    )
    _paired_cat_bn_relu_kernel[grid](
        mean,
        conv,
        var,
        weight,
        bias,
        relu,
        output,
        N=n,
        C=c,
        H=h,
        W=w,
        conv_s0=conv.stride(0),
        conv_s1=conv.stride(1),
        conv_s2=conv.stride(2),
        conv_s3=conv.stride(3),
        relu_s0=relu.stride(0),
        relu_s1=relu.stride(1),
        relu_s2=relu.stride(2),
        relu_s3=relu.stride(3),
        out_s0=output.stride(0),
        out_s1=output.stride(1),
        out_s2=output.stride(2),
        out_s3=output.stride(3),
    )
    return output


def _check_oracle_equal_nan(instance, inputs, *, atol: float, rtol: float) -> bool:
    """Check values, dtype, shape, and stride for this NaN-producing repro."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        if oracle_out.is_cuda:
            torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tensors")
        return False

    ok = True
    if eager.shape != oracle_out.shape:
        print(f"  output 0: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} eager={list(eager.shape)}")
        ok = False
    if eager.dtype != oracle_out.dtype:
        print(f"  output 0: SCOPE_MISMATCH dtype oracle={oracle_out.dtype} eager={eager.dtype}")
        ok = False
    if eager.stride() != oracle_out.stride():
        print(f"  output 0: SCOPE_MISMATCH stride oracle={oracle_out.stride()} eager={eager.stride()}")
        ok = False

    if ok:
        eager_f32 = eager.float()
        oracle_f32 = oracle_out.float()
        max_diff = (eager_f32 - oracle_f32).abs().nan_to_num(0.0).max().item()
        values_ok = torch.allclose(eager_f32, oracle_f32, atol=atol, rtol=rtol, equal_nan=True)
        status = "PASS" if values_ok else "FAIL"
        print(
            f"  output 0: {status} (shape={list(eager.shape)} dtype={eager.dtype} "
            f"stride={eager.stride()} max_diff={max_diff:.2e} equal_nan=True)"
        )
        ok = values_ok

    return ok


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
        ok = _check_oracle_equal_nan(instance, inputs, atol=args.atol, rtol=args.rtol)
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
