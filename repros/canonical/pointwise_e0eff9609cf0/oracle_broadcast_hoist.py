"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full batchnorm-affine plus ReLU6 pointwise region by precomputing each channel's affine scale/shift and applying them in a coalesced layout-preserving pass, whereas Inductor emits a single flat pointwise kernel that recomputes the broadcasted reciprocal-sqrt expression for every output element; Inductor cannot express this two-stage broadcast hoist today because pointwise scheduling lacks broadcast-aware materialization, but the measured full-scope oracle is at floor because the extra launch and temporary traffic offset the saved math and the region is dominated by dense f32 input/output bandwidth; the fix is BANDWIDTH_BOUND: no stencil/layout fusion change is indicated beyond cost-modeling this case to keep the current single-pass fusion."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _precompute_bn_affine_kernel(
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        scale_ptr,
        shift_ptr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """Precompute per-channel scale and shift for the expanded pointwise pass."""
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0)
        inv_std = 1.0 / tl.sqrt(var + 1.0e-5)
        scale = inv_std * weight
        shift = bias - mean * scale
        tl.store(scale_ptr + c_offsets, scale, mask=c_mask)
        tl.store(shift_ptr + c_offsets, shift, mask=c_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL", "C", "HW"],
    )
    @triton.jit
    def _apply_relu6_nchw_kernel(
        input_ptr,
        scale_ptr,
        shift_ptr,
        output_ptr,
        TOTAL: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Flat contiguous NCHW apply pass."""
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        c_offsets = (offsets // HW) % C
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + c_offsets, mask=mask, other=0.0)
        shift = tl.load(shift_ptr + c_offsets, mask=mask, other=0.0)
        y = x * scale + shift
        y_clamped = tl.where(y != y, y, tl.minimum(tl.maximum(y, 0.0), 6.0))
        tl.store(output_ptr + offsets, y_clamped, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL", "C"],
    )
    @triton.jit
    def _apply_relu6_channels_last_kernel(
        input_ptr,
        scale_ptr,
        shift_ptr,
        output_ptr,
        TOTAL: tl.constexpr,
        C: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Flat physical-order apply pass for dense channels-last inputs."""
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        c_offsets = offsets % C
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + c_offsets, mask=mask, other=0.0)
        shift = tl.load(shift_ptr + c_offsets, mask=mask, other=0.0)
        y = x * scale + shift
        y_clamped = tl.where(y != y, y, tl.minimum(tl.maximum(y, 0.0), 6.0))
        tl.store(output_ptr + offsets, y_clamped, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
        ],
        key=["TOTAL", "C", "H", "W", "S_N", "S_C", "S_H", "S_W"],
    )
    @triton.jit
    def _apply_relu6_strided_kernel(
        input_ptr,
        scale_ptr,
        shift_ptr,
        output_ptr,
        TOTAL: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        HW: tl.constexpr,
        S_N: tl.constexpr,
        S_C: tl.constexpr,
        S_H: tl.constexpr,
        S_W: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Generic logical-order apply pass for dense but unusual strides."""
        logical = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = logical < TOTAL
        hw = logical % HW
        c_offsets = (logical // HW) % C
        b_offsets = logical // (C * HW)
        h_offsets = hw // W
        w_offsets = hw - h_offsets * W
        physical = (
            b_offsets * S_N
            + c_offsets * S_C
            + h_offsets * S_H
            + w_offsets * S_W
        )
        x = tl.load(input_ptr + physical, mask=mask, other=0.0)
        scale = tl.load(scale_ptr + c_offsets, mask=mask, other=0.0)
        shift = tl.load(shift_ptr + c_offsets, mask=mask, other=0.0)
        y = x * scale + shift
        y_clamped = tl.where(y != y, y, tl.minimum(tl.maximum(y, 0.0), 6.0))
        tl.store(output_ptr + physical, y_clamped, mask=mask)


def _normalize_outputs(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def check_oracle_equal_nan_and_stride(
    oracle_forward_fn,
    instance,
    inputs,
    *,
    atol,
    rtol,
    skip_stochastic=True,
):
    """Correctness check with NaN equality plus the stride invariant."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward_fn(inputs)

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)

    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
                f"eager={list(expected.shape)}"
            )
            all_pass = False
            continue

        if expected.dtype != actual.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual.dtype} "
                f"eager={expected.dtype}"
            )
            all_pass = False
            continue

        if expected.stride() != actual.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual.stride()} "
                f"eager={expected.stride()}"
            )
            all_pass = False
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, actual)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
            all_pass &= ok
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        finite = torch.isfinite(expected_f32) & torch.isfinite(actual_f32)
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        else:
            max_diff = 0.0
        nan_count = torch.isnan(expected_f32).sum().item()
        ok = torch.allclose(
            expected_f32,
            actual_f32,
            atol=atol,
            rtol=rtol,
            equal_nan=True,
        )
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={expected.stride()} finite_max_diff={max_diff:.2e} "
            f"nan_count={nan_count})"
        )
        all_pass &= ok

    return all_pass


@oracle_impl(hardware="H100", shapes="(T([960], f32), T([128, 960, 7, 7], f32), T([960], f32), T([960], f32), T([960], f32))")
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
        raise RuntimeError("Triton is required for oracle_broadcast_hoist.py")
    if len(inputs) != 5:
        raise ValueError(f"expected 5 inputs, got {len(inputs)}")

    mean, convolution, var, weight, bias = inputs
    if convolution.dim() != 4:
        raise ValueError(f"expected 4D convolution input, got {convolution.dim()}D")
    if convolution.dtype is not torch.float32:
        raise ValueError(f"unexpected convolution dtype: {convolution.dtype}")
    if not convolution.is_cuda:
        raise ValueError("oracle_broadcast_hoist.py expects CUDA inputs")

    batch, channels, height, width = convolution.shape
    for name, tensor in (
        ("mean", mean),
        ("var", var),
        ("weight", weight),
        ("bias", bias),
    ):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"unexpected {name} shape: {tuple(tensor.shape)}")
        if tensor.dtype is not torch.float32:
            raise ValueError(f"unexpected {name} dtype: {tensor.dtype}")
        if tensor.device != convolution.device:
            raise ValueError(f"{name} is on {tensor.device}, expected {convolution.device}")

    output = torch.empty_strided(
        tuple(convolution.shape),
        tuple(convolution.stride()),
        device=convolution.device,
        dtype=convolution.dtype,
    )
    scale = torch.empty((channels,), device=convolution.device, dtype=torch.float32)
    shift = torch.empty((channels,), device=convolution.device, dtype=torch.float32)
    hw = height * width
    total = convolution.numel()
    s_n, s_c, s_h, s_w = convolution.stride()

    _precompute_bn_affine_kernel[(triton.cdiv(channels, 1024),)](
        mean,
        var,
        weight,
        bias,
        scale,
        shift,
        C=channels,
        BLOCK_C=1024,
    )

    contiguous_stride = (channels * hw, hw, width, 1)
    channels_last_stride = (channels * hw, 1, width * channels, channels)
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_N"]),)
    if tuple(convolution.stride()) == contiguous_stride:
        _apply_relu6_nchw_kernel[grid](
            convolution,
            scale,
            shift,
            output,
            TOTAL=total,
            C=channels,
            HW=hw,
        )
    elif tuple(convolution.stride()) == channels_last_stride:
        _apply_relu6_channels_last_kernel[grid](
            convolution,
            scale,
            shift,
            output,
            TOTAL=total,
            C=channels,
        )
    else:
        _apply_relu6_strided_kernel[grid](
            convolution,
            scale,
            shift,
            output,
            TOTAL=total,
            C=channels,
            H=height,
            W=width,
            HW=hw,
            S_N=s_n,
            S_C=s_c,
            S_H=s_h,
            S_W=s_w,
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
        ok = check_oracle_equal_nan_and_stride(
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
