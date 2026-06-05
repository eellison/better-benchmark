"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ResNet inference BN-affine, fp16 cast, residual add, and ReLU scope by hoisting the channel-invariant normalization into fp32 scale/shift coefficients and applying the full epilogue in one flat Triton output kernel, whereas Inductor's compiled graph is already within timing noise of that full-scope hand kernel on the measured shape; Inductor cannot get a confirmed local win here because the remaining cost is dominated by required fp16 input/residual reads, output stores, parameter traffic, and launch overhead rather than a missing layout/stencil fusion; the fix is BANDWIDTH_BOUND: treat this repro as at floor unless a broader BN-affine-residual template beats the current compiled pointwise schedule on the same full scope."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 512}, num_warps=8, num_stages=3),
        ],
        key=["C"],
    )
    @triton.jit
    def _bn_scale_shift_kernel(
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        coeff_ptr,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = c_offsets < C

        mean = tl.load(mean_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)

        scale = weight / tl.sqrt(var + 1.0e-5)
        shift = bias - mean * scale
        tl.store(coeff_ptr + c_offsets, scale, mask=mask)
        tl.store(coeff_ptr + C + c_offsets, shift, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL", "C", "HW"],
    )
    @triton.jit
    def _bn_residual_relu_kernel(
        x_ptr,
        residual_ptr,
        coeff_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        safe_offsets = tl.where(mask, offsets, 0)
        channel = (safe_offsets // HW) % C

        x = tl.load(x_ptr + safe_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + safe_offsets, mask=mask, other=0.0)
        scale = tl.load(coeff_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(coeff_ptr + C + channel, mask=mask, other=0.0).to(tl.float32)

        affine_h = (x * scale + shift).to(tl.float16)
        y = (affine_h + residual).to(tl.float16)
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        tl.store(out_ptr + offsets, y, mask=mask)


def _require_f16_cuda_contiguous(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} must be torch.float16, got {value.dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must use the captured contiguous layout")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_residual_relu.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    mean, x, var, weight, bias, residual = (
        _require_f16_cuda_contiguous(name, value)
        for name, value in zip(
            (
                "arg757_1",
                "convolution_151",
                "arg758_1",
                "arg759_1",
                "arg760_1",
                "relu_144",
            ),
            inputs,
        )
    )

    if x.ndim != 4:
        raise ValueError(f"convolution_151 must be rank 4, got shape {tuple(x.shape)}")
    n, c, h, w = x.shape
    if n <= 0 or c <= 0 or h <= 0 or w <= 0:
        raise ValueError(f"convolution_151 has invalid shape {tuple(x.shape)}")
    if residual.shape != x.shape:
        raise ValueError(f"relu_144 shape must match convolution_151, got {tuple(residual.shape)}")
    if residual.device != x.device:
        raise ValueError("relu_144 must be on the same CUDA device as convolution_151")
    for name, tensor in (
        ("arg757_1", mean),
        ("arg758_1", var),
        ("arg759_1", weight),
        ("arg760_1", bias),
    ):
        if tuple(tensor.shape) != (c,):
            raise ValueError(f"{name} shape must be ({c},), got {tuple(tensor.shape)}")
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same CUDA device as convolution_151")

    return mean, x, var, weight, bias, residual


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
    mean, x, var, weight, bias, residual = _validate_inputs(inputs)
    n, c, h, w = x.shape
    hw = h * w
    total = n * c * hw

    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float16,
    )
    coeff = torch.empty((2, c), device=x.device, dtype=torch.float32)

    coeff_grid = lambda meta: (triton.cdiv(c, meta["BLOCK_C"]),)
    _bn_scale_shift_kernel[coeff_grid](
        mean,
        var,
        weight,
        bias,
        coeff,
        C=c,
    )

    out_grid = lambda meta: (triton.cdiv(total, meta["BLOCK_N"]),)
    _bn_residual_relu_kernel[out_grid](
        x,
        residual,
        coeff,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
    )
    return out


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (tuple, list)):
        return tuple(outputs)
    return (outputs,)


def _check_oracle_nan_aware(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    actual_list = _normalize_outputs(actual)
    if len(eager_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (expected, observed) in enumerate(zip(eager_list, actual_list)):
        if not isinstance(expected, torch.Tensor) or not isinstance(observed, torch.Tensor):
            ok = expected == observed
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        metadata_ok = (
            expected.shape == observed.shape
            and expected.dtype == observed.dtype
            and expected.stride() == observed.stride()
            and expected.device == observed.device
        )
        if not metadata_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"oracle=(shape={list(observed.shape)} stride={observed.stride()} dtype={observed.dtype}) "
                f"eager=(shape={list(expected.shape)} stride={expected.stride()} dtype={expected.dtype})"
            )
            all_pass = False
            continue

        if not expected.is_floating_point():
            ok = torch.equal(expected, observed)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
            all_pass = all_pass and ok
            continue

        expected_f32 = expected.float()
        observed_f32 = observed.float()
        expected_nan = torch.isnan(expected_f32)
        observed_nan = torch.isnan(observed_f32)
        nan_mask_ok = torch.equal(expected_nan, observed_nan)
        finite = ~(expected_nan | observed_nan)
        if finite.any():
            max_diff = (expected_f32[finite] - observed_f32[finite]).abs().max().item()
            finite_ok = torch.allclose(expected_f32[finite], observed_f32[finite], atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            finite_ok = True

        ok = nan_mask_ok and finite_ok
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
            f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
        )
        all_pass = all_pass and ok

    return all_pass


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
        ok = _check_oracle_nan_aware(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
