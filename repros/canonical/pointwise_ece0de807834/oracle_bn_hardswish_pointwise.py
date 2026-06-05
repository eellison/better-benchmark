"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete MobileNetV3 fp32 BN-inference affine plus hard-swish scope by hoisting the channel-invariant mean/variance/weight/bias expression into per-channel scale and shift vectors, then streaming the dense NCHW or channels-last activation once into the exact returned layout, whereas Inductor lowers the decomposed unsqueeze/sub/sqrt/reciprocal/mul/add/clamp/mul/div graph as a generic pointwise schedule that recomputes the sqrt-derived broadcast affine arithmetic for every output element; Inductor cannot do this today because its algebraic simplifier/codegen does not factor broadcast-invariant BN-inference subexpressions before the hard-swish epilogue; the fix is ALGEBRAIC_ELIMINATION: add a guarded BN-affine factoring rewrite that materializes or hoists per-channel scale/shift when the accuracy policy allows the reassociation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps import usable without Triton.
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


EPS = 1.0e-5
LAYOUT_CONTIGUOUS = 0
LAYOUT_CHANNELS_LAST = 1
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _affine_params_kernel(
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        scale_ptr,
        shift_ptr,
        C: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = c_offsets < C

        mean = tl.load(mean_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=mask, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + eps)
        scale = invstd * weight
        shift = bias - mean * scale
        tl.store(scale_ptr + c_offsets, scale, mask=mask)
        tl.store(shift_ptr + c_offsets, shift, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4096}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL", "C", "HW", "LAYOUT"],
    )
    @triton.jit
    def _affine_hardswish_kernel(
        x_ptr,
        scale_ptr,
        shift_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        LAYOUT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        if LAYOUT == 0:
            channels = (offsets // HW) % C
        else:
            channels = offsets % C

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(
            scale_ptr + channels,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        shift = tl.load(
            shift_ptr + channels,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        )

        affine = x * scale + shift
        relu6 = tl.maximum(affine + 3.0, 0.0, propagate_nan=tl.PropagateNan.ALL)
        relu6 = tl.minimum(relu6, 6.0, propagate_nan=tl.PropagateNan.ALL)
        out = affine * relu6 * (1.0 / 6.0)
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_f32_cuda_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if value.storage_offset() != 0:
        raise ValueError(f"{name} must have storage_offset=0")
    return value


def _layout_kind(x: torch.Tensor) -> int:
    if x.is_contiguous():
        return LAYOUT_CONTIGUOUS
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last):
        return LAYOUT_CHANNELS_LAST
    raise ValueError(
        f"unsupported activation layout for {REPRO_ID}: "
        f"shape={tuple(x.shape)} stride={tuple(x.stride())}"
    )


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_hardswish_pointwise.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    mean, x, var, weight, bias = (
        _require_f32_cuda_tensor(name, value)
        for name, value in zip(
            ("arg240_1", "convolution_56", "arg241_1", "arg242_1", "arg243_1"),
            inputs,
        )
    )

    if x.dim() != 4:
        raise ValueError(f"convolution_56 must be rank 4, got shape={tuple(x.shape)}")

    _, channels, height, width = x.shape
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"convolution_56 has invalid shape {tuple(x.shape)}")

    for name, tensor in (
        ("arg240_1", mean),
        ("arg241_1", var),
        ("arg242_1", weight),
        ("arg243_1", bias),
    ):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"{name} shape must be ({channels},), got {tuple(tensor.shape)}")
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same CUDA device as convolution_56")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    return mean, x, var, weight, bias, _layout_kind(x)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward BN-inference plus hard-swish scope."""
    mean, x, var, weight, bias, layout = _validate_inputs(inputs)
    _, channels, height, width = x.shape
    hw = height * width
    total = x.numel()

    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )
    scale = torch.empty((channels,), device=x.device, dtype=torch.float32)
    shift = torch.empty((channels,), device=x.device, dtype=torch.float32)

    _affine_params_kernel[(triton.cdiv(channels, 256),)](
        mean,
        var,
        weight,
        bias,
        scale,
        shift,
        C=channels,
        eps=EPS,
        BLOCK_C=256,
        num_warps=8,
        num_stages=3,
    )
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_N"]),)
    _affine_hardswish_kernel[grid](
        x,
        scale,
        shift,
        output,
        TOTAL=total,
        C=channels,
        HW=hw,
        LAYOUT=layout,
    )
    return output


def _normalize_outputs(outputs: Any) -> tuple[Any, ...]:
    if isinstance(outputs, (tuple, list)):
        return tuple(outputs)
    return (outputs,)


def _check_oracle_equal_nan(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """NaN-aware check for deterministic NaNs from random variance inputs."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in inputs):
            torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    all_pass = True
    for i, (expected_value, actual_value) in enumerate(zip(expected_list, actual_list)):
        if not isinstance(expected_value, torch.Tensor) or not isinstance(actual_value, torch.Tensor):
            ok = expected_value == actual_value
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        metadata_ok = (
            expected_value.shape == actual_value.shape
            and expected_value.dtype == actual_value.dtype
            and expected_value.stride() == actual_value.stride()
            and expected_value.device == actual_value.device
        )
        if not metadata_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"oracle=(shape={list(actual_value.shape)} stride={actual_value.stride()} "
                f"dtype={actual_value.dtype}) "
                f"eager=(shape={list(expected_value.shape)} stride={expected_value.stride()} "
                f"dtype={expected_value.dtype})"
            )
            all_pass = False
            continue

        if not expected_value.is_floating_point():
            ok = torch.equal(expected_value, actual_value)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected_value.dtype})")
            all_pass = all_pass and ok
            continue

        expected_f32 = expected_value.float()
        actual_f32 = actual_value.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_mask_ok = torch.equal(expected_nan, actual_nan)
        comparable = ~(expected_nan | actual_nan)
        if comparable.any():
            expected_cmp = expected_f32[comparable]
            actual_cmp = actual_f32[comparable]
            diff = (expected_cmp - actual_cmp).abs()
            finite_diff = diff[torch.isfinite(diff)]
            max_diff = finite_diff.max().item() if finite_diff.any() else 0.0
            values_ok = torch.allclose(expected_cmp, actual_cmp, atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            values_ok = True

        ok = nan_mask_ok and values_ok
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_value.shape)} dtype={expected_value.dtype} "
            f"stride={expected_value.stride()} max_finite_diff={max_diff:.2e} "
            f"nan_count={int(expected_nan.sum().item())})"
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
        ok = _check_oracle_equal_nan(
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
