"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full ResNet inference BN-affine, required fp16 round, residual fp16 add, NaN-preserving ReLU, and 7x7 spatial mean in one shape-specialized row-tiled reduction kernel that returns only the final fp16 [N,C] view, whereas Inductor emits a generic persistent reduction that repeats per-row BN parameter work and does not preserve the explicit fp16 cast before the residual add in its generated arithmetic; Inductor cannot do this today because norm/reduction codegen has no BN-residual-spatial-mean template with a required fp16 producer boundary and channel-row tiling for fixed 7x7 reductions; the fix is NEW_PATTERN: add a ResNet BN residual ReLU spatial-mean lowering that preserves the fp16 round/add semantics and emits a tuned small-spatial reduction directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
BLOCK_ROWS = 8
BLOCK_HW = 64
CLASSIFICATION = "NEW_PATTERN"
ACTIONABLE = False
TRUE_FLOOR = True


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _bn_residual_relu_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        out_ptr,
        total_rows,
        channels: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        channel_offsets = row_offsets - (row_offsets // channels) * channels
        hw_offsets = tl.arange(0, BLOCK_HW_)

        valid_rows = row_offsets < total_rows
        valid_hw = hw_offsets < hw
        valid = valid_rows[:, None] & valid_hw[None, :]

        element_offsets = row_offsets[:, None] * hw + hw_offsets[None, :]
        x = tl.load(x_ptr + element_offsets, mask=valid, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + element_offsets, mask=valid, other=0.0).to(tl.float16)

        mean = tl.load(mean_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel_offsets, mask=valid_rows, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)

        normalized = (x - mean[:, None]) * tl.rsqrt(var[:, None] + 1.0e-5)
        affine_h = (normalized * weight[:, None] + bias[:, None]).to(tl.float16)
        added_h = (affine_h + residual).to(tl.float16)
        zero_h = tl.full((BLOCK_ROWS_, BLOCK_HW_), 0.0, tl.float16)
        relu_h = tl.where(added_h < zero_h, zero_h, added_h)
        reduced = tl.sum(tl.where(valid, relu_h.to(tl.float32), 0.0), axis=1) * (1.0 / 49.0)

        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


def _require_f16_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype is not torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, residual, view_shape = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"convolution_154 must be a tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"convolution_154 must be rank 4, got shape {tuple(x.shape)}")
    batch, channels, height, width = x.shape
    if (height, width) != (HEIGHT, WIDTH):
        raise ValueError(f"expected 7x7 spatial shape, got {(height, width)}")

    vector_shape = (channels,)
    vector_stride = (1,)
    image_shape = (batch, channels, HEIGHT, WIDTH)
    image_stride = (channels * HW, HW, WIDTH, 1)

    mean_t = _require_f16_tensor("arg772_1", mean, vector_shape, vector_stride)
    x_t = _require_f16_tensor("convolution_154", x, image_shape, image_stride)
    var_t = _require_f16_tensor("arg773_1", var, vector_shape, vector_stride)
    weight_t = _require_f16_tensor("arg774_1", weight, vector_shape, vector_stride)
    bias_t = _require_f16_tensor("arg775_1", bias, vector_shape, vector_stride)
    residual_t = _require_f16_tensor("relu_147", residual, image_shape, image_stride)

    if list(view_shape) != [batch, channels]:
        raise ValueError(f"unexpected output view shape parameter: {view_shape!r}")

    device = x_t.device
    if any(t.device != device for t in (mean_t, var_t, weight_t, bias_t, residual_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x_t, var_t, weight_t, bias_t, residual_t, batch, channels


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_resnet_bn_residual_spatial_mean.py")

    mean, x, var, weight, bias, residual, batch, channels = _validate_inputs(inputs)
    output = torch.empty_strided((batch, channels), (channels, 1), device=x.device, dtype=torch.float16)

    _bn_residual_relu_spatial_mean_kernel[(triton.cdiv(batch * channels, BLOCK_ROWS),)](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        output,
        batch * channels,
        channels=channels,
        hw=HW,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        num_warps=1,
        num_stages=3,
    )
    return output


def _max_finite_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).abs()
    finite = torch.isfinite(diff)
    if not finite.any():
        return 0.0
    return float(diff[finite].max().item())


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate full-scope values while treating deterministic NaNs as equal."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    shape_ok = expected.shape == actual.shape
    dtype_ok = expected.dtype == actual.dtype
    stride_ok = expected.stride() == actual.stride()
    if not shape_ok:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
            f"eager={list(expected.shape)}"
        )
        return False

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_ok = torch.equal(expected_nan, actual_nan)
    finite = ~(expected_nan | actual_nan)
    if finite.any():
        values_ok = torch.allclose(actual_f32[finite], expected_f32[finite], atol=atol, rtol=rtol)
    else:
        values_ok = True
    max_diff = _max_finite_diff(actual, expected)

    ok = shape_ok and dtype_ok and stride_ok and nan_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
    )
    if not dtype_ok:
        print(f"  output 0: SCOPE_MISMATCH dtype oracle={actual.dtype} eager={expected.dtype}")
    if not stride_ok:
        print(f"  output 0: SCOPE_MISMATCH stride oracle={actual.stride()} eager={expected.stride()}")
    if not nan_ok:
        print(
            f"  output 0: NaN mask mismatch "
            f"(oracle_nan={int(actual_nan.sum().item())}, eager_nan={int(expected_nan.sum().item())})"
        )
    return ok


# --- CLI entry point ---
def main() -> None:
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
        ok = _run_check(
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
