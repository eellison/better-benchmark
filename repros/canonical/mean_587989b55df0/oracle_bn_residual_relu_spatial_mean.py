"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full f32 BN-inference affine, residual add, NaN-preserving ReLU, and spatial mean directly into the final contiguous `[N,C]` view for the recorded MoCo/ResNet shapes, whereas Inductor lowers the fused reduction without canonicalizing the batch-invariant BN affine algebra into reusable per-row scale/shift values; Inductor cannot do this today because its reduction codegen does not hoist normalization parameter algebra cleanly across the spatial reduction while preserving the residual epilogue and output layout; the fix is ALGEBRAIC_ELIMINATION: recognize inference BN affine feeding activation/spatial-mean reductions and generate row-wise scale/shift reuse inside the reduction kernel."""
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
# Do not add custom benchmark functions. bench_oracle() owns timing so graph
# capture, GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


EPS = 1.0e-5
BLOCK_HW = 64
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=8, num_stages=3),
        ],
        key=["total_rows", "channels", "hw_size"],
    )
    @triton.jit
    def _bn_residual_relu_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        channel_offsets = row_offsets - (row_offsets // channels) * channels
        hw_offsets = tl.arange(0, BLOCK_HW_)

        valid_rows = row_offsets < total_rows
        valid_hw = hw_offsets < hw_size
        valid = valid_rows[:, None] & valid_hw[None, :]

        elem_offsets = row_offsets[:, None] * hw_size + hw_offsets[None, :]
        x = tl.load(x_ptr + elem_offsets, mask=valid, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + elem_offsets, mask=valid, other=0.0).to(tl.float32)

        mean = tl.load(mean_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel_offsets, mask=valid_rows, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + eps)
        y = (x - mean[:, None]) * invstd[:, None]
        y = y * weight[:, None]
        y = y + bias[:, None]
        y = y + residual
        y = tl.where(y < 0.0, 0.0, y)
        reduced = tl.sum(tl.where(valid, y, 0.0), axis=1) * (1.0 / hw_size)

        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, residual, view_shape = inputs
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise TypeError("convolution_4 must be a 4D tensor")

    batch, channels, height, width = (int(dim) for dim in x.shape)
    if height != width or (height, width) not in ((7, 7), (8, 8)):
        raise ValueError(f"unsupported spatial shape {(height, width)}")
    if (batch, channels) not in ((32, 2048), (128, 64)):
        raise ValueError(f"unsupported recorded shape batch={batch} channels={channels}")
    if list(view_shape) != [batch, channels]:
        raise ValueError(f"unexpected output view shape parameter: {view_shape!r}")

    hw_size = height * width
    vector_shape = (channels,)
    vector_stride = (1,)
    image_shape = (batch, channels, height, width)
    image_stride = (channels * hw_size, hw_size, width, 1)

    mean_t = _require_f32_tensor("arg23_1", mean, vector_shape, vector_stride)
    x_t = _require_f32_tensor("convolution_4", x, image_shape, image_stride)
    var_t = _require_f32_tensor("arg24_1", var, vector_shape, vector_stride)
    weight_t = _require_f32_tensor("arg25_1", weight, vector_shape, vector_stride)
    bias_t = _require_f32_tensor("arg26_1", bias, vector_shape, vector_stride)
    residual_t = _require_f32_tensor("relu_1", residual, image_shape, image_stride)

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t, residual_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean_t, x_t, var_t, weight_t, bias_t, residual_t, batch, channels, hw_size


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: accepts the same seven inputs as Repro.forward() and
    returns the same single f32 contiguous `[N,C]` tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_residual_relu_spatial_mean.py")

    mean, x, var, weight, bias, residual, batch, channels, hw_size = _validate_inputs(inputs)
    output = torch.empty_strided(
        (batch, channels),
        (channels, 1),
        device=x.device,
        dtype=torch.float32,
    )
    total_rows = batch * channels
    grid = lambda meta: (triton.cdiv(total_rows, meta["BLOCK_ROWS"]),)
    _bn_residual_relu_spatial_mean_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        output,
        total_rows=total_rows,
        channels=channels,
        hw_size=hw_size,
        BLOCK_HW_=BLOCK_HW,
        eps=EPS,
    )
    return output


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate deterministic full-scope output, including matching NaN masks."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False
    if expected.shape != actual.shape:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
            f"eager={list(expected.shape)}"
        )
        return False
    if expected.stride() != actual.stride():
        print(
            f"  output 0: SCOPE_MISMATCH stride oracle={actual.stride()} "
            f"eager={expected.stride()}"
        )
        return False

    dtype_ok = expected.dtype == actual.dtype
    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_mask_ok = torch.equal(expected_nan, actual_nan)
    finite = ~expected_nan & ~actual_nan
    if finite.any():
        finite_expected = expected_f32[finite]
        finite_actual = actual_f32[finite]
        max_diff = (finite_expected - finite_actual).abs().max().item()
        values_ok = torch.allclose(finite_expected, finite_actual, atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = dtype_ok and nan_mask_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
    )
    if not dtype_ok:
        print(f"  output 0: WARNING dtype mismatch oracle={actual.dtype} eager={expected.dtype}")
    if not nan_mask_ok:
        print(
            f"  output 0: NaN mask mismatch "
            f"(expected_nan={int(expected_nan.sum().item())}, oracle_nan={int(actual_nan.sum().item())})"
        )
    return ok


# --- CLI entry point ---
def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {result['status']}")


if __name__ == "__main__":
    main()
