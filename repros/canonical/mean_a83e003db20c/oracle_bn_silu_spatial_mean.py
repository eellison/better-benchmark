"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete f32 EfficientNet BN-inference affine, SiLU producer, and spatial mean in one Triton reduction kernel that writes only the final `[N,C,1,1]` tensor, whereas Inductor currently lowers the decomposed broadcast pointwise chain and small spatial mean through its generic fused reduction schedule; Inductor cannot do this today because its scheduler/codegen lacks a reusable guarded template for per-channel BN-affine plus SiLU feeding a fixed spatial reduction while preserving f32 NaN propagation; the fix is SCHEDULER_FUSION: add a benchmark-gated BN-affine activation-spatial-mean reduction template that folds the broadcast parameters into the reduction kernel and falls back to the generic schedule when it wins."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
EPS = 1.0e-3
CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True


if triton is not None:

    @triton.jit
    def _bn_silu_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        stride_n: tl.constexpr,
        stride_c: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        valid_rows = row_offsets < total_rows
        valid_hw = hw_offsets < hw_size

        n_offsets = row_offsets // channels
        c_offsets = row_offsets - n_offsets * channels
        h_offsets = hw_offsets // width
        w_offsets = hw_offsets - h_offsets * width

        x_offsets = (
            n_offsets[:, None] * stride_n
            + c_offsets[:, None] * stride_c
            + h_offsets[None, :] * stride_h
            + w_offsets[None, :] * stride_w
        )
        x = tl.load(
            x_ptr + x_offsets,
            mask=valid_rows[:, None] & valid_hw[None, :],
            other=0.0,
        ).to(tl.float32)

        mean = tl.load(mean_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=valid_rows, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=valid_rows, other=0.0).to(tl.float32)

        scale = weight / tl.sqrt(var + eps)
        shift = bias - mean * scale
        affine = x * scale[:, None] + shift[:, None]
        silu = affine * tl.sigmoid(affine)
        reduced = tl.sum(tl.where(valid_hw[None, :], silu, 0.0), axis=1) / hw_size

        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _select_block_rows(hw_size: int) -> int:
    if hw_size <= 64:
        return 8
    if hw_size <= 1024:
        return 8
    if hw_size <= 4096:
        return 4
    return 1


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise TypeError("convolution_76 must be a rank-4 tensor")

    batch, channels, height, width = (int(dim) for dim in x.shape)
    if batch != 128:
        raise ValueError(f"unsupported batch size {batch}; expected 128")
    if height != width or height not in (7, 14, 28, 56, 112):
        raise ValueError(f"unsupported spatial shape {(height, width)}")

    mean_t = _require_f32_tensor("arg292_1", mean, (channels,))
    x_t = _require_f32_tensor("convolution_76", x, (batch, channels, height, width))
    var_t = _require_f32_tensor("arg293_1", var, (channels,))
    weight_t = _require_f32_tensor("arg294_1", weight, (channels,))
    bias_t = _require_f32_tensor("arg295_1", bias, (channels,))

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x_t, var_t, weight_t, bias_t


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full-scope BN-affine SiLU spatial-mean oracle."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_silu_spatial_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    batch, channels, height, width = (int(dim) for dim in x.shape)
    hw_size = height * width
    block_rows = _select_block_rows(hw_size)
    block_hw = triton.next_power_of_2(hw_size)
    output = torch.empty_strided(
        (batch, channels, 1, 1),
        (channels, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _bn_silu_spatial_mean_kernel[(triton.cdiv(batch * channels, block_rows),)](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        stride_n=x.stride(0),
        stride_c=x.stride(1),
        stride_h=x.stride(2),
        stride_w=x.stride(3),
        total_rows=batch * channels,
        channels=channels,
        width=width,
        hw_size=hw_size,
        BLOCK_ROWS_=block_rows,
        BLOCK_HW_=block_hw,
        eps=EPS,
        num_warps=4,
        num_stages=3,
    )
    return output


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate the deterministic full-scope output, including matching NaNs."""
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

    dtype_ok = expected.dtype == actual.dtype
    layout_ok = expected.stride() == actual.stride()

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

    ok = dtype_ok and layout_ok and nan_mask_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} "
        f"oracle_dtype={actual.dtype} expected_stride={expected.stride()} "
        f"oracle_stride={actual.stride()} nan_mask={'PASS' if nan_mask_ok else 'FAIL'} "
        f"max_finite_diff={max_diff:.2e})"
    )
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
