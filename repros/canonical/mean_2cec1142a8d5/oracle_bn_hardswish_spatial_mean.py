"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 BN-inference affine, hard-swish activation, and 7x7 spatial mean returned by Repro.forward in one Triton reduction kernel while preserving the final as_strided-compatible `[512,960,1,1]` output layout, whereas Inductor currently lowers the decomposed broadcast normalization, clamp-based hard-swish, and mean through a generic fused reduction schedule that still carries the activation producer inside reduction codegen rather than a dedicated BN-activation-spatial-mean template; Inductor cannot do this today because its norm-template canonicalization does not recognize fixed small-spatial hard-swish mean pooling as a benchmark-gated fused schedule with direct output-layout emission; the fix is SCHEDULER_FUSION: add a guarded BN-affine hard-swish spatial-mean lowering for NCHW tensors with contiguous or channels-last strides."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps import/py_compile useful without Triton.
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
BATCH = 512
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ROWS = BATCH * CHANNELS
EPS = 1.0e-5
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
CONTIGUOUS_INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
CHANNELS_LAST_INPUT_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
PARAM_SHAPE = (CHANNELS,)
OUTPUT_SHAPE = (BATCH, CHANNELS, 1, 1)
OUTPUT_STRIDE = (CHANNELS, 1, CHANNELS, CHANNELS)
BLOCK_HW = 64
CLASSIFICATION = "SCHEDULER_FUSION"


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 64}, num_warps=8, num_stages=3),
        ],
        key=[
            "total_rows",
            "channels",
            "x_stride_n",
            "x_stride_c",
            "x_stride_h",
            "x_stride_w",
        ],
    )
    @triton.jit
    def _bn_hardswish_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        hw: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        spatial = tl.arange(0, BLOCK_HW_)
        row_mask = rows < total_rows
        spatial_mask = spatial < hw

        n = rows // channels
        c = rows - n * channels
        h = spatial // width
        w = spatial - h * width
        offsets = (
            n[:, None] * x_stride_n
            + c[:, None] * x_stride_c
            + h[None, :] * x_stride_h
            + w[None, :] * x_stride_w
        )
        mask = row_mask[:, None] & spatial_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

        normalized = (x - mean[:, None]) * tl.rsqrt(var[:, None] + eps)
        affine = normalized * weight[:, None] + bias[:, None]
        relu6 = tl.minimum(tl.maximum(affine + 3.0, 0.0), 6.0)
        hardswish = affine * relu6 * (1.0 / 6.0)
        reduced = tl.sum(tl.where(spatial_mask[None, :], hardswish, 0.0), axis=1)
        reduced = reduced * (1.0 / 49.0)

        tl.store(out_ptr + rows, reduced, mask=row_mask)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    allowed_strides: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if allowed_strides is not None and tuple(value.stride()) not in allowed_strides:
        raise ValueError(
            f"{name} has stride {tuple(value.stride())}, expected one of {allowed_strides}"
        )
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    mean_t = _require_f32_tensor("arg259_1", mean, PARAM_SHAPE, ((1,),))
    x_t = _require_f32_tensor(
        "convolution_61",
        x,
        INPUT_SHAPE,
        (CONTIGUOUS_INPUT_STRIDE, CHANNELS_LAST_INPUT_STRIDE),
    )
    var_t = _require_f32_tensor("arg260_1", var, PARAM_SHAPE, ((1,),))
    weight_t = _require_f32_tensor("arg261_1", weight, PARAM_SHAPE, ((1,),))
    bias_t = _require_f32_tensor("arg262_1", bias, PARAM_SHAPE, ((1,),))

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean_t, x_t, var_t, weight_t, bias_t


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same output shape, dtype, values, and stride.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_hardswish_spatial_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_ROWS"]),)
    _bn_hardswish_spatial_mean_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        total_rows=ROWS,
        channels=CHANNELS,
        width=WIDTH,
        hw=HW,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        eps=EPS,
        BLOCK_HW_=BLOCK_HW,
    )
    return output


def _normalize_outputs(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        return list(value)
    return [value]


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    expected_outputs = _normalize_outputs(expected)
    actual_outputs = _normalize_outputs(actual)
    if len(expected_outputs) != len(actual_outputs):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_outputs)} outputs, "
            f"eager produces {len(expected_outputs)}"
        )
        return False

    all_pass = True
    for index, (expected_output, actual_output) in enumerate(
        zip(expected_outputs, actual_outputs)
    ):
        if not isinstance(expected_output, torch.Tensor) or not isinstance(
            actual_output, torch.Tensor
        ):
            ok = expected_output == actual_output
            print(f"  output {index}: {'PASS' if ok else 'FAIL'} (non-tensor)")
            all_pass = all_pass and bool(ok)
            continue

        ok = True
        if expected_output.shape != actual_output.shape:
            print(
                f"  output {index}: SCOPE_MISMATCH shape oracle={list(actual_output.shape)} "
                f"eager={list(expected_output.shape)}"
            )
            ok = False
        if expected_output.stride() != actual_output.stride():
            print(
                f"  output {index}: SCOPE_MISMATCH stride oracle={list(actual_output.stride())} "
                f"eager={list(expected_output.stride())}"
            )
            ok = False
        if expected_output.dtype != actual_output.dtype:
            print(
                f"  output {index}: SCOPE_MISMATCH dtype oracle={actual_output.dtype} "
                f"eager={expected_output.dtype}"
            )
            ok = False

        if not ok:
            all_pass = False
            continue

        if not expected_output.is_floating_point():
            values_ok = torch.equal(expected_output, actual_output)
            print(
                f"  output {index}: {'PASS' if values_ok else 'FAIL'} "
                f"(shape={list(expected_output.shape)} dtype={expected_output.dtype}, exact)"
            )
            all_pass = all_pass and values_ok
            continue

        expected_f32 = expected_output.float()
        actual_f32 = actual_output.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_masks_ok = torch.equal(expected_nan, actual_nan)
        finite_mask = torch.isfinite(expected_f32) & torch.isfinite(actual_f32)
        if finite_mask.any():
            max_diff = (expected_f32[finite_mask] - actual_f32[finite_mask]).abs().max().item()
            finite_ok = torch.allclose(
                expected_f32[finite_mask],
                actual_f32[finite_mask],
                atol=atol,
                rtol=rtol,
            )
        else:
            max_diff = 0.0
            finite_ok = True
        values_ok = nan_masks_ok and finite_ok
        status = "PASS" if values_ok else "FAIL"
        print(
            f"  output {index}: {status} (shape={list(expected_output.shape)} "
            f"stride={list(expected_output.stride())} dtype={expected_output.dtype} "
            f"max_finite_diff={max_diff:.2e} nan_mask_match={nan_masks_ok})"
        )
        all_pass = all_pass and values_ok

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
