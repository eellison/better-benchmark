"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Visformer residual add, per-channel normalization affine, 7x7 spatial mean, as_strided-compatible reshape, and final contiguous f32 `[128,768]` output in one channels-last Triton reduction kernel, but algebraically moves the batch-invariant affine from every spatial element to the single reduced output value; Inductor currently fuses this scope as a generic norm-template spatial reduction but still evaluates the broadcast subtract/rsqrt/multiply/bias chain inside the 49-element reduction body, and it cannot do this today because its reduction simplifier does not canonicalize affine producers through `mean(dim=[-1,-2])` when broadcast parameters and non-contiguous logical strides are involved; the fix is ALGEBRAIC_ELIMINATION: teach Inductor to sink/hoist per-channel affine normalization across fixed spatial means and emit the direct final view layout for channels-last NCHW tensors."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 128
CHANNELS = 768
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ROWS = BATCH * CHANNELS
EPS = 1.0e-5
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
CHANNELS_LAST_INPUT_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
PARAM_SHAPE = (CHANNELS,)
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
BLOCK_HW = 64
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 8}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 64}, num_warps=8, num_stages=3),
        ],
        key=["n_rows", "channels", "hw"],
    )
    @triton.jit
    def _channels_last_affine_spatial_mean_kernel(
        add_ptr,
        convolution_ptr,
        mean_ptr,
        variance_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        channels: tl.constexpr,
        width: tl.constexpr,
        hw: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        spatial = tl.arange(0, BLOCK_HW_)
        row_mask = rows < n_rows
        spatial_mask = spatial < hw

        n = rows // channels
        c = rows - n * channels
        h = spatial // width
        w = spatial - h * width
        offsets = (
            n[:, None] * (channels * hw)
            + c[:, None]
            + h[None, :] * (width * channels)
            + w[None, :] * channels
        )
        mask = row_mask[:, None] & spatial_mask[None, :]

        add_values = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        conv_values = tl.load(convolution_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        spatial_sum = tl.sum(add_values + conv_values, axis=1)
        spatial_mean = spatial_sum * (1.0 / 49.0)

        mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        variance = tl.load(variance_ptr + c, mask=row_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(variance + eps)
        output = (spatial_mean - mean) * invstd * weight + bias
        tl.store(out_ptr + rows, output, mask=row_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _expect_tensor(
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
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    add_97, convolution_56, arg172_1, arg173_1, arg174_1, arg175_1, shape_param = inputs
    add_t = _expect_tensor("add_97", add_97, INPUT_SHAPE, CHANNELS_LAST_INPUT_STRIDE)
    convolution_t = _expect_tensor(
        "convolution_56",
        convolution_56,
        INPUT_SHAPE,
        CHANNELS_LAST_INPUT_STRIDE,
    )
    mean_t = _expect_tensor("arg172_1", arg172_1, PARAM_SHAPE, (1,))
    variance_t = _expect_tensor("arg173_1", arg173_1, PARAM_SHAPE, (1,))
    weight_t = _expect_tensor("arg174_1", arg174_1, PARAM_SHAPE, (1,))
    bias_t = _expect_tensor("arg175_1", arg175_1, PARAM_SHAPE, (1,))
    if _shape_tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    tensors = (add_t, convolution_t, mean_t, variance_t, weight_t, bias_t)
    if any(t.device != add_t.device for t in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return add_t, convolution_t, mean_t, variance_t, weight_t, bias_t


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
        raise RuntimeError("Triton is required for oracle_affine_spatial_mean_channels_last.py")

    add_t, convolution_t, mean_t, variance_t, weight_t, bias_t = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=add_t.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_ROWS"]),)
    _channels_last_affine_spatial_mean_kernel[grid](
        add_t,
        convolution_t,
        mean_t,
        variance_t,
        weight_t,
        bias_t,
        output,
        n_rows=ROWS,
        channels=CHANNELS,
        width=WIDTH,
        hw=HW,
        eps=EPS,
        BLOCK_HW_=BLOCK_HW,
    )
    return output


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate deterministic output values while requiring matching NaN masks."""
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

    dtype_ok = expected.dtype == actual.dtype
    stride_ok = expected.stride() == actual.stride()
    ok = dtype_ok and stride_ok and nan_mask_ok and values_ok
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
