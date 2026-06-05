"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet BN-inference affine over f32 `[128,1280,7,7]`, evaluates SiLU as `x/(exp(-x)+1)`, preserves the eager NaN mask from invalid variance inputs, and writes the final contiguous f32 `[128,1280]` spatial-mean view from one batch/channel-tiled Triton reduction that reuses per-channel parameters across the 7x7 tile, whereas Inductor emits a generic fused reduction schedule for the same scope with materially higher constant overhead; Inductor cannot do this today because norm-template canonicalization does not provide a guarded BN-affine plus activation plus small-spatial-mean schedule specialized for this NCHW/channels-last family; the fix is SCHEDULER_FUSION: add a benchmark-gated scheduler template that fuses BN-affine, SiLU, and small spatial pooling with per-channel parameter reuse and the exact output layout."""
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


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def check_oracle(
    oracle_forward,
    instance,
    inputs,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    skip_stochastic: bool = True,
) -> bool:
    """NaN-aware full-scope check for this deterministic BN-inference repro."""
    del skip_stochastic
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (eager_tensor, oracle_tensor) in enumerate(zip(eager_list, oracle_list)):
        shape_ok = eager_tensor.shape == oracle_tensor.shape
        dtype_ok = eager_tensor.dtype == oracle_tensor.dtype
        stride_ok = eager_tensor.stride() == oracle_tensor.stride()
        if not shape_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(oracle_tensor.shape)} "
                f"eager={list(eager_tensor.shape)}"
            )
            all_pass = False
            continue
        if not dtype_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={oracle_tensor.dtype} "
                f"eager={eager_tensor.dtype}"
            )
            all_pass = False
            continue
        if not stride_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={oracle_tensor.stride()} "
                f"eager={eager_tensor.stride()}"
            )
            all_pass = False
            continue
        if not eager_tensor.is_floating_point():
            ok = torch.equal(eager_tensor, oracle_tensor)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={eager_tensor.dtype})")
            all_pass = all_pass and bool(ok)
            continue

        eager_f32 = eager_tensor.float()
        oracle_f32 = oracle_tensor.float()
        eager_nan = torch.isnan(eager_f32)
        oracle_nan = torch.isnan(oracle_f32)
        nan_ok = torch.equal(eager_nan, oracle_nan)
        finite = ~eager_nan & ~oracle_nan
        if finite.any():
            max_diff = (eager_f32[finite] - oracle_f32[finite]).abs().max().item()
            values_ok = torch.allclose(eager_f32[finite], oracle_f32[finite], atol=atol, rtol=rtol)
        else:
            max_diff = 0.0
            values_ok = True
        ok = nan_ok and values_ok
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
            f"stride={eager_tensor.stride()} nan_count={int(eager_nan.sum().item())} "
            f"max_finite_diff={max_diff:.2e})"
        )
        all_pass = all_pass and bool(ok)

    return all_pass


# --- Oracle kernel(s) ---

BATCH = 128
CHANNELS = 1280
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ROWS = BATCH * CHANNELS
EPS = 1.0e-3
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
CONTIGUOUS_X_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
CHANNELS_LAST_X_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_B": 8, "BLOCK_C": 2, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_B": 8, "BLOCK_C": 4, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_B": 16, "BLOCK_C": 2, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_B": 16, "BLOCK_C": 4, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
        ],
        key=["x_stride_n", "x_stride_c", "x_stride_h", "x_stride_w"],
    )
    @triton.jit
    def _bn_silu_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch_offsets = tl.program_id(0) * BLOCK_B + tl.arange(0, BLOCK_B)
        channel_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)
        h_offsets = hw_offsets // 7
        w_offsets = hw_offsets - h_offsets * 7

        valid_b = batch_offsets < 128
        valid_c = channel_offsets < 1280
        valid_hw = hw_offsets < 49
        valid_bc = valid_b[:, None] & valid_c[None, :]

        x_offsets = (
            batch_offsets[:, None, None] * x_stride_n
            + channel_offsets[None, :, None] * x_stride_c
            + h_offsets[None, None, :] * x_stride_h
            + w_offsets[None, None, :] * x_stride_w
        )
        x = tl.load(
            x_ptr + x_offsets,
            mask=valid_bc[:, :, None] & valid_hw[None, None, :],
            other=0.0,
        ).to(tl.float32)

        mean = tl.load(mean_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel_offsets, mask=valid_c, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=valid_c, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + eps)
        affine = ((x - mean[None, :, None]) * invstd[None, :, None]) * weight[None, :, None]
        affine = affine + bias[None, :, None]
        silu = affine / (tl.exp(-affine) + 1.0)
        reduced = tl.sum(tl.where(valid_hw[None, None, :], silu, 0.0), axis=2) * (1.0 / 49.0)

        out_offsets = batch_offsets[:, None] * 1280 + channel_offsets[None, :]
        tl.store(out_ptr + out_offsets, reduced, mask=valid_bc)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    allowed_strides: tuple[tuple[int, ...], ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) not in allowed_strides:
        raise ValueError(
            f"{name} has stride {tuple(value.stride())}, expected one of {allowed_strides}"
        )
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(int(dim) for dim in value)
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")
    return shape


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, view_shape = inputs
    mean_t = _require_f32_tensor("arg306_1", mean, (CHANNELS,), ((1,),))
    x_t = _require_f32_tensor(
        "convolution_80",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CONTIGUOUS_X_STRIDE, CHANNELS_LAST_X_STRIDE),
    )
    var_t = _require_f32_tensor("arg307_1", var, (CHANNELS,), ((1,),))
    weight_t = _require_f32_tensor("arg308_1", weight, (CHANNELS,), ((1,),))
    bias_t = _require_f32_tensor("arg309_1", bias, (CHANNELS,), ((1,),))
    _require_shape("_shape_param_0", view_shape, OUTPUT_SHAPE)

    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean_t, x_t, var_t, weight_t, bias_t


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
        raise RuntimeError("Triton is required for oracle_bn_silu_spatial_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (
        triton.cdiv(BATCH, meta["BLOCK_B"]),
        triton.cdiv(CHANNELS, meta["BLOCK_C"]),
    )
    _bn_silu_spatial_mean_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        eps=EPS,
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
