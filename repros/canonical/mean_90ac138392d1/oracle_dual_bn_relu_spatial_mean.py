"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RepVGG dual inference-BatchNorm, branch add, ReLU, and 7x7 spatial mean returned by Repro.forward in one channels-last Triton kernel, whereas Inductor currently lowers the decomposed broadcast normalization, branch add/ReLU, and spatial reduction as generic scheduled work that materializes the post-activation tensor before pooling; Inductor cannot do this today because the scheduler/norm-template path does not sink the channelwise normalization epilogue and sibling branch sum directly into the reduction over the NHWC-contiguous input layout; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse fixed-shape inference BatchNorm producers and ReLU into the spatial mean consumer for channels-last tensors."""
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

N = 128
CHANNELS = 1408
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_N = 2
BLOCK_C = 64
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _dual_bn_relu_spatial_mean_kernel(
        mean0_ptr,
        x0_ptr,
        var0_ptr,
        weight0_ptr,
        bias0_ptr,
        mean1_ptr,
        x1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        out_ptr,
        n_size: tl.constexpr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        n_offsets = tl.program_id(0) * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        c_offsets = tl.program_id(1) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        n_mask = n_offsets < n_size
        c_mask = c_offsets < channels

        mean0 = tl.load(mean0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var0 = tl.load(var0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weight0 = tl.load(weight0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        mean1 = tl.load(mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var1 = tl.load(var1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        scale0 = (1.0 / tl.sqrt(var0 + eps)) * weight0
        scale1 = (1.0 / tl.sqrt(var1 + eps)) * weight1
        acc = tl.zeros((BLOCK_N_, BLOCK_C_), tl.float32)
        mask = n_mask[:, None] & c_mask[None, :]
        base_n = n_offsets[:, None] * channels * spatial + c_offsets[None, :]

        for hw in tl.static_range(0, spatial):
            offsets = base_n + hw * channels
            x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            y0 = (x0 - mean0[None, :]) * scale0[None, :] + bias0[None, :]
            y1 = (x1 - mean1[None, :]) * scale1[None, :] + bias1[None, :]
            summed = y0 + y1
            relu = tl.where((summed > 0.0) | (summed != summed), summed, 0.0)
            acc += relu

        pooled = acc * (1.0 / spatial)
        out_offsets = n_offsets[:, None] * channels + c_offsets[None, :]
        tl.store(out_ptr + out_offsets, pooled, mask=mask)


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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects eleven inputs, got {len(inputs)}")

    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        shape_param,
    ) = inputs

    activation_shape = (N, CHANNELS, HEIGHT, WIDTH)
    channels_last_stride = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
    vector_shape = (CHANNELS,)
    vector_stride = (1,)
    mean0_t = _expect_tensor("arg280_1", mean0, vector_shape, vector_stride)
    x0_t = _expect_tensor("convolution_42", x0, activation_shape, channels_last_stride)
    var0_t = _expect_tensor("arg281_1", var0, vector_shape, vector_stride)
    weight0_t = _expect_tensor("arg282_1", weight0, vector_shape, vector_stride)
    bias0_t = _expect_tensor("arg283_1", bias0, vector_shape, vector_stride)
    mean1_t = _expect_tensor("arg285_1", mean1, vector_shape, vector_stride)
    x1_t = _expect_tensor("convolution_43", x1, activation_shape, channels_last_stride)
    var1_t = _expect_tensor("arg286_1", var1, vector_shape, vector_stride)
    weight1_t = _expect_tensor("arg287_1", weight1, vector_shape, vector_stride)
    bias1_t = _expect_tensor("arg288_1", bias1, vector_shape, vector_stride)

    if tuple(shape_param) != (N, CHANNELS):
        raise ValueError(f"unexpected reshape parameter: {shape_param!r}")

    device = x0_t.device
    tensor_inputs = (
        mean0_t,
        var0_t,
        weight0_t,
        bias0_t,
        mean1_t,
        x1_t,
        var1_t,
        weight1_t,
        bias1_t,
    )
    if any(t.device != device for t in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        mean0_t,
        x0_t,
        var0_t,
        weight0_t,
        bias0_t,
        mean1_t,
        x1_t,
        var1_t,
        weight1_t,
        bias1_t,
    )


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation and return the final [128, 1408] tensor."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dual_bn_relu_spatial_mean.py")

    (
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
    ) = _validate_inputs(inputs)
    out = torch.empty_strided(
        (N, CHANNELS),
        (CHANNELS, 1),
        device=x0.device,
        dtype=torch.float32,
    )

    _dual_bn_relu_spatial_mean_kernel[(triton.cdiv(N, BLOCK_N), triton.cdiv(CHANNELS, BLOCK_C))](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        out,
        n_size=N,
        channels=CHANNELS,
        spatial=HW,
        eps=EPS,
        BLOCK_N_=BLOCK_N,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    return out


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_oracle_equal_nan(
    oracle_fn,
    instance,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool = True,
) -> bool:
    if skip_stochastic and has_stochastic_ops(REPRO_PATH):
        return check_oracle(
            oracle_fn,
            instance,
            inputs,
            atol=atol,
            rtol=rtol,
            skip_stochastic=True,
        )

    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_fn(inputs)
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
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} eager={list(expected.shape)}")
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(f"  output {i}: SCOPE_MISMATCH dtype oracle={actual.dtype} eager={expected.dtype}")
            all_pass = False
            continue
        if expected.stride() != actual.stride():
            print(f"  output {i}: SCOPE_MISMATCH stride oracle={actual.stride()} eager={expected.stride()}")
            all_pass = False
            continue

        if expected.is_floating_point():
            expected_f32 = expected.float()
            actual_f32 = actual.float()
            expected_nan = torch.isnan(expected_f32)
            actual_nan = torch.isnan(actual_f32)
            finite_mask = ~(expected_nan | actual_nan)
            nan_match = torch.equal(expected_nan, actual_nan)
            if finite_mask.any():
                max_diff = (expected_f32[finite_mask] - actual_f32[finite_mask]).abs().max().item()
            else:
                max_diff = 0.0
            ok = nan_match and torch.allclose(
                expected_f32,
                actual_f32,
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            )
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} "
                f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
            )
        else:
            ok = torch.equal(expected, actual)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
        all_pass = all_pass and bool(ok)

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
