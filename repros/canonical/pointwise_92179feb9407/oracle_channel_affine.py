"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete layernorm-ish pointwise scope in one coalesced Triton kernel and folds the channel-only affine into `scale = rsqrt(var + eps) * weight` and `offset = bias - mean * scale`, whereas Inductor lowers the captured decomposed add/sub/sqrt/reciprocal/mul/mul/add graph as a generic pointwise expression over every activation element; Inductor cannot do this today because its algebraic simplification does not canonicalize broadcasted normalization parameters into per-channel scale/offset coefficients before pointwise scheduling; the fix is ALGEBRAIC_ELIMINATION: add a broadcast-aware affine-normalization folding pass that preserves NaN masks and emits the final activation update as one FMA-shaped pointwise kernel."""
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

EPS = 1.0e-5
LAYOUT_NCHW = 0
LAYOUT_CHANNELS_LAST = 1
CLASSIFICATION = "ALGEBRAIC_ELIMINATION"


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


def _contiguous_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    n, c, h, w = shape
    return (c * h * w, h * w, w, 1)


def _channels_last_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    n, c, h, w = shape
    return (c * h * w, 1, w * c, c)


def _dense_layout(shape: tuple[int, int, int, int], stride: tuple[int, ...]) -> int:
    if stride == _contiguous_stride(shape):
        return LAYOUT_NCHW
    if stride == _channels_last_stride(shape):
        return LAYOUT_CHANNELS_LAST
    raise ValueError(f"unsupported dense 4D input stride {stride} for shape {shape}")


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...] | tuple[Any, ...]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    add_94, convolution_54, mean, variance, weight, bias = inputs
    tensor_inputs = (add_94, convolution_54, mean, variance, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("all repro inputs must be tensors")
    if not all(value.device.type == "cuda" for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.dtype is torch.float32 for value in tensor_inputs):
        dtypes = [getattr(value, "dtype", None) for value in tensor_inputs]
        raise TypeError(f"all tensor inputs must be torch.float32, got {dtypes}")

    shape = tuple(int(dim) for dim in add_94.shape)
    if len(shape) != 4:
        raise ValueError(f"activation inputs must be rank 4, got shape {shape}")
    if tuple(convolution_54.shape) != shape:
        raise ValueError(f"input 1 shape {tuple(convolution_54.shape)} != {shape}")

    channels = shape[1]
    for index, value in enumerate((mean, variance, weight, bias), start=2):
        if tuple(value.shape) != (channels,):
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {(channels,)}")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    layout = _dense_layout(shape, tuple(add_94.stride()))
    if tuple(convolution_54.stride()) != tuple(add_94.stride()):
        raise ValueError(
            f"activation input strides must match, got {add_94.stride()} and {convolution_54.stride()}"
        )

    return add_94, convolution_54, mean, variance, weight, bias, layout


def _block_size_for_numel(numel: int) -> int:
    if numel % 1024 == 0:
        return 1024
    if numel % 256 == 0:
        return 256
    raise ValueError(f"unsupported numel {numel}: expected a dense shape divisible by 256")


if triton is not None:

    @triton.jit
    def _channel_affine_kernel(
        add_ptr,
        conv_ptr,
        mean_ptr,
        variance_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        spatial: tl.constexpr,
        layout: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        if layout == 1:
            channel = offsets % channels
        else:
            channel = (offsets // spatial) % channels

        add_value = tl.load(add_ptr + offsets)
        conv_value = tl.load(conv_ptr + offsets)
        mean = tl.load(mean_ptr + channel, eviction_policy="evict_last")
        variance = tl.load(variance_ptr + channel, eviction_policy="evict_last")
        weight = tl.load(weight_ptr + channel, eviction_policy="evict_last")
        bias = tl.load(bias_ptr + channel, eviction_policy="evict_last")

        scale = (1.0 / tl.sqrt_rn(variance + 1.0e-5)) * weight
        offset = bias - mean * scale
        out = (add_value + conv_value) * scale + offset
        tl.store(out_ptr + offsets, out)


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    """Run the full Repro.forward computation with the same output layout."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_channel_affine.py")

    add_94, convolution_54, mean, variance, weight, bias, layout = _validate_inputs(inputs)
    shape = tuple(int(dim) for dim in add_94.shape)
    _, channels, height, width = shape
    spatial = height * width
    total = add_94.numel()
    output = torch.empty_strided(
        shape,
        tuple(int(stride) for stride in add_94.stride()),
        device=add_94.device,
        dtype=add_94.dtype,
    )
    block_size = _block_size_for_numel(total)
    grid = (triton.cdiv(total, block_size),)
    _channel_affine_kernel[grid](
        add_94,
        convolution_54,
        mean,
        variance,
        weight,
        bias,
        output,
        total=total,
        channels=channels,
        spatial=spatial,
        layout=layout,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=4,
    )
    return output


def _normalize_outputs(out: object) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_oracle_nan_aware(
    oracle_forward_fn,
    instance: torch.nn.Module,
    inputs: tuple[Any, ...] | list[Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Check values and strict tensor layout while treating deterministic NaNs as equal."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward_fn(inputs)
        if any(value.is_cuda for value in _normalize_outputs(actual)):
            torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(actual_list) != len(expected_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    all_pass = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.dtype == expected_tensor.dtype
            and tuple(actual_tensor.stride()) == tuple(expected_tensor.stride())
            and actual_tensor.storage_offset() == expected_tensor.storage_offset()
        )
        if not layout_ok:
            print(
                f"  output {index}: SCOPE_MISMATCH layout "
                f"expected_shape={list(expected_tensor.shape)} actual_shape={list(actual_tensor.shape)} "
                f"expected_dtype={expected_tensor.dtype} actual_dtype={actual_tensor.dtype} "
                f"expected_stride={tuple(expected_tensor.stride())} actual_stride={tuple(actual_tensor.stride())} "
                f"expected_offset={expected_tensor.storage_offset()} actual_offset={actual_tensor.storage_offset()}"
            )
            all_pass = False
            continue

        if not expected_tensor.is_floating_point():
            values_ok = torch.equal(expected_tensor, actual_tensor)
            print(f"  output {index}: {'PASS' if values_ok else 'FAIL'} (exact, dtype={expected_tensor.dtype})")
            all_pass = values_ok and all_pass
            continue

        expected_f32 = expected_tensor.float()
        actual_f32 = actual_tensor.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_ok = torch.equal(expected_nan, actual_nan)
        finite = ~(expected_nan | actual_nan)
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
            finite_ok = torch.allclose(
                expected_f32[finite],
                actual_f32[finite],
                atol=atol,
                rtol=rtol,
            )
        else:
            max_diff = 0.0
            finite_ok = True

        ok = bool(nan_ok and finite_ok)
        print(
            f"  output {index}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} "
            f"stride={tuple(expected_tensor.stride())} finite_max_diff={max_diff:.2e} "
            f"nan_count={int(expected_nan.sum().item())})"
        )
        if not nan_ok:
            print(
                f"  output {index}: NaN mask mismatch "
                f"(eager={int(expected_nan.sum().item())}, oracle={int(actual_nan.sum().item())})"
            )
        all_pass = ok and all_pass

    return all_pass


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
        ok = _check_oracle_nan_aware(
            oracle_forward,
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


if __name__ == "__main__":
    main()
