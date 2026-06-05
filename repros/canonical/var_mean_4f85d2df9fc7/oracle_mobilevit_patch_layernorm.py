"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MobileViT patch-layout LayerNorm scope in one Triton row kernel, including the NCHW-contiguous clone semantics over the channels-last `[128,240,8,8]` input, both reshape/transpose/clone layout transforms into `[512,16,240]`, `var_mean(correction=0, keepdim=True)` over hidden size 240, affine scale/bias, final contiguous `[8192,240]` view, and sibling `rsqrt(var + 1e-5) / 240` output, whereas coordinate-descent Inductor already measures within the same full-scope floor band for this fixed shape; Inductor cannot materially improve this repro through a local oracle-style rewrite because the remaining work is dominated by mandatory activation/affine reads, one small row reduction, output and invstd stores, and launch overhead rather than avoidable sibling work; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope normalization/layout case unless broader normalization-template or launch-overhead changes move the baseline."""
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

BATCH = 128
CHANNELS = 240
HEIGHT = 8
WIDTH = 8
PATCH_H = 2
PATCH_W = 2
PATCH_AREA = PATCH_H * PATCH_W
NUM_PATCH_H = HEIGHT // PATCH_H
NUM_PATCH_W = WIDTH // PATCH_W
NUM_PATCHES = NUM_PATCH_H * NUM_PATCH_W
ROWS_512 = BATCH * PATCH_AREA
ROWS = ROWS_512 * NUM_PATCHES
EPS = 1.0e-5
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HEIGHT * WIDTH, 1, WIDTH * CHANNELS, CHANNELS)
WEIGHT_SHAPE = (CHANNELS,)
OUTPUT_SHAPE = (ROWS, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
DIV_SHAPE = (ROWS_512, NUM_PATCHES, 1)
DIV_STRIDE = (NUM_PATCHES, 1, 1)
SHAPE_PARAM_0 = (BATCH * CHANNELS * NUM_PATCH_H, PATCH_H, NUM_PATCH_W, PATCH_W)
SHAPE_PARAM_1 = (BATCH, CHANNELS, NUM_PATCHES, PATCH_AREA)
SHAPE_PARAM_2 = (ROWS_512, NUM_PATCHES, CHANNELS)
SHAPE_PARAM_3 = OUTPUT_SHAPE
BLOCK_C = 256
CLASSIFICATION = "BANDWIDTH_BOUND"


if triton is not None:

    @triton.jit
    def _mobilevit_patch_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        div_ptr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        block_c: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_c)
        mask = cols < channels

        row512 = row // 16
        patch = row - row512 * 16
        batch = row512 // 4
        patch_area = row512 - batch * 4

        local_h = patch_area // 2
        local_w = patch_area - local_h * 2
        patch_h = patch // 4
        patch_w = patch - patch_h * 4
        h = patch_h * 2 + local_h
        w = patch_w * 2 + local_w

        input_offsets = batch * 15360 + h * 1920 + w * 240 + cols
        x = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)

        x_masked = tl.where(mask, x, 0.0)
        mean = tl.sum(x_masked, axis=0) / channels
        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        centered = x - mean
        variance = sum_sq / channels - mean * mean
        invstd = tl.rsqrt(tl.maximum(variance, 0.0) + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        output_offsets = row * channels + cols
        tl.store(out_ptr + output_offsets, y, mask=mask)
        tl.store(div_ptr + row, invstd / channels)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    convolution_31, primals_255, primals_256, shape0, shape1, shape2, shape3 = inputs
    x = _expect_tensor("convolution_31", convolution_31, INPUT_SHAPE, INPUT_STRIDE)
    weight = _expect_tensor("primals_255", primals_255, WEIGHT_SHAPE, (1,))
    bias = _expect_tensor("primals_256", primals_256, WEIGHT_SHAPE, (1,))
    if x.device != weight.device or x.device != bias.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    expected_shapes = (SHAPE_PARAM_0, SHAPE_PARAM_1, SHAPE_PARAM_2, SHAPE_PARAM_3)
    actual_shapes = (_shape_tuple(shape0), _shape_tuple(shape1), _shape_tuple(shape2), _shape_tuple(shape3))
    for index, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
        if actual != expected:
            raise ValueError(f"unexpected shape parameter {index}: {actual}, expected {expected}")

    return x, weight, bias


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
        raise RuntimeError("Triton is required for oracle_mobilevit_patch_layernorm.py")

    x, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=x.dtype)
    div = torch.empty_strided(DIV_SHAPE, DIV_STRIDE, device=x.device, dtype=x.dtype)

    _mobilevit_patch_layernorm_kernel[(ROWS,)](
        x,
        weight,
        bias,
        output,
        div,
        channels=CHANNELS,
        eps=EPS,
        block_c=BLOCK_C,
        num_warps=1,
        num_stages=2,
    )
    return output, div


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
