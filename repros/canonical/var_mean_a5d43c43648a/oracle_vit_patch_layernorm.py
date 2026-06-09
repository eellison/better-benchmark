"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ViT patch/class-token LayerNorm scope in one fixed-hidden Triton row kernel, including class-token expansion, convolution patch view/permute gather layout, positional add, fp32 var_mean over hidden dim 384 with eps=1e-6, fp16 affine inputs, fp16 output cast, and final [6304, 384] view, whereas Inductor lowers the decomposed expand/view/permute/cat/add/var_mean/affine graph through generic layout and normalization code; Inductor cannot materially improve this today because the compiled full-scope kernel already measures within the oracle floor band for the required patch gather, row reduction, and output-store traffic; the fix is BANDWIDTH_BOUND: keep this as an at-floor ViT patch LayerNorm case unless a broader launch or memory-traffic improvement appears."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    oracle_impl,
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


ROWS = 32 * 197
BATCH = 32
TOKENS = 197
PATCHES = 196
CHANNELS = 384
HEIGHT = 14
WIDTH = 14
EPS = 1.0e-6

if triton is not None:

    @triton.jit
    def _vit_patch_layernorm_kernel(
        class_token_ptr,
        convolution_ptr,
        position_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        patches: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden

        batch = row // tokens
        token = row - batch * tokens
        is_class_token = token == 0
        patch = token - 1

        class_values = tl.load(
            class_token_ptr + cols,
            mask=mask & is_class_token,
            other=0.0,
        )
        patch_values = tl.load(
            convolution_ptr + batch * hidden * patches + cols * patches + patch,
            mask=mask & (token != 0),
            other=0.0,
        )
        position_values = tl.load(
            position_ptr + token * hidden + cols,
            mask=mask,
            other=0.0,
        )

        x = (class_values + patch_values + position_values).to(tl.float16).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        sum_x2 = tl.sum(x_for_reduce * x_for_reduce, axis=0)
        variance = sum_x2 / hidden - mean * mean
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        centered = x - mean
        y = centered * invstd * weight + bias
        tl.store(output_ptr + row * hidden + cols, y, mask=mask)


def _shape_tuple(value):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        class_token,
        convolution,
        position,
        weight,
        bias,
        expand_shape,
        conv_view_shape,
        output_shape,
    ) = inputs

    tensor_inputs = (class_token, convolution, position, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (
        (1, 1, CHANNELS),
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (1, TOKENS, CHANNELS),
        (CHANNELS,),
        (CHANNELS,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    expand_shape_tuple = _shape_tuple(expand_shape)
    conv_view_shape_tuple = _shape_tuple(conv_view_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if expand_shape_tuple not in ((BATCH, 1, CHANNELS), (BATCH, -1, -1)):
        raise ValueError(f"unexpected expand shape parameter: {expand_shape!r}")
    if conv_view_shape_tuple != (BATCH, CHANNELS, PATCHES):
        raise ValueError(f"unexpected convolution view shape parameter: {conv_view_shape!r}")
    if output_shape_tuple != (ROWS, CHANNELS):
        raise ValueError(f"unexpected output shape parameter: {output_shape!r}")

    return class_token, convolution, position, weight, bias, output_shape_tuple


@oracle_impl(hardware="H100", shapes="(T([1, 1, 384], f16), T([32, 384, 14, 14], f16), T([1, 197, 384], f16), T([384], f16), T([384], f16), S([32, -1, -1]), S([32, 384, 196]), S([6304, 384]))")
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
        raise RuntimeError("Triton is required for oracle_vit_patch_layernorm.py")

    class_token, convolution, position, weight, bias, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        (CHANNELS, 1),
        device=convolution.device,
        dtype=torch.float16,
    )
    _vit_patch_layernorm_kernel[(ROWS,)](
        class_token,
        convolution,
        position,
        weight,
        bias,
        output,
        hidden=CHANNELS,
        tokens=TOKENS,
        patches=PATCHES,
        eps=EPS,
        BLOCK_H=512,
        num_warps=1,
        num_stages=3,
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
