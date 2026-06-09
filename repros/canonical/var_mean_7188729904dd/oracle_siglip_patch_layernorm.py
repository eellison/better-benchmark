"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full SigLIP patch positional-add LayerNorm scope in a fixed-shape Triton normalization template, including the channels-contiguous convolution view/permute gather, positional add, fp32 population var_mean over hidden size 768, eps=1e-6 affine epilogue, and final contiguous [32768, 768] view, whereas Inductor already fuses the decomposed view/permute/add/var_mean/affine graph into a generic persistent reduction; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, recompute fusion, or a new narrow pattern because the remaining cost is dominated by compulsory input/position/affine reads, the row reduction, and output writes; the fix is BANDWIDTH_BOUND: record this as an at-floor patch LayerNorm case unless broader normalization-template or memory-traffic work moves both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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
TOKENS = 256
CHANNELS = 768
PATCH_H = 16
PATCH_W = 16
ROWS = BATCH * TOKENS
OUTPUT_SHAPE = (ROWS, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
EPS = 1.0e-6

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 1}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_H": 1024, "ROW_BLOCK": 8}, num_warps=8, num_stages=3),
        ],
        key=["total_rows"],
    )
    @triton.jit
    def _siglip_patch_layernorm_kernel(
        convolution_ptr,
        position_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
        cols = tl.arange(0, BLOCK_H)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        token = rows % tokens

        patch_values = tl.load(
            convolution_ptr + rows * hidden + cols,
            mask=mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        position_values = tl.load(
            position_ptr + token * hidden + cols,
            mask=mask,
            eviction_policy="evict_last",
            other=0.0,
        )

        x = patch_values + position_values
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None].to(tl.float32) / hidden

        centered_for_var = x - mean
        variance = (
            tl.sum(tl.where(mask, centered_for_var * centered_for_var, 0.0), axis=1)[:, None]
            .to(tl.float32)
            / hidden
        )
        centered = x - mean
        invstd = libdevice.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            eviction_policy="evict_last",
            other=0.0,
        )
        y = (centered * invstd) * weight + bias
        tl.store(output_ptr + rows * hidden + cols, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"shape parameter {value!r} is not iterable") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    convolution, position, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (convolution, position, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (
        (BATCH, CHANNELS, PATCH_H, PATCH_W),
        (1, TOKENS, CHANNELS),
        (CHANNELS,),
        (CHANNELS,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")

    device = convolution.device
    if not all(value.device == device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    if convolution.stride() != (CHANNELS * TOKENS, 1, CHANNELS * PATCH_W, CHANNELS):
        raise ValueError(f"unexpected convolution stride: {convolution.stride()}")
    for index, value in enumerate((position, weight, bias), start=1):
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if _shape_tuple(shape0) != (BATCH, CHANNELS, TOKENS):
        raise ValueError(f"unexpected convolution view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape1!r}")

    return convolution, position, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation with a Triton LayerNorm kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single contiguous float32[32768, 768] output tensor. The convolution
    view, permute, positional add, var_mean, affine, and final view semantics
    are represented by direct row/source indexing inside the kernel.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_siglip_patch_layernorm.py")

    convolution, position, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=convolution.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _siglip_patch_layernorm_kernel[grid](
        convolution,
        position,
        weight,
        bias,
        output,
        total_rows=ROWS,
        hidden=CHANNELS,
        tokens=TOKENS,
        eps=EPS,
    )
    return output


def _check_layout_and_alias(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()

    layout_ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.dtype == expected.dtype
        and actual.storage_offset() == expected.storage_offset()
    )
    alias_ok = True
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()}, "
        f"expected_storage_offset={expected.storage_offset()}, "
        f"oracle_storage_offset={actual.storage_offset()})"
    )
    print("  output aliases: PASS (single output)")
    return layout_ok and alias_ok


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
        ok = _check_layout_and_alias(instance, inputs) and ok
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
