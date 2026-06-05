"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes grouped layernorm over each `[2,256]` group and immediately applies affine, ReLU, and padded 3x3 stride-2 low-memory maxpool-with-offsets, producing both f32 pooled values and int8 offsets directly from one Triton group-row kernel, whereas Inductor currently emits a generic grouped norm reduction/pointwise kernel that materializes the full `[64,64,16,16]` ReLU activation before a separate low-memory maxpool-with-offsets kernel; Inductor cannot do this today because scheduler fusion cannot sink a reduction-backed normalized activation producer into a multi-output pooling stencil while preserving low-memory offset semantics; the fix is SCHEDULER_FUSION: add guarded producer fusion from grouped-norm affine/ReLU into low-memory maxpool-with-offsets and emit values plus offsets from the same generated loop nest."""
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


BATCH = 64
CHANNELS = 64
GROUPS = 32
CHANNELS_PER_GROUP = CHANNELS // GROUPS
HEIGHT = 16
WIDTH = 16
OUT_HEIGHT = 8
OUT_WIDTH = 8
GROUP_ELEMS = CHANNELS_PER_GROUP * HEIGHT * WIDTH
POOL_ELEMS_PER_GROUP = CHANNELS_PER_GROUP * OUT_HEIGHT * OUT_WIDTH
EPS = 1.0e-5

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
GROUP_SHAPE = (BATCH, GROUPS, CHANNELS_PER_GROUP, HEIGHT * WIDTH)
OUTPUT_SHAPE = (BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
OUTPUT_STRIDE = (CHANNELS * OUT_HEIGHT * OUT_WIDTH, OUT_HEIGHT * OUT_WIDTH, OUT_WIDTH, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where(x <= 0.0, 0.0, x)


    @triton.jit
    def _groupnorm_relu_maxpool_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        offsets_ptr,
        BLOCK_K: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        group_row = tl.program_id(0)
        batch = group_row // 32
        group = group_row - batch * 32

        elems = tl.arange(0, BLOCK_K)
        x_offsets = group_row * 512 + elems
        x = tl.load(x_ptr + x_offsets).to(tl.float32)

        mean = tl.sum(x, axis=0) * (1.0 / 512.0)
        sq_mean = tl.sum(x * x, axis=0) * (1.0 / 512.0)
        var = tl.maximum(sq_mean - mean * mean, 0.0)
        inv_std = tl.rsqrt(var + 1.0e-5)

        out_linear = tl.arange(0, BLOCK_OUT)
        channel_in_group = out_linear // 64
        pool_linear = out_linear - channel_in_group * 64
        out_h = pool_linear // 8
        out_w = pool_linear - out_h * 8
        channel = group * 2 + channel_in_group

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        input_base = group_row * 512 + channel_in_group * 256

        best = tl.full((BLOCK_OUT,), -float("inf"), tl.float32)
        best_offset = tl.zeros((BLOCK_OUT,), tl.int32)

        for kh in tl.static_range(0, 3):
            input_h = out_h * 2 + kh - 1
            valid_h = (input_h >= 0) & (input_h < 16)
            for kw in tl.static_range(0, 3):
                input_w = out_w * 2 + kw - 1
                valid = valid_h & (input_w >= 0) & (input_w < 16)
                pool_input = tl.load(
                    x_ptr + input_base + input_h * 16 + input_w,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                value = _relu_preserve_nan((pool_input - mean) * inv_std * weight + bias)
                better = valid & ((value > best) | (value != value))
                best = tl.where(better, value, best)
                best_offset = tl.where(better, kh * 3 + kw, best_offset)

        out_offsets = (batch * 64 + channel) * 64 + pool_linear
        tl.store(out_ptr + out_offsets, best)
        tl.store(offsets_ptr + out_offsets, best_offset.to(tl.int8))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x, weight, bias, group_shape, output_shape = inputs
    tensor_inputs = (x, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise ValueError("oracle_groupnorm_maxpool.py expects CUDA tensors")
    if not all(value.dtype is torch.float32 for value in tensor_inputs):
        dtypes = ", ".join(str(value.dtype) for value in tensor_inputs)
        raise ValueError(f"expected all tensor inputs to be f32, got {dtypes}")
    if not all(value.is_contiguous() for value in tensor_inputs):
        strides = ", ".join(str(value.stride()) for value in tensor_inputs)
        raise ValueError(f"expected contiguous tensors, got strides {strides}")

    if tuple(x.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected input shape {tuple(x.shape)}; expected {INPUT_SHAPE}")
    if tuple(weight.shape) != (CHANNELS,) or tuple(bias.shape) != (CHANNELS,):
        raise ValueError(
            f"weight and bias must have shape ({CHANNELS},), "
            f"got {tuple(weight.shape)} and {tuple(bias.shape)}"
        )
    if tuple(group_shape) != GROUP_SHAPE:
        raise ValueError(f"unexpected group view shape {group_shape!r}; expected {GROUP_SHAPE}")
    if tuple(output_shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected output view shape {output_shape!r}; expected {INPUT_SHAPE}")

    return x, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full grouped layernorm, affine, ReLU, and low-memory maxpool scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_groupnorm_maxpool.py")

    x, weight, bias = _validate_inputs(inputs)
    values = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    offsets = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.int8,
    )

    _groupnorm_relu_maxpool_kernel[(BATCH * GROUPS,)](
        x,
        weight,
        bias,
        values,
        offsets,
        BLOCK_K=GROUP_ELEMS,
        BLOCK_OUT=POOL_ELEMS_PER_GROUP,
        num_warps=8,
        num_stages=3,
    )
    return values, offsets


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
