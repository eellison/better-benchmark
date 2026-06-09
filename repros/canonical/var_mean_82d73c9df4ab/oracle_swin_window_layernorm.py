"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete hidden-size-1024 population var_mean LayerNorm affine and singleton Swin 7x7 window-partition reshape/permute/flatten aliases into one contiguous row kernel, whereas Inductor currently schedules the captured var_mean/affine/view graph through a generic normalization path with singleton-window alias handling outside the specialized row epilogue; Inductor cannot do this today because its scheduler does not canonicalize fixed size-one window-grid reshape/permute aliases early enough to make the flattened store part of the normalization schedule; the fix is SCHEDULER_FUSION: extend the normalization scheduler to recognize singleton Swin window-partition aliases and emit the affine LayerNorm directly in the flattened output layout."""
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


BATCH = 128
HEIGHT = 7
WIDTH = 7
SPATIAL = HEIGHT * WIDTH
HIDDEN = 1024
ROWS = BATCH * SPATIAL
EPS = 1.0e-5
DTYPE = torch.float32

INPUT_SHAPE = (ROWS, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
VIEW_4D_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
WINDOW_VIEW_SHAPE = (BATCH, 1, HEIGHT, 1, WIDTH, HIDDEN)
WINDOW_4D_SHAPE = (-1, HEIGHT, WIDTH, HIDDEN)
WINDOW_3D_SHAPE = (-1, SPATIAL, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["total_rows", "hidden"],
    )
    @triton.jit
    def _swin_window_layernorm_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, BLOCK_H)
        offsets = rows[:, None] * hidden + cols[None, :]

        x = tl.load(input_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
        mean = (tl.sum(x, axis=1) / hidden)[:, None]
        centered = x - mean
        variance = (tl.sum(centered * centered, axis=1) / hidden)[:, None]
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        y = centered * invstd * weight[None, :] + bias[None, :]

        tl.store(output_ptr + offsets, y)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != DTYPE:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {DTYPE}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        mm_2,
        primals_334,
        primals_335,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
    ) = inputs

    input_tensor = _require_f32_tensor("mm_2", mm_2, INPUT_SHAPE, OUTPUT_STRIDE)
    weight = _require_f32_tensor("primals_334", primals_334, AFFINE_SHAPE, (1,))
    bias = _require_f32_tensor("primals_335", primals_335, AFFINE_SHAPE, (1,))

    expected_shapes = (
        VIEW_4D_SHAPE,
        WINDOW_VIEW_SHAPE,
        WINDOW_4D_SHAPE,
        WINDOW_3D_SHAPE,
        OUTPUT_SHAPE,
    )
    actual_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2, shape3, shape4))
    for index, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
        if actual != expected:
            raise ValueError(f"unexpected _shape_param_{index}: {actual!r}, expected {expected!r}")

    device = input_tensor.device
    for name, tensor in (("primals_334", weight), ("primals_335", bias)):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return input_tensor, weight, bias, shape0, shape1, shape2, shape3, shape4


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    input_tensor, weight, bias, shape0, shape1, shape2, shape3, shape4 = _validate_inputs(inputs)
    x = torch.ops.aten.reshape.default(input_tensor, shape0)
    variance, mean = torch.ops.aten.var_mean.correction(
        x,
        [3],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    x = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(x, mean), invstd)
    x = torch.ops.aten.mul.Tensor(x, weight)
    x = torch.ops.aten.add.Tensor(x, bias)
    x = torch.ops.aten.reshape.default(x, shape1)
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.reshape.default(x, shape2)
    x = torch.ops.aten.reshape.default(x, shape3)
    return torch.ops.aten.reshape.default(x, shape4)


@oracle_impl(hardware="H100", shapes="(T([6272, 1024], f32), T([1024], f32), T([1024], f32), S([128, 7, 7, 1024]), S([128, 1, 7, 1, 7, 1024]), S([-1, 7, 7, 1024]), S([-1, 49, 1024]), S([6272, 1024]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward LayerNorm plus singleton-window alias scope.

    SCOPE INVARIANT: accepts the same eight inputs as Repro.forward() and
    returns the same single contiguous float32 [6272, 1024] tensor. The window
    partition has a 1x1 window grid for this shape, so the Triton kernel writes
    the equivalent flattened row-major layout directly.
    """
    input_tensor, weight, bias, _shape0, _shape1, _shape2, _shape3, _shape4 = _validate_inputs(inputs)
    if triton is None or not input_tensor.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=input_tensor.device,
        dtype=DTYPE,
    )
    block_h = triton.next_power_of_2(HIDDEN)
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_window_layernorm_kernel[grid](
        input_tensor,
        weight,
        bias,
        output,
        total_rows=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=block_h,
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
