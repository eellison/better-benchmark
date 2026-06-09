"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete fp32 hidden-size-512 population LayerNorm affine followed by the fixed Swin 7x7 window-partition clone by traversing final window rows and storing contiguously into the clone/view output, whereas Inductor fuses the same ATen scope but schedules source-contiguous rows and emits strided stores into the clone buffer; parent `bench_oracle()` measures the two schedules at the same floor, so the output-contiguous remap is not a profitable local scheduler fix for this repro; the fix is BANDWIDTH_BOUND: record this as an at-floor Swin window LayerNorm case unless broader normalization-template or memory-traffic work moves both paths."""
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


BATCH = 128
HEIGHT = 14
WIDTH = 14
WINDOW = 7
WINDOW_BLOCKS = 2
CHANNELS = 512
ROWS = BATCH * HEIGHT * WIDTH
EPS = 1.0e-5
DTYPE = torch.float32

INPUT_SHAPE = (ROWS, CHANNELS)
AFFINE_SHAPE = (CHANNELS,)
OUTPUT_SHAPE = (ROWS, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
CLONE_SHAPE = (BATCH, WINDOW_BLOCKS, WINDOW_BLOCKS, WINDOW, WINDOW, CHANNELS)
CLONE_STRIDE = (
    HEIGHT * WIDTH * CHANNELS,
    WINDOW * WINDOW_BLOCKS * WINDOW * CHANNELS,
    WINDOW * WINDOW * CHANNELS,
    WINDOW * CHANNELS,
    CHANNELS,
    1,
)
SHAPE_PARAMS = (
    (BATCH, HEIGHT, WIDTH, CHANNELS),
    (BATCH, WINDOW_BLOCKS, WINDOW, WINDOW_BLOCKS, WINDOW, CHANNELS),
    (-1, WINDOW, WINDOW, CHANNELS),
    (-1, WINDOW * WINDOW, CHANNELS),
    OUTPUT_SHAPE,
)


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
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
        ],
        key=["total_rows"],
    )
    @triton.jit
    def _swin_window_layernorm_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        window: tl.constexpr,
        window_blocks: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, BLOCK_C)
        row_mask = out_rows < total_rows
        col_mask = cols < channels
        mask = row_mask[:, None] & col_mask[None, :]

        batch = out_rows // (height * width)
        within_image = out_rows - batch * (height * width)
        window_id = within_image // (window * window)
        position = within_image - window_id * (window * window)
        block_h = window_id // window_blocks
        block_w = window_id - block_h * window_blocks
        inner_h = position // window
        inner_w = position - inner_h * window
        src_h = block_h * window + inner_h
        src_w = block_w * window + inner_w
        src_rows = batch * (height * width) + src_h * width + src_w

        input_offsets = src_rows[:, None] * channels + cols[None, :]
        x = tl.load(
            input_ptr + input_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        x_for_reduce = tl.where(mask, x, 0.0)
        mean_denom = tl.full([ROW_BLOCK], 512, tl.int32).to(tl.float32)
        mean = tl.sum(x_for_reduce, axis=1) / mean_denom

        centered_for_var = x - mean[:, None]
        variance_terms = centered_for_var * centered_for_var
        variance_denom = tl.full([ROW_BLOCK], 512.0, tl.float32)
        variance = tl.sum(tl.where(mask, variance_terms, 0.0), axis=1) / variance_denom
        invstd = libdevice.rsqrt(variance + tl.full([ROW_BLOCK], eps, tl.float32))

        centered = x - mean[:, None]
        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        output = centered * invstd[:, None] * weight[None, :] + bias[None, :]

        output_offsets = out_rows[:, None] * channels + cols[None, :]
        tl.store(output_ptr + output_offsets, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects eight inputs, got {len(inputs)}")

    mm_1 = _require_tensor("mm_1", inputs[0], INPUT_SHAPE, DTYPE, OUTPUT_STRIDE)
    weight = _require_tensor("primals_70", inputs[1], AFFINE_SHAPE, DTYPE, (1,))
    bias = _require_tensor("primals_71", inputs[2], AFFINE_SHAPE, DTYPE, (1,))

    actual_shapes = tuple(_shape_tuple(shape_param) for shape_param in inputs[3:])
    if actual_shapes != SHAPE_PARAMS:
        raise ValueError(f"shape parameters {actual_shapes} != {SHAPE_PARAMS}")

    device = mm_1.device
    for name, tensor in (("primals_70", weight), ("primals_71", bias)):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return mm_1, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    mm_1, weight, bias = _validate_inputs(inputs)
    reshape_default = torch.ops.aten.reshape.default(mm_1, _shape_tuple(inputs[3]))
    var, mean = torch.ops.aten.var_mean.correction(
        reshape_default,
        [3],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS))
    centered = torch.ops.aten.sub.Tensor(reshape_default, mean)
    normalized = torch.ops.aten.mul.Tensor(centered, invstd)
    scaled = torch.ops.aten.mul.Tensor(normalized, weight)
    shifted = torch.ops.aten.add.Tensor(scaled, bias)
    windows = torch.ops.aten.reshape.default(shifted, _shape_tuple(inputs[4]))
    permuted = torch.ops.aten.permute.default(windows, [0, 1, 3, 2, 4, 5])
    cloned = torch.ops.aten.clone.default(
        permuted,
        memory_format=torch.contiguous_format,
    )
    flattened = torch.ops.aten.reshape.default(cloned, _shape_tuple(inputs[5]))
    flattened = torch.ops.aten.reshape.default(flattened, _shape_tuple(inputs[6]))
    return torch.ops.aten.reshape.default(flattened, _shape_tuple(inputs[7]))


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
    mm_1, weight, bias = _validate_inputs(inputs)
    if triton is None or not mm_1.is_cuda:
        return _torch_full_scope(inputs)

    base = torch.empty_strided(
        CLONE_SHAPE,
        CLONE_STRIDE,
        device=mm_1.device,
        dtype=DTYPE,
    )
    output = torch.ops.aten.reshape.default(base, _shape_tuple(inputs[7]))
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_window_layernorm_kernel[grid](
        mm_1,
        weight,
        bias,
        output,
        total_rows=ROWS,
        height=HEIGHT,
        width=WIDTH,
        window=WINDOW,
        window_blocks=WINDOW_BLOCKS,
        channels=CHANNELS,
        eps=EPS,
        BLOCK_C=CHANNELS,
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
