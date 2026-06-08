"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the residual add, hidden-size-512 population var_mean LayerNorm affine, and fixed Swin 7x7 window-partition clone in one output-contiguous Triton row kernel, whereas Inductor currently fuses the same scope but traverses source rows and emits strided window-layout stores from the normalization epilogue; Inductor cannot do this today because its normalization scheduler preserves producer-contiguous row order when sinking deterministic reshape/permute/clone consumers instead of cost-modeling an output-contiguous remap; the fix is SCHEDULER_FUSION: teach the LayerNorm scheduler to choose between source-contiguous and output-contiguous epilogue traversal for fixed layout clones."""
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


BATCH = 128
TOKENS = 196
HEIGHT = 14
WIDTH = 14
WINDOW = 7
WINDOW_BLOCKS = 2
CHANNELS = 512
ROWS = BATCH * TOKENS
EPS = 1.0e-5
DTYPE = torch.float32

ADDMM_SHAPE = (ROWS, CHANNELS)
RESIDUAL_SHAPE = (BATCH, TOKENS, CHANNELS)
OUTPUT_SHAPE = (ROWS, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)

VIEW_SHAPES = (
    (BATCH, TOKENS, CHANNELS),
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
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        tokens: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        window: tl.constexpr,
        window_blocks: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, 512)
        row_mask = out_rows < total_rows

        batch = out_rows // tokens
        within_image = out_rows - batch * tokens
        window_id = within_image // (window * window)
        position = within_image - window_id * (window * window)
        block_h = window_id // window_blocks
        block_w = window_id - block_h * window_blocks
        inner_h = position // window
        inner_w = position - inner_h * window
        src_h = block_h * window + inner_h
        src_w = block_w * window + inner_w
        src_rows = batch * tokens + src_h * width + src_w

        input_offsets = src_rows[:, None] * channels + cols[None, :]
        mask = row_mask[:, None]
        addmm = tl.load(
            addmm_ptr + input_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + input_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = addmm + residual
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=1) * (1.0 / 512.0)
        centered = tl.where(mask, x - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, axis=1) * (1.0 / 512.0)
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
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
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects ten inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_79", inputs[0], ADDMM_SHAPE, DTYPE)
    residual = _require_tensor("view_540", inputs[1], RESIDUAL_SHAPE, DTYPE)
    weight = _require_tensor("arg301_1", inputs[2], (CHANNELS,), DTYPE)
    bias = _require_tensor("arg302_1", inputs[3], (CHANNELS,), DTYPE)

    for index, expected_shape in enumerate(VIEW_SHAPES, start=4):
        shape = _shape_tuple(inputs[index])
        if shape != expected_shape:
            raise ValueError(
                f"unexpected shape parameter {index}: {shape}, expected {expected_shape}"
            )

    tensors = (addmm, residual, weight, bias)
    device = addmm.device
    for index, tensor in enumerate(tensors):
        if tensor.device != device:
            raise ValueError(f"input {index} device {tensor.device} != {device}")
        if not tensor.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={tensor.stride()}")

    return addmm, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    x = torch.ops.aten.view.default(addmm, _shape_tuple(inputs[4]))
    x = torch.ops.aten.add.Tensor(residual, x)
    x = torch.ops.aten.view.default(x, _shape_tuple(inputs[5]))
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [3], correction=0, keepdim=True
    )
    x = torch.ops.aten.sub.Tensor(x, mean)
    x = torch.ops.aten.mul.Tensor(x, torch.ops.aten.rsqrt.default(variance + EPS))
    x = torch.ops.aten.mul.Tensor(x, weight)
    x = torch.ops.aten.add.Tensor(x, bias)
    x = torch.ops.aten.view.default(x, _shape_tuple(inputs[6]))
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    x = torch.ops.aten.view.default(x, _shape_tuple(inputs[7]))
    x = torch.ops.aten.view.default(x, _shape_tuple(inputs[8]))
    return torch.ops.aten.view.default(x, _shape_tuple(inputs[9]))


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
    addmm, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=DTYPE,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_window_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        total_rows=ROWS,
        tokens=TOKENS,
        height=HEIGHT,
        width=WIDTH,
        window=WINDOW,
        window_blocks=WINDOW_BLOCKS,
        channels=CHANNELS,
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
