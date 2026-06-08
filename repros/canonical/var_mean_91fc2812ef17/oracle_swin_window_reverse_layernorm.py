"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the Swin window-reverse reshape/permute/clone, residual add from the channels-last-strided tensor, hidden-size-128 population LayerNorm, affine scale/bias, and final contiguous flatten into one Triton row kernel, whereas Inductor currently materializes the window reverse/add tensor and then runs a separate var_mean LayerNorm/affine/clone path; Inductor cannot do this today because the normalization scheduler does not sink fixed reshape/permute/clone layout indexing and residual producer loads into the row-statistics template; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to fold deterministic window-reverse indexing plus pointwise residual producers into the LayerNorm load plan and write the final flattened layout directly."""
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
HEIGHT = 56
WIDTH = 56
CHANNELS = 128
WINDOW = 7
WINDOW_BLOCKS = 8
ROWS = BATCH * HEIGHT * WIDTH
ADDMM_SHAPE = (ROWS, CHANNELS)
RESIDUAL_SHAPE = (BATCH, HEIGHT, WIDTH, CHANNELS)
RESIDUAL_STRIDE = (ROWS, WIDTH, 1, HEIGHT * WIDTH)
OUTPUT_SHAPE = (ROWS, CHANNELS)
EPS = 1.0e-5
BLOCK_C = 128
ROW_BLOCK = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _swin_window_reverse_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        window: tl.constexpr,
        window_blocks: tl.constexpr,
        eps: tl.constexpr,
        block_c: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, block_c)
        row_mask = rows < total_rows

        hw = rows % (height * width)
        batch = rows // (height * width)
        h = hw // width
        w = hw % width
        block_h = h // window
        block_w = w // window
        inner_h = h % window
        inner_w = w % window

        window_id = batch * (window_blocks * window_blocks) + block_h * window_blocks + block_w
        token = inner_h * window + inner_w
        addmm_row = window_id * (window * window) + token

        addmm_offsets = addmm_row[:, None] * channels + cols[None, :]
        residual_offsets = (
            batch[:, None] * (height * width * channels)
            + cols[None, :] * (height * width)
            + h[:, None] * width
            + w[:, None]
        )
        mask = row_mask[:, None]

        addmm = tl.load(
            addmm_ptr + addmm_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = addmm + residual

        mean = tl.sum(tl.where(mask, x, 0.0), axis=1) * (1.0 / 128.0)
        centered = tl.where(mask, x - mean[:, None], 0.0)
        variance = tl.sum(centered * centered, axis=1) * (1.0 / 128.0)
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        out = centered * invstd[:, None] * weight[None, :] + bias[None, :]

        out_offsets = rows[:, None] * channels + cols[None, :]
        tl.store(output_ptr + out_offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects nine inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_1", inputs[0], ADDMM_SHAPE, torch.float32)
    residual = _require_tensor("add_1", inputs[1], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg13_1", inputs[2], (CHANNELS,), torch.float32)
    bias = _require_tensor("arg14_1", inputs[3], (CHANNELS,), torch.float32)

    if tuple(addmm.stride()) != (CHANNELS, 1):
        raise ValueError(f"addmm_1 has stride {tuple(addmm.stride())}, expected {(CHANNELS, 1)}")
    if tuple(residual.stride()) != RESIDUAL_STRIDE:
        raise ValueError(
            f"add_1 has stride {tuple(residual.stride())}, expected {RESIDUAL_STRIDE}"
        )
    if tuple(weight.stride()) != (1,) or tuple(bias.stride()) != (1,):
        raise ValueError("arg13_1 and arg14_1 must be contiguous 1D tensors")

    expected_shapes = (
        (8192, 49, CHANNELS),
        (-1, WINDOW, WINDOW, CHANNELS),
        (-1, WINDOW_BLOCKS, WINDOW_BLOCKS, WINDOW, WINDOW, CHANNELS),
        (-1, HEIGHT, WIDTH, CHANNELS),
        (BATCH, HEIGHT * WIDTH, CHANNELS),
    )
    for index, expected in enumerate(expected_shapes, start=4):
        actual = _shape_tuple(inputs[index])
        if actual != expected:
            raise ValueError(f"unexpected _shape_param_{index - 4}: {actual}, expected {expected}")

    device = addmm.device
    if residual.device != device or weight.device != device or bias.device != device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return addmm, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    x = torch.ops.aten.view.default(addmm, _shape_tuple(inputs[4]))
    x = torch.ops.aten.view.default(x, _shape_tuple(inputs[5]))
    x = torch.ops.aten.view.default(x, _shape_tuple(inputs[6]))
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    x = torch.ops.aten.view.default(x, _shape_tuple(inputs[7]))
    x = torch.ops.aten.add.Tensor(residual, x)
    x = torch.ops.aten.view.default(x, _shape_tuple(inputs[8]))
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    x = torch.ops.aten.sub.Tensor(x, mean)
    x = torch.ops.aten.mul.Tensor(x, torch.ops.aten.rsqrt.default(variance + EPS))
    x = torch.ops.aten.mul.Tensor(x, weight)
    x = torch.ops.aten.add.Tensor(x, bias)
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    return torch.ops.aten._unsafe_view.default(x, [ROWS, CHANNELS])


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Swin window-reverse residual LayerNorm repro scope."""
    if triton is None:
        return _torch_full_scope(inputs)

    addmm, residual, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        (CHANNELS, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = (triton.cdiv(ROWS, ROW_BLOCK),)
    _swin_window_reverse_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        total_rows=ROWS,
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        window=WINDOW,
        window_blocks=WINDOW_BLOCKS,
        eps=EPS,
        block_c=BLOCK_C,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=4,
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
