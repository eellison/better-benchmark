"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes both hidden-size-128 LayerNorm affine stages from the channels-last convolution input and stores the second affine result directly in the final contiguous 7x7 Swin window-partition layout, whereas Inductor currently schedules the two var_mean LayerNorm templates and the fixed permute/clone layout copy as separate materializing steps; Inductor cannot do this today because the normalization scheduler does not compose consecutive row-reduction epilogues or sink a deterministic window-partition layout store into the second normalization epilogue; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to fuse chained LayerNorm epilogues with fixed reshape/permute/clone layout indexing for direct final-layout stores."""
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
CHANNELS = 128
HEIGHT = 56
WIDTH = 56
WINDOW = 7
WINDOW_BLOCKS = 8
ROWS = BATCH * HEIGHT * WIDTH
INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HEIGHT * WIDTH, 1, CHANNELS * WIDTH, CHANNELS)
OUTPUT_SHAPE = (ROWS, CHANNELS)
EPS = 1.0e-5
BLOCK_C = 128


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
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 16}, num_warps=8, num_stages=3),
        ],
        key=["total_rows"],
    )
    @triton.jit
    def _dual_layernorm_window_kernel(
        convolution_ptr,
        weight1_ptr,
        bias1_ptr,
        weight2_ptr,
        bias2_ptr,
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
        out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, block_c)
        row_mask = out_rows < total_rows

        inner_w = out_rows % window
        tmp = out_rows // window
        inner_h = tmp % window
        tmp = tmp // window
        block_w = tmp % window_blocks
        tmp = tmp // window_blocks
        block_h = tmp % window_blocks
        batch = tmp // window_blocks

        src_h = block_h * window + inner_h
        src_w = block_w * window + inner_w
        src_rows = (batch * height + src_h) * width + src_w

        offsets = src_rows[:, None] * channels + cols[None, :]
        mask = row_mask[:, None]
        x = tl.load(
            convolution_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        mean1 = tl.sum(tl.where(mask, x, 0.0), axis=1) * (1.0 / 128.0)
        centered1 = tl.where(mask, x - mean1[:, None], 0.0)
        var1 = tl.sum(centered1 * centered1, axis=1) * (1.0 / 128.0)
        invstd1 = 1.0 / tl.sqrt(var1 + eps)

        weight1 = tl.load(weight1_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias1 = tl.load(bias1_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        y = centered1 * invstd1[:, None] * weight1[None, :] + bias1[None, :]

        mean2 = tl.sum(tl.where(mask, y, 0.0), axis=1) * (1.0 / 128.0)
        centered2 = tl.where(mask, y - mean2[:, None], 0.0)
        var2 = tl.sum(centered2 * centered2, axis=1) * (1.0 / 128.0)
        invstd2 = 1.0 / tl.sqrt(var2 + eps)

        weight2 = tl.load(weight2_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias2 = tl.load(bias2_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        out = centered2 * invstd2[:, None] * weight2[None, :] + bias2[None, :]

        out_offsets = out_rows[:, None] * channels + cols[None, :]
        tl.store(output_ptr + out_offsets, out, mask=mask)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects nine inputs, got {len(inputs)}")

    convolution = _require_tensor("convolution", inputs[0], INPUT_SHAPE, torch.float32)
    weight1 = _require_tensor("arg3_1", inputs[1], (CHANNELS,), torch.float32)
    bias1 = _require_tensor("arg4_1", inputs[2], (CHANNELS,), torch.float32)
    weight2 = _require_tensor("arg5_1", inputs[3], (CHANNELS,), torch.float32)
    bias2 = _require_tensor("arg6_1", inputs[4], (CHANNELS,), torch.float32)

    expected_shapes = (
        (BATCH, WINDOW_BLOCKS, WINDOW, WINDOW_BLOCKS, WINDOW, CHANNELS),
        (-1, WINDOW, WINDOW, CHANNELS),
        (-1, WINDOW * WINDOW, CHANNELS),
        OUTPUT_SHAPE,
    )
    for index, expected in enumerate(expected_shapes, start=5):
        if _shape_tuple(inputs[index]) != expected:
            raise ValueError(f"unexpected shape parameter {index}: {inputs[index]!r}")

    if tuple(convolution.stride()) != INPUT_STRIDE:
        raise ValueError(
            f"convolution has stride {tuple(convolution.stride())}, expected {INPUT_STRIDE}"
        )

    device = convolution.device
    if not (weight1.device == bias1.device == weight2.device == bias2.device == device):
        raise ValueError("all tensor inputs must be on the same device")
    return convolution, weight1, bias1, weight2, bias2


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    convolution, weight1, bias1, weight2, bias2 = _validate_inputs(inputs)
    x = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1])
    var1, mean1 = torch.ops.aten.var_mean.correction(
        x, [3], correction=0, keepdim=True
    )
    x = (x - mean1) * torch.ops.aten.rsqrt.default(var1 + EPS) * weight1 + bias1
    var2, mean2 = torch.ops.aten.var_mean.correction(
        x, [3], correction=0, keepdim=True
    )
    x = (x - mean2) * torch.ops.aten.rsqrt.default(var2 + EPS) * weight2 + bias2
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[5]))
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[6]))
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[7]))
    return torch.ops.aten.reshape.default(x, _shape_tuple(inputs[8]))


@oracle_impl(hardware="H100", shapes="(T([128, 128, 56, 56], f32, stride=(401408, 1, 7168, 128)), T([128], f32), T([128], f32), T([128], f32), T([128], f32), S([128, 8, 7, 8, 7, 128]), S([-1, 7, 7, 128]), S([-1, 49, 128]), S([401408, 128]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    convolution, weight1, bias1, weight2, bias2 = _validate_inputs(inputs)
    if triton is None or not convolution.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        (CHANNELS, 1),
        device=convolution.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _dual_layernorm_window_kernel[grid](
        convolution,
        weight1,
        bias1,
        weight2,
        bias2,
        output,
        total_rows=ROWS,
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        window=WINDOW,
        window_blocks=WINDOW_BLOCKS,
        eps=EPS,
        block_c=BLOCK_C,
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
