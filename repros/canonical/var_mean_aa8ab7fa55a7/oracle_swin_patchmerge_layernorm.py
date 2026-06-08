"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin patch-merge residual add, fixed 2x2 layout clone, population var_mean LayerNorm, affine epilogue, and final contiguous `[6272, 2048]` reshape in one row Triton kernel, whereas Inductor currently schedules the residual/add layout materialization and the hidden-size-2048 normalization as separate generic regions; Inductor cannot do this today because the normalization scheduler does not recognize the deterministic Swin patch-merge reshape/permute/clone producer as a direct row-source for the LayerNorm template; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to sink fixed patch-merge layout indexing and residual-add producers into the row-wise LayerNorm load plan."""
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
INPUT_TOKENS = 196
INPUT_CHANNELS = 512
INPUT_HEIGHT = 14
INPUT_WIDTH = 14
PATCH = 2
MERGED_HEIGHT = 7
MERGED_WIDTH = 7
HIDDEN = PATCH * PATCH * INPUT_CHANNELS
ROWS = BATCH * MERGED_HEIGHT * MERGED_WIDTH
EPS = 1.0e-5
BLOCK_H = 2048


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
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
        ],
        key=["total_rows"],
    )
    @triton.jit
    def _swin_patchmerge_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        input_tokens: tl.constexpr,
        input_channels: tl.constexpr,
        input_width: tl.constexpr,
        merged_width: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, block_h)
        row_mask = out_rows < total_rows
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]

        channel_group = cols // input_channels
        channel = cols - channel_group * input_channels
        inner_w = channel_group // 2
        inner_h = channel_group - inner_w * 2

        merged_w = out_rows % merged_width
        tmp = out_rows // merged_width
        merged_h = tmp % merged_width
        batch = tmp // merged_width

        source_h = merged_h[:, None] * 2 + inner_h[None, :]
        source_w = merged_w[:, None] * 2 + inner_w[None, :]
        source_token = source_h * input_width + source_w
        source_offsets = (
            batch[:, None] * input_tokens * input_channels
            + source_token * input_channels
            + channel[None, :]
        )

        addmm = tl.load(
            addmm_ptr + source_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + source_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = addmm + residual

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

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
        output = centered * invstd * weight[None, :] + bias[None, :]

        output_offsets = out_rows[:, None] * hidden + cols[None, :]
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
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects nine inputs, got {len(inputs)}")

    addmm = _require_tensor(
        "addmm_87",
        inputs[0],
        (BATCH * INPUT_TOKENS, INPUT_CHANNELS),
        torch.float32,
    )
    residual = _require_tensor(
        "view_596",
        inputs[1],
        (BATCH, INPUT_TOKENS, INPUT_CHANNELS),
        torch.float32,
    )
    weight = _require_tensor("arg330_1", inputs[2], (HIDDEN,), torch.float32)
    bias = _require_tensor("arg331_1", inputs[3], (HIDDEN,), torch.float32)

    expected_shapes = (
        (BATCH, INPUT_TOKENS, INPUT_CHANNELS),
        (BATCH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS),
        (BATCH, MERGED_HEIGHT, PATCH, MERGED_WIDTH, PATCH, INPUT_CHANNELS),
        (BATCH, MERGED_HEIGHT, MERGED_WIDTH, HIDDEN),
        (ROWS, HIDDEN),
    )
    for index, expected in enumerate(expected_shapes, start=4):
        actual = _shape_tuple(inputs[index])
        if actual != expected:
            raise ValueError(f"shape parameter {index} is {actual}, expected {expected}")

    device = addmm.device
    if not (residual.device == weight.device == bias.device == device):
        raise ValueError("all tensor inputs must be on the same device")

    return addmm, residual, weight, bias, expected_shapes[-1]


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias, output_shape = _validate_inputs(inputs)
    x = torch.ops.aten.reshape.default(addmm, _shape_tuple(inputs[4]))
    x = torch.ops.aten.add.Tensor(residual, x)
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[5]))
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[6]))
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 4, 2, 5])
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[7]))
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [3], correction=0, keepdim=True
    )
    x = torch.ops.aten.sub.Tensor(x, mean)
    x = torch.ops.aten.mul.Tensor(
        x,
        torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS)),
    )
    x = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(x, weight), bias)
    return torch.ops.aten.reshape.default(x, output_shape)


def oracle_forward(inputs):
    """Run the complete Repro.forward computation.

    SCOPE INVARIANT: accepts the same nine inputs as Repro.forward() and returns
    the same single float32 `[6272, 2048]` contiguous output tensor.
    """
    addmm, residual, weight, bias, output_shape = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        output_shape,
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_patchmerge_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        total_rows=ROWS,
        input_tokens=INPUT_TOKENS,
        input_channels=INPUT_CHANNELS,
        input_width=INPUT_WIDTH,
        merged_width=MERGED_WIDTH,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
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
