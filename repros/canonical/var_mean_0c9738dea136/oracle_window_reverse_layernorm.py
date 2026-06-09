"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin window-reverse residual LayerNorm scope in one shape-specialized Triton row kernel, keeping the reconstructed residual-add row tile live through correction=0 Welford var_mean, libdevice.rsqrt, affine output, final contiguous `[401408,128]` view, and `[128,3136,1]` `rsqrt * 1/128` side output, whereas Inductor already emits one fused generic Welford reduction kernel for the same full scope and CUDAGraph timing shows the retained-tile variant at floor rather than a material speedup; Inductor cannot materially improve this local repro today because the remaining fixed-width row reconstruction, Welford reduction, affine parameter reads, output store, and side-output store dominate over avoidable scheduler overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor Swin LayerNorm layout case unless broader normalization-template, launch, or memory-traffic work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    triton_helpers = None
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
HEIGHT = 56
WIDTH = 56
WINDOW = 7
WINDOW_BLOCKS = 8
TOKENS = HEIGHT * WIDTH
CHANNELS = 128
ROWS = BATCH * TOKENS
EPS = 1.0e-5
INV_CHANNELS = 0.0078125
DTYPE = torch.float32

ADDMM_SHAPE = (ROWS, CHANNELS)
RESIDUAL_SHAPE = (BATCH, HEIGHT, WIDTH, CHANNELS)
WEIGHT_SHAPE = (CHANNELS,)
OUTPUT_SHAPE = (ROWS, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
SIDE_SHAPE = (BATCH, TOKENS, 1)
SIDE_STRIDE = (TOKENS, 1, 1)

SHAPE_PARAMS = (
    (8192, 49, 128),
    (-1, 7, 7, 128),
    (-1, 8, 8, 7, 7, 128),
    (-1, 56, 56, 128),
    (128, -1, 128),
    (401408, 128),
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


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

    addmm = _require_tensor("addmm_1", inputs[0], ADDMM_SHAPE, DTYPE)
    residual = _require_tensor("add_1", inputs[1], RESIDUAL_SHAPE, DTYPE)
    weight = _require_tensor("primals_14", inputs[2], WEIGHT_SHAPE, DTYPE)
    bias = _require_tensor("primals_15", inputs[3], WEIGHT_SHAPE, DTYPE)

    for index, expected in enumerate(SHAPE_PARAMS, start=4):
        actual = _shape_tuple(inputs[index])
        if actual != expected:
            raise ValueError(
                f"unexpected shape parameter {index}: {actual}, expected {expected}"
            )

    device = addmm.device
    for index, tensor in enumerate((addmm, residual, weight, bias)):
        if tensor.device != device:
            raise ValueError(f"input {index} device {tensor.device} != {device}")
        if not tensor.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={tensor.stride()}")

    return addmm, residual, weight, bias


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
        ],
        key=["total_rows"],
    )
    @triton.jit
    def _window_reverse_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        side_ptr,
        total_rows: tl.constexpr,
        tokens: tl.constexpr,
        width: tl.constexpr,
        window: tl.constexpr,
        window_blocks: tl.constexpr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        inv_channels: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        row_mask = row_ids < total_rows
        rows = row_ids[:, None]
        cols = tl.arange(0, 128)[None, :]

        token = rows % tokens
        batch = rows // tokens
        h = token // width
        w = token - h * width
        block_h = h // window
        block_w = w // window
        inner_h = h - block_h * window
        inner_w = w - block_w * window

        addmm_offsets = (
            cols
            + channels * inner_w
            + channels * window * inner_h
            + channels * window * window * block_w
            + channels * window * window * window_blocks * block_h
            + channels * window * window * window_blocks * window_blocks * batch
        )
        residual_offsets = rows * channels + cols
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
        x = residual + addmm

        mean_state = tl.zeros([ROW_BLOCK, 128], tl.float32)
        m2_state = tl.zeros([ROW_BLOCK, 128], tl.float32)
        weight_state = tl.zeros([ROW_BLOCK, 128], tl.float32)
        mean_state, m2_state, weight_state = triton_helpers.welford_reduce(
            x,
            mean_state,
            m2_state,
            weight_state,
            True,
        )
        mean, m2, _count = triton_helpers.welford(mean_state, m2_state, weight_state, 1)
        mean = mean[:, None]
        m2 = m2[:, None]
        channel_count = tl.full([1, 1], 128.0, tl.float32)
        eps_value = tl.full([1, 1], eps, tl.float32)
        variance = m2 / channel_count
        invstd = libdevice.rsqrt(variance + eps_value)

        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        centered = x - mean
        normalized = centered * invstd
        scaled = normalized * weight
        output = scaled + bias

        output_offsets = rows * channels + cols
        tl.store(output_ptr + output_offsets, output, mask=mask)

        side_variance = m2 / channel_count
        side_invstd = libdevice.rsqrt(side_variance + eps_value)
        side_scale = tl.full([1, 1], inv_channels, tl.float32)
        tl.store(side_ptr + rows, side_invstd * side_scale, mask=mask)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    x = torch.ops.aten.reshape.default(addmm, _shape_tuple(inputs[4]))
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[5]))
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[6]))
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[7]))
    x = torch.ops.aten.add.Tensor(residual, x)
    x = torch.ops.aten.reshape.default(x, _shape_tuple(inputs[8]))
    variance, mean = torch.ops.aten.var_mean.correction(
        x, [2], correction=0, keepdim=True
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    y = torch.ops.aten.sub.Tensor(x, mean)
    y = torch.ops.aten.mul.Tensor(y, invstd)
    y = torch.ops.aten.mul.Tensor(y, weight)
    y = torch.ops.aten.add.Tensor(y, bias)
    y = torch.ops.aten.reshape.default(y, _shape_tuple(inputs[9]))
    side = torch.ops.aten.mul.Tensor(invstd, INV_CHANNELS)
    return y, side


@oracle_impl(hardware="H100", shapes="(T([401408, 128], f32), T([128, 56, 56, 128], f32), T([128], f32), T([128], f32), S([8192, 49, 128]), S([-1, 7, 7, 128]), S([-1, 8, 8, 7, 7, 128]), S([-1, 56, 56, 128]), S([128, -1, 128]), S([401408, 128]))")
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
    if triton is None or triton_helpers is None or libdevice is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=DTYPE,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=DTYPE,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _window_reverse_layernorm_kernel[grid](
        addmm,
        residual,
        weight,
        bias,
        output,
        side,
        total_rows=ROWS,
        tokens=TOKENS,
        width=WIDTH,
        window=WINDOW,
        window_blocks=WINDOW_BLOCKS,
        channels=CHANNELS,
        eps=EPS,
        inv_channels=INV_CHANNELS,
    )
    return output, side


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
