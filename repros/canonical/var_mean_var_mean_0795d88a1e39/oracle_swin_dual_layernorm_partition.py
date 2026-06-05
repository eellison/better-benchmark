"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin dual channel-LayerNorm region, including both var_mean reductions, affine epilogues, and the 7x7 window partition clone/view by scheduling source spatial rows and scattering them directly to the final `[401408,128]` window layout, whereas Inductor lowers the decomposed graph as generic norm-template kernels plus separate materialization/layout work; Inductor cannot do this today because its scheduler has no chained LayerNorm plus Swin window-partition template that keeps the first normalization in registers and sinks the final clone layout into the second normalization store; the fix is NEW_PATTERN: add a guarded Swin dual-LayerNorm partition lowering or a general dependent-normalization fusion that emits the final layout directly."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
BATCH = 128
CHANNELS = 128
HEIGHT = 56
WIDTH = 56
GRID_H = 8
GRID_W = 8
WINDOW_H = 7
WINDOW_W = 7
ROWS = BATCH * HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_C = 128
ROW_BLOCK = 64

EXPECTED_SHAPE0 = (BATCH, GRID_H, WINDOW_H, GRID_W, WINDOW_W, CHANNELS)
EXPECTED_SHAPE1 = (-1, WINDOW_H, WINDOW_W, CHANNELS)
EXPECTED_SHAPE2 = (-1, WINDOW_H * WINDOW_W, CHANNELS)
EXPECTED_SHAPE3 = (ROWS, CHANNELS)

if triton is not None:

    @triton.jit
    def _swin_dual_layernorm_source_order_kernel(
        convolution_ptr,
        weight1_ptr,
        bias1_ptr,
        weight2_ptr,
        bias2_ptr,
        out_ptr,
        conv_s0: tl.constexpr,
        conv_s1: tl.constexpr,
        conv_s2: tl.constexpr,
        conv_s3: tl.constexpr,
        rows_total: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        channels: tl.constexpr,
        grid_h: tl.constexpr,
        grid_w: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_CH: tl.constexpr,
        ROW_BLK: tl.constexpr,
    ):
        source_row = tl.program_id(0) * ROW_BLK + tl.arange(0, ROW_BLK)[:, None]
        cols = tl.arange(0, BLOCK_CH)[None, :]
        row_mask = source_row < rows_total
        col_mask = cols < channels
        mask = row_mask & col_mask

        spatial = source_row % (height * width)
        batch = source_row // (height * width)
        src_h = spatial // width
        src_w = spatial - src_h * width

        window_row = src_h // window_h
        inner_h = src_h - window_row * window_h
        window_col = src_w // window_w
        inner_w = src_w - window_col * window_w
        out_row = (
            (((batch * grid_h + window_row) * grid_w + window_col) * window_h + inner_h)
            * window_w
            + inner_w
        )

        conv_offsets = (
            batch * conv_s0
            + cols * conv_s1
            + src_h * conv_s2
            + src_w * conv_s3
        )
        x = tl.load(
            convolution_ptr + conv_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        mean1 = (tl.sum(tl.where(mask, x, 0.0), axis=1) / channels)[:, None]
        centered1 = x - mean1
        var1 = (
            tl.sum(tl.where(mask, centered1 * centered1, 0.0), axis=1) / channels
        )[:, None]
        invstd1 = tl.rsqrt(var1 + eps)

        weight1 = tl.load(
            weight1_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias1 = tl.load(
            bias1_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y1 = centered1 * invstd1 * weight1 + bias1

        mean2 = (tl.sum(tl.where(mask, y1, 0.0), axis=1) / channels)[:, None]
        centered2 = y1 - mean2
        var2 = (
            tl.sum(tl.where(mask, centered2 * centered2, 0.0), axis=1) / channels
        )[:, None]
        invstd2 = tl.rsqrt(var2 + eps)

        weight2 = tl.load(
            weight2_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias2 = tl.load(
            bias2_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y2 = centered2 * invstd2 * weight2 + bias2

        tl.store(out_ptr + out_row * channels + cols, y2, mask=mask)

    @triton.jit
    def _swin_dual_layernorm_source_order_ch_major_kernel(
        convolution_ptr,
        weight1_ptr,
        bias1_ptr,
        weight2_ptr,
        bias2_ptr,
        out_ptr,
        conv_s0: tl.constexpr,
        conv_s1: tl.constexpr,
        conv_s2: tl.constexpr,
        conv_s3: tl.constexpr,
        rows_total: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        channels: tl.constexpr,
        grid_h: tl.constexpr,
        grid_w: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_CH: tl.constexpr,
        ROW_BLK: tl.constexpr,
    ):
        cols = tl.arange(0, BLOCK_CH)[:, None]
        source_row = tl.program_id(0) * ROW_BLK + tl.arange(0, ROW_BLK)[None, :]
        col_mask = cols < channels
        row_mask = source_row < rows_total
        mask = col_mask & row_mask

        spatial = source_row % (height * width)
        batch = source_row // (height * width)
        src_h = spatial // width
        src_w = spatial - src_h * width

        window_row = src_h // window_h
        inner_h = src_h - window_row * window_h
        window_col = src_w // window_w
        inner_w = src_w - window_col * window_w
        out_row = (
            (((batch * grid_h + window_row) * grid_w + window_col) * window_h + inner_h)
            * window_w
            + inner_w
        )

        conv_offsets = (
            batch * conv_s0
            + cols * conv_s1
            + src_h * conv_s2
            + src_w * conv_s3
        )
        x = tl.load(
            convolution_ptr + conv_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        mean1 = (tl.sum(tl.where(mask, x, 0.0), axis=0) / channels)[None, :]
        centered1 = x - mean1
        var1 = (
            tl.sum(tl.where(mask, centered1 * centered1, 0.0), axis=0) / channels
        )[None, :]
        invstd1 = tl.rsqrt(var1 + eps)

        weight1 = tl.load(
            weight1_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias1 = tl.load(
            bias1_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y1 = centered1 * invstd1 * weight1 + bias1

        mean2 = (tl.sum(tl.where(mask, y1, 0.0), axis=0) / channels)[None, :]
        centered2 = y1 - mean2
        var2 = (
            tl.sum(tl.where(mask, centered2 * centered2, 0.0), axis=0) / channels
        )[None, :]
        invstd2 = tl.rsqrt(var2 + eps)

        weight2 = tl.load(
            weight2_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias2 = tl.load(
            bias2_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y2 = centered2 * invstd2 * weight2 + bias2

        tl.store(out_ptr + out_row * channels + cols, y2, mask=mask)


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
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    if len(inputs) != 9:
        raise ValueError(f"{REPRO_ID} expects 9 inputs, got {len(inputs)}")

    convolution, weight1, bias1, weight2, bias2, shape0, shape1, shape2, shape3 = inputs
    convolution = _require_f32_tensor(
        "convolution",
        convolution,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
    )
    weight1 = _require_f32_tensor("arg3_1", weight1, (CHANNELS,))
    bias1 = _require_f32_tensor("arg4_1", bias1, (CHANNELS,))
    weight2 = _require_f32_tensor("arg5_1", weight2, (CHANNELS,))
    bias2 = _require_f32_tensor("arg6_1", bias2, (CHANNELS,))

    device = convolution.device
    for name, tensor in (
        ("arg3_1", weight1),
        ("arg4_1", bias1),
        ("arg5_1", weight2),
        ("arg6_1", bias2),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    shape0 = _shape_tuple(shape0)
    shape1 = _shape_tuple(shape1)
    shape2 = _shape_tuple(shape2)
    shape3 = _shape_tuple(shape3)
    if shape0 != EXPECTED_SHAPE0:
        raise ValueError(f"unexpected first view shape: {shape0!r}")
    if shape1 != EXPECTED_SHAPE1:
        raise ValueError(f"unexpected second view shape: {shape1!r}")
    if shape2 != EXPECTED_SHAPE2:
        raise ValueError(f"unexpected third view shape: {shape2!r}")
    if shape3 != EXPECTED_SHAPE3:
        raise ValueError(f"unexpected output view shape: {shape3!r}")

    return convolution, weight1, bias1, weight2, bias2, shape0, shape1, shape2, shape3


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    convolution, weight1, bias1, weight2, bias2, shape0, shape1, shape2, shape3 = _validate_inputs(inputs)
    permuted = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1])
    var1, mean1 = torch.ops.aten.var_mean.correction(
        permuted,
        [3],
        correction=0,
        keepdim=True,
    )
    normalized1 = (permuted - mean1) * torch.ops.aten.rsqrt.default(var1 + EPS)
    affine1 = normalized1 * weight1 + bias1
    var2, mean2 = torch.ops.aten.var_mean.correction(
        affine1,
        [3],
        correction=0,
        keepdim=True,
    )
    normalized2 = (affine1 - mean2) * torch.ops.aten.rsqrt.default(var2 + EPS)
    affine2 = normalized2 * weight2 + bias2
    view0 = torch.ops.aten.view.default(affine2, shape0)
    permuted_windows = torch.ops.aten.permute.default(view0, [0, 1, 3, 2, 4, 5])
    contiguous = torch.ops.aten.clone.default(
        permuted_windows,
        memory_format=torch.contiguous_format,
    )
    view1 = torch.ops.aten.view.default(contiguous, shape1)
    view2 = torch.ops.aten.view.default(view1, shape2)
    return torch.ops.aten.view.default(view2, shape3)


def oracle_forward(inputs):
    """Run the complete dual LayerNorm plus Swin window-partition repro scope."""
    convolution, weight1, bias1, weight2, bias2, _shape0, _shape1, _shape2, shape3 = _validate_inputs(inputs)
    if triton is None or not convolution.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        shape3,
        (CHANNELS, 1),
        device=convolution.device,
        dtype=torch.float32,
    )
    kernel = (
        _swin_dual_layernorm_source_order_kernel
        if convolution.stride(1) == 1
        else _swin_dual_layernorm_source_order_ch_major_kernel
    )
    kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        convolution,
        weight1,
        bias1,
        weight2,
        bias2,
        output,
        convolution.stride(0),
        convolution.stride(1),
        convolution.stride(2),
        convolution.stride(3),
        rows_total=ROWS,
        height=HEIGHT,
        width=WIDTH,
        channels=CHANNELS,
        grid_h=GRID_H,
        grid_w=GRID_W,
        window_h=WINDOW_H,
        window_w=WINDOW_W,
        eps=EPS,
        BLOCK_CH=BLOCK_C,
        ROW_BLK=ROW_BLOCK,
        num_warps=8,
        num_stages=2,
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
