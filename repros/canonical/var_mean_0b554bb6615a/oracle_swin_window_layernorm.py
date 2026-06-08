"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Swin residual-add LayerNorm plus unshifted 7x7 window partition scope in one Triton row-reduction kernel, preserving Inductor's generated fp32 population mean, centered-variance, libdevice.rsqrt eps=1e-5, affine epilogue, and final contiguous `[25088, 512]` clone/view layout, whereas Inductor already emits the same fused normalization and layout-store region through its generic persistent-reduction scheduler; Inductor cannot materially improve this local repro today because the remaining work is dominated by required activation/residual/affine reads, one hidden-size-512 reduction, rsqrt, and output stores rather than an avoidable intermediate; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless broader norm-template, launch, or memory-traffic changes move both paths."""
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


EPS = 1.0e-5
EXPECTED_BATCH = 128
EXPECTED_HEIGHT = 14
EXPECTED_WIDTH = 14
EXPECTED_HIDDEN = 512
EXPECTED_WINDOW_H = 7
EXPECTED_WINDOW_W = 7
EXPECTED_ROWS = EXPECTED_BATCH * EXPECTED_HEIGHT * EXPECTED_WIDTH
BLOCK_H = 512
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:
    from torch._inductor.runtime.triton_helpers import libdevice

    # Inductor lowered correction=0 var_mean for this static hidden size to
    # fp32 mean plus centered second moment, followed by libdevice.rsqrt.
    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 1}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["total_rows", "hidden"],
    )
    @triton.jit
    def _swin_window_layernorm_kernel(
        residual_ptr,
        addmm_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        hidden: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        source_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, BLOCK)
        row_mask = source_rows < total_rows
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]

        spatial_size: tl.constexpr = height * width
        batch = source_rows // spatial_size
        spatial = source_rows - batch * spatial_size
        src_h = spatial // width
        src_w = spatial - src_h * width

        window_grid_w: tl.constexpr = width // window_w
        window_h_idx = src_h // window_h
        inner_h = src_h - window_h_idx * window_h
        window_w_idx = src_w // window_w
        inner_w = src_w - window_w_idx * window_w
        out_rows = (
            (((batch * (height // window_h) + window_h_idx) * window_grid_w + window_w_idx)
             * window_h + inner_h)
            * window_w + inner_w
        )

        input_offsets = source_rows[:, None] * hidden + cols[None, :]
        residual = tl.load(
            residual_ptr + input_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        addmm = tl.load(
            addmm_ptr + input_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
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

        x = residual + addmm
        x_for_mean = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_mean, axis=1)[:, None].to(tl.float32) / hidden
        centered_for_var = x - mean
        variance = (
            tl.sum(tl.where(mask, centered_for_var * centered_for_var, 0.0), axis=1)[:, None].to(tl.float32)
            / hidden
        )
        centered = x - mean
        invstd = libdevice.rsqrt(variance + eps)
        normalized = centered * invstd
        scaled = normalized * weight[None, :]
        output = scaled + bias[None, :]

        output_offsets = out_rows[:, None] * hidden + cols[None, :]
        tl.store(out_ptr + output_offsets, output, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected_shape}")
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {expected_dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        addmm,
        residual,
        weight,
        bias,
        shape0_raw,
        shape1_raw,
        shape2_raw,
        shape3_raw,
        shape4_raw,
        shape5_raw,
    ) = inputs

    addmm = _require_tensor(
        "addmm_79",
        addmm,
        (EXPECTED_ROWS, EXPECTED_HIDDEN),
        torch.float32,
    )
    residual = _require_tensor(
        "view_542",
        residual,
        (EXPECTED_BATCH, EXPECTED_HEIGHT * EXPECTED_WIDTH, EXPECTED_HIDDEN),
        torch.float32,
    )
    weight = _require_tensor("arg301_1", weight, (EXPECTED_HIDDEN,), torch.float32)
    bias = _require_tensor("arg302_1", bias, (EXPECTED_HIDDEN,), torch.float32)

    if not (residual.device == weight.device == bias.device == addmm.device):
        raise ValueError("all tensor inputs must be on the same device")

    shape0 = _shape_tuple(shape0_raw)
    shape1 = _shape_tuple(shape1_raw)
    shape2 = _shape_tuple(shape2_raw)
    shape3 = _shape_tuple(shape3_raw)
    shape4 = _shape_tuple(shape4_raw)
    shape5 = _shape_tuple(shape5_raw)

    expected_shape0 = (EXPECTED_BATCH, EXPECTED_HEIGHT * EXPECTED_WIDTH, EXPECTED_HIDDEN)
    expected_shape1 = (EXPECTED_BATCH, EXPECTED_HEIGHT, EXPECTED_WIDTH, EXPECTED_HIDDEN)
    expected_shape2 = (
        EXPECTED_BATCH,
        EXPECTED_HEIGHT // EXPECTED_WINDOW_H,
        EXPECTED_WINDOW_H,
        EXPECTED_WIDTH // EXPECTED_WINDOW_W,
        EXPECTED_WINDOW_W,
        EXPECTED_HIDDEN,
    )
    expected_shape3 = (-1, EXPECTED_WINDOW_H, EXPECTED_WINDOW_W, EXPECTED_HIDDEN)
    expected_shape4 = (-1, EXPECTED_WINDOW_H * EXPECTED_WINDOW_W, EXPECTED_HIDDEN)
    expected_shape5 = (EXPECTED_ROWS, EXPECTED_HIDDEN)

    expected_shapes = (
        expected_shape0,
        expected_shape1,
        expected_shape2,
        expected_shape3,
        expected_shape4,
        expected_shape5,
    )
    actual_shapes = (shape0, shape1, shape2, shape3, shape4, shape5)
    for index, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes), start=4):
        if actual != expected:
            raise ValueError(f"shape parameter {index} is {actual}, expected {expected}")

    return addmm, residual, weight, bias, shape0, shape1, shape2, shape3, shape4, shape5


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    addmm, residual, weight, bias, shape0, shape1, shape2, shape3, shape4, shape5 = _validate_inputs(inputs)
    x = torch.ops.aten.reshape.default(addmm, shape0)
    x = torch.ops.aten.add.Tensor(residual, x)
    x = torch.ops.aten.reshape.default(x, shape1)
    variance, mean = torch.ops.aten.var_mean.correction(x, [3], correction=0, keepdim=True)
    x = torch.ops.aten.sub.Tensor(x, mean)
    x = torch.ops.aten.mul.Tensor(
        x,
        torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS)),
    )
    x = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(x, weight), bias)
    x = torch.ops.aten.reshape.default(x, shape2)
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.clone.default(x, memory_format=torch.contiguous_format)
    x = torch.ops.aten.reshape.default(x, shape3)
    x = torch.ops.aten.reshape.default(x, shape4)
    return torch.ops.aten.reshape.default(x, shape5)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same 10 inputs as Repro.forward() and returns
    the same single float32 contiguous `[25088, 512]` output.
    """
    addmm, residual, weight, bias, _shape0, shape1, shape2, _shape3, _shape4, shape5 = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    _batch, height, width, hidden = shape1
    _batch2, _grid_h, window_h, _grid_w, window_w, _hidden = shape2
    total_rows, out_hidden = shape5
    if out_hidden != hidden:
        raise ValueError(f"output hidden {out_hidden} does not match input hidden {hidden}")

    output = torch.empty_strided(
        shape5,
        (hidden, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(total_rows, meta["ROW_BLOCK"]),)
    _swin_window_layernorm_kernel[grid](
        residual,
        addmm,
        weight,
        bias,
        output,
        total_rows,
        height,
        width,
        hidden,
        window_h,
        window_w,
        EPS,
        BLOCK_H,
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
