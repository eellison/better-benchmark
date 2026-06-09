"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin residual add, hidden-size-128 population LayerNorm, affine epilogue, cyclic +3 spatial shift, 7x7 window partition clone, final contiguous flatten, and sibling rsqrt/128 output in one row-reduction Triton kernel, whereas Inductor currently materializes the strided residual add and normalized tensor before separate generated-index shift and window-layout clone work; Inductor cannot do this today because its scheduler/codegen pattern library has no guarded Swin shifted-window residual LayerNorm template that sinks the strided producer, cyclic index loads, layout clone, and inverse-std side output into one normalization row schedule; the fix is NEW_PATTERN: add a Swin shifted-window residual LayerNorm lowering that maps final window rows directly to shifted source rows and emits both the affine output and saved inverse-std side output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
HEIGHT = 56
WIDTH = 56
SPATIAL = HEIGHT * WIDTH
HIDDEN = 128
ROWS = BATCH * SPATIAL
WINDOW_H = 7
WINDOW_W = 7
WINDOW_AREA = WINDOW_H * WINDOW_W
GRID_H = HEIGHT // WINDOW_H
GRID_W = WIDTH // WINDOW_W
SHIFT = 3
EPS = 1.0e-5
BLOCK_H = 128
ROW_BLOCK = 4

ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, SPATIAL, HIDDEN)
RESIDUAL_STRIDE = (ROWS, 1, SPATIAL)
AFFINE_SHAPE = (HIDDEN,)
VIEW_SHAPE_0 = (BATCH, SPATIAL, HIDDEN)
VIEW_SHAPE_1 = (BATCH, HEIGHT, WIDTH, HIDDEN)
VIEW_SHAPE_2 = (BATCH, GRID_H, WINDOW_H, GRID_W, WINDOW_W, HIDDEN)
VIEW_SHAPE_3 = (-1, WINDOW_H, WINDOW_W, HIDDEN)
VIEW_SHAPE_4 = (-1, WINDOW_AREA, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, HEIGHT, WIDTH, 1)
SIDE_STRIDE = (SPATIAL, WIDTH, 1, 1)
EXPECTED_SHAPE_PARAMS = (
    VIEW_SHAPE_0,
    VIEW_SHAPE_1,
    VIEW_SHAPE_2,
    VIEW_SHAPE_3,
    VIEW_SHAPE_4,
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

    @triton.jit
    def _swin_shifted_window_residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        side_ptr,
        residual_s0: tl.constexpr,
        residual_s1: tl.constexpr,
        residual_s2: tl.constexpr,
        side_s0: tl.constexpr,
        side_s1: tl.constexpr,
        side_s2: tl.constexpr,
        rows_total: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        hidden: tl.constexpr,
        grid_w: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        shift: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        out_rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = out_rows < rows_total
        col_mask = cols < hidden
        mask = row_mask & col_mask

        batch = out_rows // (height * width)
        within_batch = out_rows - batch * height * width
        window_id = within_batch // (window_h * window_w)
        window_pos = within_batch - window_id * window_h * window_w
        window_row = window_id // grid_w
        window_col = window_id - window_row * grid_w
        inner_h = window_pos // window_w
        inner_w = window_pos - inner_h * window_w

        logical_h = window_row * window_h + inner_h
        logical_w = window_col * window_w + inner_w
        source_h = (logical_h + shift) % height
        source_w = (logical_w + shift) % width
        source_spatial = source_h * width + source_w
        source_row = batch * height * width + source_spatial

        addmm = tl.load(
            addmm_ptr + source_row * hidden + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + batch * residual_s0 + source_spatial * residual_s1 + cols * residual_s2,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = addmm + residual

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
        centered = x - mean
        variance = (tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden)[:, None]
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
        y = centered * invstd * weight + bias

        tl.store(output_ptr + out_rows * hidden + cols, y, mask=mask)
        side_offsets = batch * side_s0 + source_h * side_s1 + source_w * side_s2
        tl.store(side_ptr + side_offsets, invstd / hidden, mask=row_mask)


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
    *,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    addmm = _require_tensor(
        "addmm_3",
        inputs[0],
        ADDMM_SHAPE,
        torch.float32,
        stride=OUTPUT_STRIDE,
    )
    residual = _require_tensor(
        "view_20",
        inputs[1],
        RESIDUAL_SHAPE,
        torch.float32,
        stride=RESIDUAL_STRIDE,
    )
    weight = _require_tensor(
        "arg19_1",
        inputs[2],
        AFFINE_SHAPE,
        torch.float32,
        stride=(1,),
    )
    bias = _require_tensor(
        "arg20_1",
        inputs[3],
        AFFINE_SHAPE,
        torch.float32,
        stride=(1,),
    )

    actual_shapes = tuple(_shape_tuple(shape) for shape in inputs[4:])
    if actual_shapes != EXPECTED_SHAPE_PARAMS:
        raise ValueError(f"shape parameters {actual_shapes} != {EXPECTED_SHAPE_PARAMS}")

    device = addmm.device
    for name, tensor in (
        ("view_20", residual),
        ("arg19_1", weight),
        ("arg20_1", bias),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return addmm, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, residual, weight, bias = _validate_inputs(inputs)
    shape0, shape1, shape2, shape3, shape4, shape5 = inputs[4:]
    viewed = torch.ops.aten.view.default(addmm, shape0)
    added = torch.ops.aten.add.Tensor(residual, viewed)
    spatial = torch.ops.aten.view.default(added, shape1)
    variance, mean = torch.ops.aten.var_mean.correction(
        spatial,
        [3],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(spatial, mean),
        invstd,
    )
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    index = torch.ops.aten.fmod.Scalar(
        torch.ops.aten.add.Tensor(
            torch.ops.prims.iota.default(
                HEIGHT,
                start=0,
                step=1,
                dtype=torch.int64,
                device=addmm.device,
                requires_grad=False,
            ),
            SHIFT,
        ),
        HEIGHT,
    )
    shifted = torch.ops.aten.index.Tensor(affine, [None, index])
    shifted = torch.ops.aten.index.Tensor(shifted, [None, None, index])
    windows = torch.ops.aten.view.default(shifted, shape2)
    permuted = torch.ops.aten.permute.default(windows, [0, 1, 3, 2, 4, 5])
    contiguous = torch.ops.aten.clone.default(permuted, memory_format=torch.contiguous_format)
    output = torch.ops.aten.view.default(
        torch.ops.aten.view.default(
            torch.ops.aten.view.default(contiguous, shape3),
            shape4,
        ),
        shape5,
    )
    return output, torch.ops.aten.div.Tensor(invstd, HIDDEN)


@oracle_impl(hardware="H100", shapes="(T([401408, 128], f32), T([128, 3136, 128], f32, stride=(401408, 1, 3136)), T([128], f32), T([128], f32), S([128, 3136, 128]), S([128, 56, 56, 128]), S([128, 8, 7, 8, 7, 128]), S([-1, 7, 7, 128]), S([-1, 49, 128]), S([401408, 128]))")
def oracle_forward(inputs):
    """Run the complete shifted residual LayerNorm plus Swin window-partition scope."""
    addmm, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    side_output = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    _swin_shifted_window_residual_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        addmm,
        residual,
        weight,
        bias,
        output,
        side_output,
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        side_output.stride(0),
        side_output.stride(1),
        side_output.stride(2),
        ROWS,
        HEIGHT,
        WIDTH,
        HIDDEN,
        GRID_W,
        WINDOW_H,
        WINDOW_W,
        SHIFT,
        EPS,
        BLOCK_H,
        ROW_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return output, side_output


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
