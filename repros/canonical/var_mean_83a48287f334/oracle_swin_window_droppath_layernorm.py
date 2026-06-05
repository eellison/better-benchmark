"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin window-reverse, cyclic index, stochastic drop-path, residual add, hidden-size-128 LayerNorm, affine epilogue, final contiguous `[401408,128]` view, and sibling `rsqrt / 128` output in one shape-specialized Triton row kernel, whereas Inductor lowers the captured graph as generic window-layout/indexing, stochastic pointwise, residual add, normalization, clone, and final view work; Inductor cannot do this today because its pattern library does not recognize Swin window-reverse plus drop-path residual LayerNorm as one semantic lowering that sinks layout/index loads into the normalization schedule and preserves the saved inverse-std side output; the fix is NEW_PATTERN: add a guarded Swin window-reverse drop-path LayerNorm template that maps output rows directly to windowed source rows, threads Inductor RNG through the row kernel, and emits both outputs. Exact stochastic equality is skipped by the harness, so the floor status is not_true_floor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops

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
WINDOW_H = 7
WINDOW_W = 7
WINDOW_AREA = WINDOW_H * WINDOW_W
GRID_H = HEIGHT // WINDOW_H
GRID_W = WIDTH // WINDOW_W
WINDOWS = BATCH * GRID_H * GRID_W
ROWS = BATCH * SPATIAL
EPS = 1.0e-5
SHIFT = 53
KEEP_PROB = 0.9956521736457944
SEED_COUNT = 46
SEED_INDEX = 0
BLOCK_H = 128
ROW_BLOCK = 4
FLOOR_STATUS = "not_true_floor"

INPUT_SHAPE = (ROWS, HIDDEN)
IOTA_SHAPE = (HEIGHT,)
RESIDUAL_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
RESIDUAL_STRIDE = (ROWS, WIDTH, 1, SPATIAL)
WINDOW_VIEW_SHAPE = (WINDOWS, WINDOW_AREA, HIDDEN)
WINDOW_4D_SHAPE = (-1, WINDOW_H, WINDOW_W, HIDDEN)
WINDOW_6D_SHAPE = (-1, GRID_H, GRID_W, WINDOW_H, WINDOW_W, HIDDEN)
SPATIAL_VIEW_SHAPE = (-1, HEIGHT, WIDTH, HIDDEN)
NORM_VIEW_SHAPE = (BATCH, SPATIAL, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SPATIAL, 1)
SIDE_STRIDE = (SPATIAL, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_window_droppath_layernorm_kernel(
        addmm_ptr,
        iota_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        seeds_ptr,
        out_ptr,
        side_ptr,
        residual_s0: tl.constexpr,
        residual_s1: tl.constexpr,
        residual_s2: tl.constexpr,
        residual_s3: tl.constexpr,
        rows_total: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        hidden: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        grid_h: tl.constexpr,
        grid_w: tl.constexpr,
        shift: tl.constexpr,
        keep_prob: tl.constexpr,
        eps: tl.constexpr,
        seed_index: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids < rows_total
        col_mask = cols < hidden
        mask = row_mask & col_mask

        batch = row_ids // (height * width)
        spatial = row_ids - batch * height * width
        h = spatial // width
        w = spatial - h * width

        indexed_h = (tl.load(iota_ptr + h, mask=row_mask, other=0) + shift) % height
        indexed_w = (tl.load(iota_ptr + w, mask=row_mask, other=0) + shift) % width

        window_row = indexed_h // window_h
        inner_h = indexed_h - window_row * window_h
        window_col = indexed_w // window_w
        inner_w = indexed_w - window_col * window_w
        window_index = (batch * grid_h + window_row) * grid_w + window_col
        source_row = window_index * (window_h * window_w) + inner_h * window_w + inner_w

        addmm_offsets = source_row * hidden + cols
        residual_offsets = (
            batch * residual_s0
            + h * residual_s1
            + w * residual_s2
            + cols * residual_s3
        )
        windowed = tl.load(
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

        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, batch.to(tl.uint32)) < keep_prob
        dropped = tl.where(keep, windowed / keep_prob, 0.0)
        x = residual + dropped

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
        centered = x - mean
        variance = (
            tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        )[:, None]
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

        out_offsets = row_ids * hidden + cols
        tl.store(out_ptr + out_offsets, y, mask=mask)
        tl.store(side_ptr + row_ids, invstd / hidden, mask=row_mask)


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
    contiguous: bool = True,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    addmm, iota, residual, weight, bias, shape0, shape1, shape2, shape3, shape4 = inputs
    addmm_t = _require_tensor("addmm_5", addmm, INPUT_SHAPE, torch.float32)
    iota_t = _require_tensor("iota", iota, IOTA_SHAPE, torch.int64)
    residual_t = _require_tensor(
        "view_24",
        residual,
        RESIDUAL_SHAPE,
        torch.float32,
        contiguous=False,
    )
    weight_t = _require_tensor("arg28_1", weight, (HIDDEN,), torch.float32)
    bias_t = _require_tensor("arg29_1", bias, (HIDDEN,), torch.float32)

    if tuple(residual_t.stride()) != RESIDUAL_STRIDE:
        raise ValueError(
            f"view_24 has stride {tuple(residual_t.stride())}, expected {RESIDUAL_STRIDE}"
        )
    device = addmm_t.device
    for name, tensor in (
        ("iota", iota_t),
        ("view_24", residual_t),
        ("arg28_1", weight_t),
        ("arg29_1", bias_t),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    expected_shapes = (
        WINDOW_VIEW_SHAPE,
        WINDOW_4D_SHAPE,
        WINDOW_6D_SHAPE,
        SPATIAL_VIEW_SHAPE,
        NORM_VIEW_SHAPE,
    )
    actual_shapes = tuple(
        _shape_tuple(shape)
        for shape in (shape0, shape1, shape2, shape3, shape4)
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"shape parameters {actual_shapes} != {expected_shapes}")

    return addmm_t, iota_t, residual_t, weight_t, bias_t


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, iota, residual, weight, bias, shape0, shape1, shape2, shape3, shape4 = inputs
    view_default = torch.ops.aten.view.default(addmm, shape0)
    view_default_1 = torch.ops.aten.view.default(view_default, shape1)
    view_default_2 = torch.ops.aten.view.default(view_default_1, shape2)
    permute_default = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 2, 4, 5])
    clone_default = torch.ops.aten.clone.default(
        permute_default,
        memory_format=torch.contiguous_format,
    )
    view_default_3 = torch.ops.aten.view.default(clone_default, shape3)
    index = torch.ops.aten.fmod.Scalar(torch.ops.aten.add.Tensor(iota, SHIFT), HEIGHT)
    index_tensor = torch.ops.aten.index.Tensor(view_default_3, [None, index])
    index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, index])
    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, addmm.device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default([BATCH, 1, 1, 1], seed, "rand")
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    drop_scale = torch.ops.aten.div.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        KEEP_PROB,
    )
    add_tensor = torch.ops.aten.add.Tensor(residual, index_tensor_1 * drop_scale)
    view_default_4 = torch.ops.aten.view.default(add_tensor, shape4)
    variance, mean = torch.ops.aten.var_mean.correction(
        view_default_4,
        [2],
        correction=0,
        keepdim=True,
    )
    rsqrt = torch.ops.aten.rsqrt.default(variance + EPS)
    y = (view_default_4 - mean) * rsqrt * weight + bias
    contiguous = torch.ops.aten.clone.default(y, memory_format=torch.contiguous_format)
    return torch.ops.aten._unsafe_view.default(contiguous, list(OUTPUT_SHAPE)), rsqrt / HIDDEN


def oracle_forward(inputs):
    """Run the complete Swin window-reverse drop-path residual LayerNorm scope."""
    addmm, iota, residual, weight, bias = _validate_inputs(inputs)
    if triton is None:
        return _torch_full_scope(inputs)

    seeds = torch.ops.prims.inductor_seeds.default(SEED_COUNT, addmm.device)
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
    _swin_window_droppath_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        addmm,
        iota,
        residual,
        weight,
        bias,
        seeds,
        output,
        side_output,
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        residual.stride(3),
        ROWS,
        HEIGHT,
        WIDTH,
        HIDDEN,
        WINDOW_H,
        WINDOW_W,
        GRID_H,
        GRID_W,
        SHIFT,
        KEEP_PROB,
        EPS,
        SEED_INDEX,
        BLOCK_H,
        ROW_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return output, side_output


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


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
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
    if FLOOR_STATUS == "not_true_floor":
        print("NOTE: exact stochastic equality is not established; floor status not_true_floor")

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
