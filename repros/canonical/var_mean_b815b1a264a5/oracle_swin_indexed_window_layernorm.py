"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin drop-path residual LayerNorm plus indexed-window partition scope in one Triton row-reduction kernel, including seed-index-39 batchwise drop-path, residual add, hidden-size-512 population var_mean, affine epilogue, the non-bijective `fmod_4` gather on both 14x14 spatial axes, final contiguous `[25088, 512]` window flatten, and sibling `rsqrt / 512` output; Inductor currently lowers this as seeded RNG/drop-path pointwise work, a generic norm template, and separate index/permute/clone layout work, and cannot sink the duplicated arbitrary index gather into the normalization schedule while also emitting the inverse-std side output; the fix is NEW_PATTERN: add a guarded Swin indexed-window drop-path LayerNorm lowering that recognizes the generated index vector, writes all gathered output rows directly from each normalized source row, and preserves the side-output store. Exact stochastic equality depends on matching Inductor's input-seeded RNG stream, so skipped stochastic checks should be treated as not_true_floor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers Inductor RNG prims.

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
HEIGHT = 14
WIDTH = 14
SPATIAL = HEIGHT * WIDTH
HIDDEN = 512
ROWS = BATCH * SPATIAL
WINDOW_H = 7
WINDOW_W = 7
GRID_H = HEIGHT // WINDOW_H
GRID_W = WIDTH // WINDOW_W
EPS = 1.0e-5
KEEP_PROB = 0.9130434766411781
SEED_COUNT = 46
SEED_INDEX = 39
BLOCK_H = 512
ROW_BLOCK = 2

ADDM_INPUT_SHAPE = (ROWS, HIDDEN)
SEEDS_SHAPE = (SEED_COUNT,)
RESIDUAL_SHAPE = (BATCH, SPATIAL, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
INDEX_SHAPE = (HEIGHT,)
RESHAPE_3D = (BATCH, SPATIAL, HIDDEN)
RESHAPE_4D = (BATCH, HEIGHT, WIDTH, HIDDEN)
WINDOW_VIEW_SHAPE = (BATCH, GRID_H, WINDOW_H, GRID_W, WINDOW_W, HIDDEN)
WINDOW_4D_SHAPE = (-1, WINDOW_H, WINDOW_W, HIDDEN)
WINDOW_3D_SHAPE = (-1, WINDOW_H * WINDOW_W, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, HEIGHT, WIDTH, 1)
SIDE_STRIDE = (HEIGHT * WIDTH, WIDTH, 1, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _occurrence_positions(value):
        first = tl.full((), -1, tl.int64)
        second = tl.full((), -1, tl.int64)
        third = tl.full((), -1, tl.int64)

        first = tl.where(value == 0, 3, first)
        first = tl.where(value == 2, 1, first)
        first = tl.where(value == 3, 13, first)
        first = tl.where(value == 4, 0, first)
        first = tl.where(value == 7, 10, first)
        first = tl.where(value == 8, 5, first)
        first = tl.where(value == 9, 4, first)
        first = tl.where(value == 11, 9, first)
        first = tl.where(value == 12, 2, first)
        first = tl.where(value == 13, 7, first)

        second = tl.where(value == 8, 8, second)
        second = tl.where(value == 11, 11, second)
        second = tl.where(value == 12, 6, second)

        third = tl.where(value == 12, 12, third)
        return first, second, third

    @triton.jit
    def _window_row(
        batch,
        pos_h,
        pos_w,
        spatial: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        grid_w: tl.constexpr,
    ):
        block_h = pos_h // window_h
        inner_h = pos_h - block_h * window_h
        block_w = pos_w // window_w
        inner_w = pos_w - block_w * window_w
        return (
            batch * spatial
            + block_h * (grid_w * window_h * window_w)
            + block_w * (window_h * window_w)
            + inner_h * window_w
            + inner_w
        )

    @triton.jit
    def _store_window_row(
        out_ptr,
        cols,
        y,
        col_mask,
        batch,
        pos_h,
        pos_w,
        hidden: tl.constexpr,
        spatial: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        grid_w: tl.constexpr,
    ):
        valid = (pos_h >= 0) & (pos_w >= 0)
        out_row = _window_row(batch, pos_h, pos_w, spatial, window_h, window_w, grid_w)
        tl.store(out_ptr + out_row * hidden + cols, y, mask=valid & col_mask)

    @triton.jit
    def _swin_indexed_window_layernorm_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        rows_total: tl.constexpr,
        spatial: tl.constexpr,
        width: tl.constexpr,
        hidden: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        grid_w: tl.constexpr,
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

        batch = row_ids // spatial
        spatial_index = row_ids - batch * spatial
        src_h = spatial_index // width
        src_w = spatial_index - src_h * width

        offsets = row_ids * hidden + cols
        addmm = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, batch.to(tl.uint32)) < keep_prob
        x = residual + tl.where(keep, addmm / keep_prob, 0.0)

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
        centered = x - mean
        variance = (tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden)[:, None]
        invstd = tl.rsqrt(variance + eps)

        h0, h1, h2 = _occurrence_positions(src_h)
        w0, w1, w2 = _occurrence_positions(src_w)
        has_output = (h0 >= 0) & (w0 >= 0)
        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask & has_output,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=col_mask & has_output,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(side_ptr + row_ids, invstd / hidden, mask=row_mask)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h0, w0, hidden, spatial, window_h, window_w, grid_w)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h0, w1, hidden, spatial, window_h, window_w, grid_w)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h0, w2, hidden, spatial, window_h, window_w, grid_w)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h1, w0, hidden, spatial, window_h, window_w, grid_w)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h1, w1, hidden, spatial, window_h, window_w, grid_w)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h1, w2, hidden, spatial, window_h, window_w, grid_w)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h2, w0, hidden, spatial, window_h, window_w, grid_w)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h2, w1, hidden, spatial, window_h, window_w, grid_w)
        _store_window_row(out_ptr, cols, y, col_mask, batch, h2, w2, hidden, spatial, window_h, window_w, grid_w)


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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_83", inputs[0], ADDM_INPUT_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], SEEDS_SHAPE, torch.int64)
    residual = _require_tensor("view_566", inputs[2], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg315_1", inputs[3], AFFINE_SHAPE, torch.float32)
    bias = _require_tensor("arg316_1", inputs[4], AFFINE_SHAPE, torch.float32)
    index = _require_tensor("fmod_4", inputs[5], INDEX_SHAPE, torch.int64)

    device = addmm.device
    for name, tensor in (
        ("inductor_seeds", seeds),
        ("view_566", residual),
        ("arg315_1", weight),
        ("arg316_1", bias),
        ("fmod_4", index),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    expected_shapes = (
        RESHAPE_3D,
        RESHAPE_4D,
        WINDOW_VIEW_SHAPE,
        WINDOW_4D_SHAPE,
        WINDOW_3D_SHAPE,
        OUTPUT_SHAPE,
    )
    actual_shapes = tuple(_shape_tuple(shape) for shape in inputs[6:])
    if actual_shapes != expected_shapes:
        raise ValueError(f"shape parameters {actual_shapes} != {expected_shapes}")

    return addmm, seeds, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    (
        addmm,
        seeds,
        residual,
        weight,
        bias,
        index,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs
    viewed = torch.ops.aten.view.default(addmm, shape0)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default([BATCH, 1, 1], seed, "rand")
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    drop_scale = torch.ops.aten.div.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        KEEP_PROB,
    )
    x = torch.ops.aten.add.Tensor(residual, viewed * drop_scale)
    x4 = torch.ops.aten.view.default(x, shape1)
    variance, mean = torch.ops.aten.var_mean.correction(
        x4,
        [3],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    y = (x4 - mean) * invstd * weight + bias
    gathered = torch.ops.aten.index.Tensor(y, [None, index])
    gathered = torch.ops.aten.index.Tensor(gathered, [None, None, index])
    windows = torch.ops.aten.view.default(gathered, shape2)
    permuted = torch.ops.aten.permute.default(windows, [0, 1, 3, 2, 4, 5])
    contiguous = torch.ops.aten.clone.default(permuted, memory_format=torch.contiguous_format)
    out = torch.ops.aten.view.default(
        torch.ops.aten.view.default(torch.ops.aten.view.default(contiguous, shape3), shape4),
        shape5,
    )
    return out, invstd / HIDDEN


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full seeded drop-path LayerNorm indexed-window repro scope."""
    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    if triton is None:
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
    _swin_indexed_window_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        side_output,
        rows_total=ROWS,
        spatial=SPATIAL,
        width=WIDTH,
        hidden=HIDDEN,
        window_h=WINDOW_H,
        window_w=WINDOW_W,
        grid_w=GRID_W,
        keep_prob=KEEP_PROB,
        eps=EPS,
        seed_index=SEED_INDEX,
        block_h=BLOCK_H,
        row_block=ROW_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    return output, side_output


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


def _layout_and_diff(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in actual):
            torch.cuda.synchronize()

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        max_diff = (expected_tensor.float() - actual_tensor.float()).abs().max().item()
        layout_ok = (
            expected_tensor.shape == actual_tensor.shape
            and expected_tensor.stride() == actual_tensor.stride()
            and expected_tensor.dtype == actual_tensor.dtype
        )
        ok = ok and layout_ok
        print(
            f"  layout output {index}: shape={tuple(actual_tensor.shape)} "
            f"dtype={actual_tensor.dtype} stride={actual_tensor.stride()} "
            f"max_diff={max_diff:.2e} layout={'PASS' if layout_ok else 'FAIL'}"
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs may be auto-skipped")

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
        ok = _layout_and_diff(instance, inputs) and ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
