"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin drop-path residual LayerNorm plus shifted-window partition scope in one Triton row-reduction kernel, including the provided seed-index-7 `[128,1,1]` stochastic keep mask, residual add, hidden-size-512 population `var_mean`, affine epilogue, cyclic shift by 3 on both 14x14 axes, 7x7 window partition/flatten, and sibling `rsqrt / 512` output, whereas Inductor currently lowers the input-seeded drop-path producer, normalization, side output, cyclic indexing, permute, clone, and final view through generic norm-template and layout scheduling; Inductor cannot do this today because its normalization/pattern scheduler does not recognize a Swin shifted-window drop-path LayerNorm as one semantic template that threads Inductor RNG through the row reduction and sinks the window-layout store plus inverse-std side output; the fix is NEW_PATTERN: add a guarded Swin shifted-window drop-path LayerNorm lowering that maps final window rows directly to source spatial rows, fuses the stochastic residual normalization, and emits both outputs from one schedule. Exact stochastic equality depends on matching Inductor's input-seeded RNG stream, so skipped stochastic checks should be treated as not_true_floor."""
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
HEIGHT = 14
WIDTH = 14
SPATIAL = HEIGHT * WIDTH
HIDDEN = 512
WINDOW_H = 7
WINDOW_W = 7
GRID_H = HEIGHT // WINDOW_H
GRID_W = WIDTH // WINDOW_W
ROWS = BATCH * SPATIAL
EPS = 1.0e-5
SHIFT = 3
KEEP_PROB = 0.9826086945831776
SEED_COUNT = 46
SEED_INDEX = 7
BLOCK_H = 512
ROW_BLOCK = 2
CLASSIFICATION = "NEW_PATTERN"
FLOOR_STATUS = "not_true_floor"

ADDMM_SHAPE = (ROWS, HIDDEN)
SEEDS_SHAPE = (SEED_COUNT,)
RESIDUAL_SHAPE = (BATCH, SPATIAL, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
VIEW_SHAPE_0 = (BATCH, SPATIAL, HIDDEN)
VIEW_SHAPE_1 = (BATCH, HEIGHT, WIDTH, HIDDEN)
WINDOW_VIEW_SHAPE = (BATCH, GRID_H, WINDOW_H, GRID_W, WINDOW_W, HIDDEN)
FLAT_WINDOW_SHAPE_0 = (-1, WINDOW_H, WINDOW_W, HIDDEN)
FLAT_WINDOW_SHAPE_1 = (-1, WINDOW_H * WINDOW_W, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, HEIGHT, WIDTH, 1)
SIDE_STRIDE = (HEIGHT * WIDTH, WIDTH, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_droppath_shifted_window_layernorm_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        residual_s0: tl.constexpr,
        residual_s1: tl.constexpr,
        residual_s2: tl.constexpr,
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

        inner_w = row_ids % window_w
        tmp = row_ids // window_w
        inner_h = tmp % window_h
        tmp = tmp // window_h
        window_col = tmp % grid_w
        tmp = tmp // grid_w
        window_row = tmp % grid_h
        batch = tmp // grid_h

        shifted_h = window_row * window_h + inner_h
        shifted_w = window_col * window_w + inner_w
        src_h = (shifted_h + shift) % height
        src_w = (shifted_w + shift) % width
        spatial = src_h * width + src_w
        src_row = batch * height * width + spatial

        addmm = tl.load(
            addmm_ptr + src_row * hidden + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr
            + batch * residual_s0
            + spatial * residual_s1
            + cols * residual_s2,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, batch.to(tl.uint32)) < keep_prob
        dropped = tl.where(keep, addmm / keep_prob, 0.0)
        x = residual + dropped

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = (tl.sum(x_for_reduce, axis=1) / hidden)[:, None]
        centered = x - mean
        variance = (
            tl.sum(tl.where(mask, centered * centered, 0.0), axis=1) / hidden
        )[:, None]
        invstd = tl.rsqrt(tl.maximum(variance, 0.0) + eps)

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

        tl.store(out_ptr + row_ids * hidden + cols, y, mask=mask)
        tl.store(side_ptr + src_row, invstd / hidden, mask=row_mask)


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
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_19", inputs[0], ADDMM_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds", inputs[1], SEEDS_SHAPE, torch.int64)
    residual = _require_tensor("view_134", inputs[2], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("arg83_1", inputs[3], AFFINE_SHAPE, torch.float32)
    bias = _require_tensor("arg84_1", inputs[4], AFFINE_SHAPE, torch.float32)

    expected_shapes = (
        VIEW_SHAPE_0,
        VIEW_SHAPE_1,
        WINDOW_VIEW_SHAPE,
        FLAT_WINDOW_SHAPE_0,
        FLAT_WINDOW_SHAPE_1,
        OUTPUT_SHAPE,
    )
    actual_shapes = tuple(_shape_tuple(shape) for shape in inputs[5:])
    if actual_shapes != expected_shapes:
        raise ValueError(f"shape parameters {actual_shapes} != {expected_shapes}")

    device = addmm.device
    for name, tensor in (
        ("inductor_seeds", seeds),
        ("view_134", residual),
        ("arg83_1", weight),
        ("arg84_1", bias),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return addmm, seeds, residual, weight, bias


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    (
        addmm,
        seeds,
        residual,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
        shape3,
        shape4,
        shape5,
    ) = inputs
    view_default = torch.ops.aten.view.default(addmm, shape0)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default([BATCH, 1, 1], seed, "rand")
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    scale = torch.ops.aten.div.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        KEEP_PROB,
    )
    dropped = torch.ops.aten.mul.Tensor(view_default, scale)
    x = torch.ops.aten.add.Tensor(residual, dropped)
    x4 = torch.ops.aten.view.default(x, shape1)
    variance, mean = torch.ops.aten.var_mean.correction(
        x4,
        [3],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    y = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(
            torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(x4, mean), invstd),
            weight,
        ),
        bias,
    )
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
    shifted = torch.ops.aten.index.Tensor(y, [None, index])
    shifted = torch.ops.aten.index.Tensor(shifted, [None, None, index])
    windows = torch.ops.aten.view.default(shifted, shape2)
    permuted = torch.ops.aten.permute.default(windows, [0, 1, 3, 2, 4, 5])
    contiguous = torch.ops.aten.clone.default(
        permuted,
        memory_format=torch.contiguous_format,
    )
    out = torch.ops.aten.view.default(
        torch.ops.aten.view.default(torch.ops.aten.view.default(contiguous, shape3), shape4),
        shape5,
    )
    return out, torch.ops.aten.div.Tensor(invstd, HIDDEN)


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([46], i64), T([128, 196, 512], f32), T([512], f32), T([512], f32), S([128, 196, 512]), S([128, 14, 14, 512]), S([128, 2, 7, 2, 7, 512]), S([-1, 7, 7, 512]), S([-1, 49, 512]), S([25088, 512]))")
def oracle_forward(inputs):
    """Run the complete Swin drop-path residual LayerNorm shifted-window scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    `(f32[25088, 512], f32[128, 14, 14, 1])` with the same contiguous strides.
    """
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
    _swin_droppath_shifted_window_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        side_output,
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
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
        num_warps=8,
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
        print("NOTE: exact stochastic equality may be skipped; floor status not_true_floor")

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
