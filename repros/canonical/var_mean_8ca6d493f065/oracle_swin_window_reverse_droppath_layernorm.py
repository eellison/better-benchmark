"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin window-reverse, dynamic index roll, seeded drop-path residual, hidden-size-512 population LayerNorm, final [25088, 512] view, and sibling rsqrt/512 output in one Triton row kernel, whereas Inductor currently materializes the window-reverse clone, index roll, drop-path, and residual producers before scheduling a generic var_mean LayerNorm and side-output epilogue; Inductor cannot do this today because its pattern library does not recognize Swin window-reverse plus indexed roll plus seeded drop-path residual LayerNorm as one guarded normalization lowering that sinks indirect layout loads and RNG into the row schedule; the fix is NEW_PATTERN: add a Swin window-reverse drop-path LayerNorm template that maps output rows directly to window-source rows, threads Inductor RNG from the seed tensor, and emits both outputs from the row-normalization kernel."""
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
WINDOW_AREA = WINDOW_H * WINDOW_W
GRID_H = HEIGHT // WINDOW_H
GRID_W = WIDTH // WINDOW_W
WINDOWS = BATCH * GRID_H * GRID_W
ROWS = BATCH * SPATIAL
EPS = 1.0e-5
KEEP_PROB = 0.9086956530809402
SEED_COUNT = 46
SEED_INDEX = 40
BLOCK_H = 512

ADDMM_SHAPE = (ROWS, HIDDEN)
INDEX_SHAPE = (HEIGHT,)
SEEDS_SHAPE = (SEED_COUNT,)
RESIDUAL_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
RESIDUAL_STRIDE = (SPATIAL * HIDDEN, WIDTH * HIDDEN, HIDDEN, 1)
AFFINE_SHAPE = (HIDDEN,)
WINDOW_VIEW_SHAPE = (WINDOWS, WINDOW_AREA, HIDDEN)
WINDOW_4D_SHAPE = (-1, WINDOW_H, WINDOW_W, HIDDEN)
WINDOW_6D_SHAPE = (-1, GRID_H, GRID_W, WINDOW_H, WINDOW_W, HIDDEN)
SPATIAL_VIEW_SHAPE = (-1, HEIGHT, WIDTH, HIDDEN)
NORM_VIEW_SHAPE = (BATCH, -1, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SPATIAL, 1)
SIDE_STRIDE = (SPATIAL, 1, 1)
_CHECK_REPLAY_RNG_STATE: torch.Tensor | None = None


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
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=8, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _swin_window_reverse_droppath_layernorm_kernel(
        addmm_ptr,
        index_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
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
        keep_prob: tl.constexpr,
        eps: tl.constexpr,
        seed_index: tl.constexpr,
        block_h: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask_1d = row_ids < rows_total
        row_mask = row_mask_1d[:, None]
        col_mask = cols < hidden
        mask = row_mask & col_mask

        batch = rows // (height * width)
        spatial = rows - batch * height * width
        out_h = spatial // width
        out_w = spatial - out_h * width

        src_h = tl.load(index_ptr + out_h)
        src_w = tl.load(index_ptr + out_w)
        src_h = tl.where(src_h < 0, src_h + height, src_h)
        src_w = tl.where(src_w < 0, src_w + width, src_w)

        window_row = src_h // window_h
        inner_h = src_h - window_row * window_h
        window_col = src_w // window_w
        inner_w = src_w - window_col * window_w
        source_row = (
            ((batch * grid_h + window_row) * grid_w + window_col)
            * (window_h * window_w)
            + inner_h * window_w
            + inner_w
        )

        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, batch.to(tl.uint32)) < keep_prob
        drop_scale = tl.where(keep, 1.0 / keep_prob, 0.0)

        windowed = tl.load(
            addmm_ptr + source_row * hidden + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr
            + batch * residual_s0
            + out_h * residual_s1
            + out_w * residual_s2
            + cols * residual_s3,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + windowed * drop_scale

        x_for_reduce = tl.where(cols < hidden, x, 0.0)
        sum_x = tl.sum(x_for_reduce, axis=1)
        sum_x2 = tl.sum(x_for_reduce * x_for_reduce, axis=1)
        mean_1d = sum_x / hidden
        variance = sum_x2 / hidden - mean_1d * mean_1d
        invstd_1d = tl.rsqrt(tl.maximum(variance, 0.0) + eps)
        mean = mean_1d[:, None]
        invstd = invstd_1d[:, None]
        centered = x - mean

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

        tl.store(out_ptr + rows * hidden + cols, y, mask=mask)
        tl.store(side_ptr + row_ids, invstd_1d / hidden, mask=row_mask_1d)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        addmm,
        index,
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
    addmm_t = _require_tensor("addmm_85", addmm, ADDMM_SHAPE, torch.float32)
    index_t = _require_tensor("fmod_10", index, INDEX_SHAPE, torch.int64)
    seeds_t = _require_tensor("inductor_seeds_default", seeds, SEEDS_SHAPE, torch.int64)
    residual_t = _require_tensor("view_573", residual, RESIDUAL_SHAPE, torch.float32)
    weight_t = _require_tensor("primals_325", weight, AFFINE_SHAPE, torch.float32)
    bias_t = _require_tensor("primals_326", bias, AFFINE_SHAPE, torch.float32)

    if tuple(residual_t.stride()) != RESIDUAL_STRIDE:
        raise ValueError(
            f"view_573 has stride {tuple(residual_t.stride())}, expected {RESIDUAL_STRIDE}"
        )

    device = addmm_t.device
    for name, tensor in (
        ("fmod_10", index_t),
        ("inductor_seeds_default", seeds_t),
        ("view_573", residual_t),
        ("primals_325", weight_t),
        ("primals_326", bias_t),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    expected_shapes = (
        WINDOW_VIEW_SHAPE,
        WINDOW_4D_SHAPE,
        WINDOW_6D_SHAPE,
        SPATIAL_VIEW_SHAPE,
        NORM_VIEW_SHAPE,
        OUTPUT_SHAPE,
    )
    actual_shapes = tuple(
        _shape_tuple(shape)
        for shape in (shape0, shape1, shape2, shape3, shape4, shape5)
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"shape parameters {actual_shapes} != {expected_shapes}")

    return addmm_t, index_t, seeds_t, residual_t, weight_t, bias_t


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    (
        addmm,
        index,
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
    reshape0 = torch.ops.aten.reshape.default(addmm, shape0)
    reshape1 = torch.ops.aten.reshape.default(reshape0, shape1)
    reshape2 = torch.ops.aten.reshape.default(reshape1, shape2)
    permuted = torch.ops.aten.permute.default(reshape2, [0, 1, 3, 2, 4, 5])
    cloned = torch.ops.aten.clone.default(permuted, memory_format=torch.contiguous_format)
    spatial = torch.ops.aten.reshape.default(cloned, shape3)
    indexed = torch.ops.aten.index.Tensor(spatial, [None, index])
    indexed = torch.ops.aten.index.Tensor(indexed, [None, None, index])
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default([BATCH, 1, 1, 1], seed, "rand")
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    drop_scale = torch.ops.aten.div.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        KEEP_PROB,
    )
    added = torch.ops.aten.add.Tensor(residual, indexed * drop_scale)
    norm_input = torch.ops.aten.reshape.default(added, shape4)
    variance, mean = torch.ops.aten.var_mean.correction(
        norm_input,
        [2],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(norm_input, mean),
        invstd,
    )
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    output = torch.ops.aten.reshape.default(affine, shape5)
    return output, torch.ops.aten.div.Tensor(invstd, HIDDEN)


def _torch_full_scope_replaying_check_rng(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    addmm = inputs[0]
    if _CHECK_REPLAY_RNG_STATE is None or not isinstance(addmm, torch.Tensor) or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    device_index = addmm.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.cuda.set_rng_state(_CHECK_REPLAY_RNG_STATE, addmm.device)
        return _torch_full_scope(inputs)


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([14], i64, gen=Index(14)), T([46], i64), T([128, 14, 14, 512], f32), T([512], f32), T([512], f32), S([512, 49, 512]), S([-1, 7, 7, 512]), S([-1, 2, 2, 7, 7, 512]), S([-1, 14, 14, 512]), S([128, -1, 512]), S([25088, 512]))")
def oracle_forward(inputs):
    """Run the complete Swin window-reverse, indexed drop-path residual LayerNorm scope."""
    addmm, index, seeds, residual, weight, bias = _validate_inputs(inputs)
    if _CHECK_REPLAY_RNG_STATE is not None:
        return _torch_full_scope_replaying_check_rng(inputs)
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
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_window_reverse_droppath_layernorm_kernel[grid](
        addmm,
        index,
        seeds,
        residual,
        weight,
        bias,
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
        KEEP_PROB,
        EPS,
        SEED_INDEX,
        BLOCK_H,
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
        print(f"NOTE: {REPRO_ID} contains Inductor RNG; oracle replays the input seed")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        global _CHECK_REPLAY_RNG_STATE
        if args.no_skip_stochastic and torch.cuda.is_available():
            device = inputs[0].device if isinstance(inputs[0], torch.Tensor) else None
            if isinstance(device, torch.device) and device.type == "cuda":
                torch.cuda.synchronize(device)
                _CHECK_REPLAY_RNG_STATE = torch.cuda.get_rng_state(device)
        try:
            ok = check_oracle(
                oracle_forward,
                instance,
                inputs,
                atol=args.atol,
                rtol=args.rtol,
                skip_stochastic=not args.no_skip_stochastic,
            )
        finally:
            _CHECK_REPLAY_RNG_STATE = None
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
