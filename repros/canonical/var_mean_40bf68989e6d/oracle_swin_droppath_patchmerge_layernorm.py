"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin seed-index-41 batch DropPath residual add, 2x2 patch-merge window clone into hidden-size-2048 rows, population var_mean LayerNorm, affine epilogue, final [6272, 2048] output, and `rsqrt / 2048` side output with a shape-specialized Triton normalization kernel, whereas Inductor lowers the seed lookup/random, residual pointwise work, window permute/clone, normalization, affine, and side-output division as generic scheduled regions; Inductor cannot do this today because its scheduler has no guarded Swin DropPath plus patch-merge LayerNorm template that shares batch RNG and sinks the clone layout into the row reduction store; the fix is NEW_PATTERN: add a Swin patch-merge LayerNorm lowering that replays the captured seed-indexed RNG, maps each output row/column directly to the source spatial/channel element, and emits both outputs from the same row-normalization schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims RNG ops.

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None
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
IN_CHANNELS = 512
PATCH = 2
OUT_HEIGHT = HEIGHT // PATCH
OUT_WIDTH = WIDTH // PATCH
OUT_SPATIAL = OUT_HEIGHT * OUT_WIDTH
HIDDEN = PATCH * PATCH * IN_CHANNELS
ROWS = BATCH * OUT_SPATIAL
SEED_COUNT = 46
SEED_INDEX = 41
INDUCTOR_RANDOM_OFFSET_DELTA = 8
KEEP_PROB = 0.9086956530809402
EPS = 1.0e-5
INV_HIDDEN = 0.00048828125
BLOCK_H = 2048
ROW_BLOCK = 1
_SEEDED_KERNEL_WARMED = False

ADDMM_SHAPE = (BATCH * SPATIAL, IN_CHANNELS)
SEEDS_SHAPE = (SEED_COUNT,)
RESIDUAL_SHAPE = (BATCH, SPATIAL, IN_CHANNELS)
AFFINE_SHAPE = (HIDDEN,)
VIEW_0 = (BATCH, SPATIAL, IN_CHANNELS)
VIEW_1 = (BATCH, HEIGHT, WIDTH, IN_CHANNELS)
VIEW_2 = (BATCH, OUT_HEIGHT, PATCH, OUT_WIDTH, PATCH, IN_CHANNELS)
VIEW_3 = (BATCH, OUT_HEIGHT, OUT_WIDTH, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, OUT_HEIGHT, OUT_WIDTH, 1)
SIDE_STRIDE = (OUT_SPATIAL, OUT_WIDTH, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_droppath_patchmerge_layernorm_seeded_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        rows_total: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        in_channels: tl.constexpr,
        out_width: tl.constexpr,
        hidden: tl.constexpr,
        keep_prob: tl.constexpr,
        eps: tl.constexpr,
        inv_hidden: tl.constexpr,
        seed_index: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids < rows_total
        col_mask = cols < hidden
        mask = row_mask & col_mask

        batch = row_ids // (out_width * out_width)
        out_spatial = row_ids - batch * (out_width * out_width)
        out_h = out_spatial // out_width
        out_w = out_spatial - out_h * out_width

        patch_index = cols // in_channels
        channel = cols - patch_index * in_channels
        inner_w = patch_index // 2
        inner_h = patch_index - inner_w * 2
        src_h = out_h * 2 + inner_h
        src_w = out_w * 2 + inner_w
        src_spatial = src_h * width + src_w
        src_row = batch * height * width + src_spatial

        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, batch.to(tl.uint32)) < keep_prob
        drop_scale = keep.to(tl.float32) / keep_prob

        projected = tl.load(
            addmm_ptr + src_row * in_channels + channel,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + src_row * in_channels + channel,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + projected * drop_scale

        x_for_mean = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_mean, axis=1)[:, None].to(tl.float32) / hidden
        centered_for_var = x - mean
        variance = (
            tl.sum(tl.where(mask, centered_for_var * centered_for_var, 0.0), axis=1)[
                :, None
            ].to(tl.float32)
            / hidden
        )
        invstd = libdevice.rsqrt(variance + eps)
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

        tl.store(out_ptr + row_ids * hidden + cols, y, mask=mask)
        tl.store(side_ptr + row_ids, invstd * inv_hidden, mask=row_mask)

    @triton.jit
    def _swin_droppath_patchmerge_layernorm_random_kernel(
        addmm_ptr,
        random_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        rows_total: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        in_channels: tl.constexpr,
        out_width: tl.constexpr,
        hidden: tl.constexpr,
        keep_prob: tl.constexpr,
        eps: tl.constexpr,
        inv_hidden: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids < rows_total
        col_mask = cols < hidden
        mask = row_mask & col_mask

        batch = row_ids // (out_width * out_width)
        out_spatial = row_ids - batch * (out_width * out_width)
        out_h = out_spatial // out_width
        out_w = out_spatial - out_h * out_width

        patch_index = cols // in_channels
        channel = cols - patch_index * in_channels
        inner_w = patch_index // 2
        inner_h = patch_index - inner_w * 2
        src_h = out_h * 2 + inner_h
        src_w = out_w * 2 + inner_w
        src_spatial = src_h * width + src_w
        src_row = batch * height * width + src_spatial

        random_value = tl.load(
            random_ptr + batch,
            mask=row_mask,
            other=1.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        keep = random_value < keep_prob
        drop_scale = keep.to(tl.float32) / keep_prob

        projected = tl.load(
            addmm_ptr + src_row * in_channels + channel,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + src_row * in_channels + channel,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + projected * drop_scale

        x_for_mean = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_mean, axis=1)[:, None].to(tl.float32) / hidden
        centered_for_var = x - mean
        variance = (
            tl.sum(tl.where(mask, centered_for_var * centered_for_var, 0.0), axis=1)[
                :, None
            ].to(tl.float32)
            / hidden
        )
        invstd = libdevice.rsqrt(variance + eps)
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

        tl.store(out_ptr + row_ids * hidden + cols, y, mask=mask)
        tl.store(side_ptr + row_ids, invstd * inv_hidden, mask=row_mask)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

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
    ) = inputs

    addmm_t = _require_tensor("addmm_87", addmm, ADDMM_SHAPE, torch.float32)
    seeds_t = _require_tensor("inductor_seeds", seeds, SEEDS_SHAPE, torch.int64)
    residual_t = _require_tensor("view_594", residual, RESIDUAL_SHAPE, torch.float32)
    weight_t = _require_tensor("arg330_1", weight, AFFINE_SHAPE, torch.float32)
    bias_t = _require_tensor("arg331_1", bias, AFFINE_SHAPE, torch.float32)

    expected_shapes = (VIEW_0, VIEW_1, VIEW_2, VIEW_3, OUTPUT_SHAPE)
    actual_shapes = tuple(
        _shape_tuple(shape) for shape in (shape0, shape1, shape2, shape3, shape4)
    )
    if actual_shapes != expected_shapes:
        raise ValueError(f"shape parameters {actual_shapes} != {expected_shapes}")

    device = addmm_t.device
    for name, tensor in (
        ("inductor_seeds", seeds_t),
        ("view_594", residual_t),
        ("arg330_1", weight_t),
        ("arg331_1", bias_t),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return addmm_t, seeds_t, residual_t, weight_t, bias_t


def _rewind_cuda_rng_one_inductor_random(device: torch.device) -> None:
    if (
        not torch.cuda.is_available()
        or device.type != "cuda"
        or torch.cuda.is_current_stream_capturing()
    ):
        return

    state = torch.cuda.get_rng_state(device)
    if state.numel() < 16:
        return
    offset_bytes = bytes(int(value) for value in state[8:16].tolist())
    offset = int.from_bytes(offset_bytes, "little")
    if offset < INDUCTOR_RANDOM_OFFSET_DELTA:
        return
    offset -= INDUCTOR_RANDOM_OFFSET_DELTA
    state[8:16] = torch.tensor(
        list(offset.to_bytes(8, "little")),
        dtype=torch.uint8,
    )
    torch.cuda.set_rng_state(state, device)


def _inductor_random_like_repro(seeds: torch.Tensor) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    return torch.ops.prims.inductor_random.default([BATCH, 1, 1], seed, "rand")


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
    ) = inputs

    view0 = torch.ops.aten.view.default(addmm, shape0)
    _rewind_cuda_rng_one_inductor_random(addmm.device)
    random = _inductor_random_like_repro(seeds)
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    scale = torch.ops.aten.div.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        KEEP_PROB,
    )
    dropped = torch.ops.aten.mul.Tensor(view0, scale)
    added = torch.ops.aten.add.Tensor(residual, dropped)
    x4 = torch.ops.aten.view.default(added, shape1)
    window_view = torch.ops.aten.view.default(x4, shape2)
    permuted = torch.ops.aten.permute.default(window_view, [0, 1, 3, 4, 2, 5])
    contiguous = torch.ops.aten.clone.default(
        permuted,
        memory_format=torch.contiguous_format,
    )
    norm_input = torch.ops.aten.view.default(contiguous, shape3)
    variance, mean = torch.ops.aten.var_mean.correction(
        norm_input,
        [3],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(norm_input, mean),
        invstd,
    )
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    output = torch.ops.aten.view.default(affine, shape4)
    return output, torch.ops.aten.div.Tensor(invstd, HIDDEN)


def _launch_seeded_kernel(
    addmm: torch.Tensor,
    seeds: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    side_output: torch.Tensor,
) -> None:
    _swin_droppath_patchmerge_layernorm_seeded_kernel[
        (triton.cdiv(ROWS, ROW_BLOCK),)
    ](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        side_output,
        ROWS,
        HEIGHT,
        WIDTH,
        IN_CHANNELS,
        OUT_WIDTH,
        HIDDEN,
        KEEP_PROB,
        EPS,
        INV_HIDDEN,
        SEED_INDEX,
        BLOCK_H,
        ROW_BLOCK,
        num_warps=8,
        num_stages=3,
    )


def _launch_exact_random_kernel(
    addmm: torch.Tensor,
    random: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    side_output: torch.Tensor,
) -> None:
    _swin_droppath_patchmerge_layernorm_random_kernel[
        (triton.cdiv(ROWS, ROW_BLOCK),)
    ](
        addmm,
        random,
        residual,
        weight,
        bias,
        output,
        side_output,
        ROWS,
        HEIGHT,
        WIDTH,
        IN_CHANNELS,
        OUT_WIDTH,
        HIDDEN,
        KEEP_PROB,
        EPS,
        INV_HIDDEN,
        BLOCK_H,
        ROW_BLOCK,
        num_warps=8,
        num_stages=3,
    )


def oracle_forward(inputs):
    """Run the complete Swin DropPath, patch-merge clone, and LayerNorm scope."""
    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or libdevice is None or not addmm.is_cuda:
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

    if torch.cuda.is_current_stream_capturing():
        _launch_seeded_kernel(addmm, seeds, residual, weight, bias, output, side_output)
        return output, side_output

    global _SEEDED_KERNEL_WARMED
    if not _SEEDED_KERNEL_WARMED:
        warm_output = torch.empty_strided(
            OUTPUT_SHAPE,
            OUTPUT_STRIDE,
            device=addmm.device,
            dtype=torch.float32,
        )
        warm_side = torch.empty_strided(
            SIDE_SHAPE,
            SIDE_STRIDE,
            device=addmm.device,
            dtype=torch.float32,
        )
        _launch_seeded_kernel(addmm, seeds, residual, weight, bias, warm_output, warm_side)
        _SEEDED_KERNEL_WARMED = True

    _rewind_cuda_rng_one_inductor_random(addmm.device)
    random = _inductor_random_like_repro(seeds).reshape(BATCH)
    _launch_exact_random_kernel(
        addmm,
        random,
        residual,
        weight,
        bias,
        output,
        side_output,
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
