"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the
complete Swin window-reverse, iota-indexed cyclic shift, seeded drop-path
residual add, hidden-size-128 population LayerNorm, affine epilogue, final
contiguous `[401408,128]` view, and sibling `rsqrt * 1/128` output with
Inductor's generated two-stage stochastic schedule: one batch RNG kernel plus
one full-scope normalization/output kernel; Inductor already fuses the
layout/index/drop-path/residual producers into the reduction, so the remaining
cost is the required activation/affine traffic, one 128-wide row reduction,
RNG generation, `libdevice.rsqrt`, and output stores; the fix is BANDWIDTH_BOUND:
record this as an at-floor stochastic Swin LayerNorm case unless broader
normalization-template, RNG, launch, or memory-traffic work moves both paths."""
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
HEIGHT = 56
WIDTH = 56
SPATIAL = HEIGHT * WIDTH
HIDDEN = 128
ROWS = BATCH * SPATIAL
WINDOW = 7
WINDOW_AREA = WINDOW * WINDOW
GRID_H = HEIGHT // WINDOW
GRID_W = WIDTH // WINDOW
WINDOWS = BATCH * GRID_H * GRID_W
SEED_COUNT = 46
SEED_INDEX = 0
SHIFT = 53
KEEP_PROB = 0.9956521736457944
DROP_SCALE = 1.0043668124966625
EPS = 1.0e-5
INV_HIDDEN = 0.0078125
BLOCK_H = 128
ROW_BLOCK = 4

ADDMM_SHAPE = (ROWS, HIDDEN)
IOTA_SHAPE = (HEIGHT,)
RESIDUAL_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
RESIDUAL_STRIDE = (ROWS, WIDTH * HIDDEN, HIDDEN, 1)
AFFINE_SHAPE = (HIDDEN,)
WINDOW_VIEW_SHAPE = (WINDOWS, WINDOW_AREA, HIDDEN)
WINDOW_4D_SHAPE = (-1, WINDOW, WINDOW, HIDDEN)
WINDOW_6D_SHAPE = (-1, GRID_H, GRID_W, WINDOW, WINDOW, HIDDEN)
SPATIAL_VIEW_SHAPE = (-1, HEIGHT, WIDTH, HIDDEN)
NORM_VIEW_SHAPE = (BATCH, -1, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, SPATIAL, 1)
SIDE_STRIDE = (SPATIAL, 1, 1)
EXPECTED_SHAPE_PARAMS = (
    WINDOW_VIEW_SHAPE,
    WINDOW_4D_SHAPE,
    WINDOW_6D_SHAPE,
    SPATIAL_VIEW_SHAPE,
    NORM_VIEW_SHAPE,
    OUTPUT_SHAPE,
)

_CHECK_REPLAY_RNG_STATE: torch.Tensor | None = None


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _positive_mod(value, modulus: tl.constexpr):
        remainder = value % modulus
        return tl.where(remainder < 0, remainder + modulus, remainder)

    @triton.jit
    def _swin_batch_random_kernel(
        seeds_ptr,
        random_ptr,
        seed_index: tl.constexpr,
        batch_size: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < batch_size
        seed = tl.load(seeds_ptr + seed_index)
        values = tl.rand(seed, offsets.to(tl.uint32))
        tl.store(random_ptr + offsets, values, mask=mask)

    @triton.jit
    def _swin_window_reverse_droppath_layernorm_kernel(
        addmm_ptr,
        iota_ptr,
        residual_ptr,
        random_ptr,
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
        window: tl.constexpr,
        grid_w: tl.constexpr,
        keep_prob: tl.constexpr,
        drop_scale: tl.constexpr,
        shift: tl.constexpr,
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

        batch = row_ids // (height * width)
        spatial = row_ids - batch * height * width
        out_h = spatial // width
        out_w = spatial - out_h * width

        src_w = _positive_mod(tl.load(iota_ptr + out_w, mask=row_mask, other=0) + shift, width)
        src_h = _positive_mod(tl.load(iota_ptr + out_h, mask=row_mask, other=0) + shift, height)

        window_row = src_h // window
        window_col = src_w // window
        inner_h = src_h - window_row * window
        inner_w = src_w - window_col * window
        source_row = (
            ((batch * (height // window) + window_row) * grid_w + window_col)
            * (window * window)
            + inner_h * window
            + inner_w
        )

        projected = tl.load(
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

        random_value = tl.load(
            random_ptr + batch,
            mask=row_mask,
            other=1.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        keep = (random_value < keep_prob).to(tl.float32)
        scale = keep * drop_scale
        x = residual + projected * scale

        x_for_mean = tl.where(mask, x, 0.0)
        sum_x = tl.sum(x_for_mean, axis=1).to(tl.float32)[:, None]
        mean = sum_x / hidden
        centered_for_var = x - mean
        variance = (
            tl.sum(
                tl.where(mask, centered_for_var * centered_for_var, 0.0),
                axis=1,
            ).to(tl.float32)[:, None]
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
        y = centered * invstd
        y = y * weight
        y = y + bias
        side = invstd * inv_hidden

        tl.store(out_ptr + row_ids * hidden + cols, y, mask=mask)
        tl.store(side_ptr + row_ids, side, mask=row_mask)


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
    if contiguous and not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    (
        addmm,
        iota,
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

    addmm_t = _require_tensor("addmm_5", addmm, ADDMM_SHAPE, torch.float32)
    iota_t = _require_tensor("iota", iota, IOTA_SHAPE, torch.int64)
    residual_t = _require_tensor("view_25", residual, RESIDUAL_SHAPE, torch.float32)
    weight_t = _require_tensor("primals_29", weight, AFFINE_SHAPE, torch.float32)
    bias_t = _require_tensor("primals_30", bias, AFFINE_SHAPE, torch.float32)

    if tuple(residual_t.stride()) != RESIDUAL_STRIDE:
        raise ValueError(
            f"view_25 has stride {tuple(residual_t.stride())}, expected {RESIDUAL_STRIDE}"
        )

    actual_shapes = tuple(
        _shape_tuple(shape)
        for shape in (shape0, shape1, shape2, shape3, shape4, shape5)
    )
    if actual_shapes != EXPECTED_SHAPE_PARAMS:
        raise ValueError(f"shape parameters {actual_shapes} != {EXPECTED_SHAPE_PARAMS}")

    device = addmm_t.device
    for name, tensor in (
        ("iota", iota_t),
        ("view_25", residual_t),
        ("primals_29", weight_t),
        ("primals_30", bias_t),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    return addmm_t, iota_t, residual_t, weight_t, bias_t


def _make_inductor_seeds(device: torch.device) -> torch.Tensor:
    return torch.ops.prims.inductor_seeds.default(SEED_COUNT, device)


def _make_inductor_random_prims(device: torch.device) -> torch.Tensor:
    seeds = _make_inductor_seeds(device)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    return torch.ops.prims.inductor_random.default([BATCH, 1, 1, 1], seed, "rand").reshape(BATCH)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    (
        addmm,
        iota,
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

    reshape_default = torch.ops.aten.reshape.default(addmm, shape0)
    reshape_default_1 = torch.ops.aten.reshape.default(reshape_default, shape1)
    reshape_default_2 = torch.ops.aten.reshape.default(reshape_default_1, shape2)
    permute_default = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5])
    clone_default = torch.ops.aten.clone.default(
        permute_default,
        memory_format=torch.contiguous_format,
    )
    reshape_default_3 = torch.ops.aten.reshape.default(clone_default, shape3)
    shifted = torch.ops.aten.fmod.Scalar(torch.ops.aten.add.Tensor(iota, SHIFT), HEIGHT)
    index_tensor = torch.ops.aten.index.Tensor(reshape_default_3, [None, shifted])
    index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, shifted])
    random = _make_inductor_random_prims(addmm.device).reshape(BATCH, 1, 1, 1)
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    scale = torch.ops.aten.mul.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        DROP_SCALE,
    )
    dropped = torch.ops.aten.mul.Tensor(index_tensor_1, scale)
    added = torch.ops.aten.add.Tensor(residual, dropped)
    norm_input = torch.ops.aten.reshape.default(added, shape4)
    variance, mean = torch.ops.aten.var_mean.correction(
        norm_input,
        [2],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(norm_input, mean), invstd)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    output = torch.ops.aten.reshape.default(affine, shape5)
    side = torch.ops.aten.mul.Tensor(invstd, INV_HIDDEN)
    return output, side


def _launch_random_kernel(seeds: torch.Tensor, random: torch.Tensor) -> None:
    _swin_batch_random_kernel[(1,)](
        seeds,
        random,
        SEED_INDEX,
        BATCH,
        triton.next_power_of_2(BATCH),
        num_warps=4,
        num_stages=1,
    )


def _make_random_for_kernel(device: torch.device) -> torch.Tensor:
    random = torch.empty_strided(
        (BATCH,),
        (1,),
        device=device,
        dtype=torch.float32,
    )
    if _CHECK_REPLAY_RNG_STATE is None:
        seeds = _make_inductor_seeds(device)
        _launch_random_kernel(seeds, random)
        return random

    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.cuda.set_rng_state(_CHECK_REPLAY_RNG_STATE, device)
        seeds = _make_inductor_seeds(device)
        _launch_random_kernel(seeds, random)
    return random


def _launch_layernorm_kernel(
    addmm: torch.Tensor,
    iota: torch.Tensor,
    residual: torch.Tensor,
    random: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    side: torch.Tensor,
) -> None:
    _swin_window_reverse_droppath_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        addmm,
        iota,
        residual,
        random,
        weight,
        bias,
        output,
        side,
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        residual.stride(3),
        ROWS,
        HEIGHT,
        WIDTH,
        HIDDEN,
        WINDOW,
        GRID_W,
        KEEP_PROB,
        DROP_SCALE,
        SHIFT,
        EPS,
        INV_HIDDEN,
        BLOCK_H,
        ROW_BLOCK,
        num_warps=4,
        num_stages=3,
    )


def oracle_forward(inputs):
    """Run the complete Swin window-reverse drop-path residual LayerNorm scope."""
    addmm, iota, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or libdevice is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        SIDE_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    random = _make_random_for_kernel(addmm.device)
    _launch_layernorm_kernel(addmm, iota, residual, random, weight, bias, output, side)
    return output, side


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

    has_rng = has_stochastic_ops(REPRO_PATH) or _has_inductor_random()
    if has_rng:
        print(f"NOTE: {REPRO_ID} contains Inductor RNG; exact checks replay the pre-check CUDA RNG state")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        exact_replay = (
            _has_inductor_random()
            and torch.cuda.is_available()
            and isinstance(inputs[0], torch.Tensor)
            and inputs[0].is_cuda
        )
        global _CHECK_REPLAY_RNG_STATE
        if exact_replay:
            torch.cuda.synchronize(inputs[0].device)
            _CHECK_REPLAY_RNG_STATE = torch.cuda.get_rng_state(inputs[0].device)
        try:
            ok = check_oracle(
                oracle_forward,
                instance,
                inputs,
                atol=args.atol,
                rtol=args.rtol,
                skip_stochastic=False if exact_replay else not args.no_skip_stochastic,
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
