"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin drop-path residual, fp32 channel LayerNorm over hidden size 512, affine epilogue, cyclic spatial shift by 3 on both 14x14 axes, 7x7 window partition, final [25088, 512] flatten, and sibling `rsqrt / 512` output in one Triton row kernel, whereas Inductor lowers the captured graph as seeded RNG/drop-path pointwise work, generic residual LayerNorm, and separate index/permute/clone layout kernels; Inductor cannot do this today because its scheduler/codegen pattern library has no Swin drop-path shifted-window LayerNorm-partition template that sinks the cyclic gather and window-layout store into the normalization row schedule while preserving the saved inverse-std side output; the fix is NEW_PATTERN: add a guarded Swin drop-path shifted-window LayerNorm lowering that maps final window rows directly to source spatial rows, replays Inductor RNG in the row kernel, and emits both outputs."""
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
SEED_INDEX = 7
BLOCK_H = 512

ADDM_INPUT_SHAPE = (ROWS, HIDDEN)
SEEDS_SHAPE = (46,)
RESIDUAL_SHAPE = (BATCH, SPATIAL, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
RESHAPE_3D = (BATCH, SPATIAL, HIDDEN)
RESHAPE_4D = (BATCH, HEIGHT, WIDTH, HIDDEN)
WINDOW_VIEW_SHAPE = (BATCH, GRID_H, WINDOW_H, GRID_W, WINDOW_W, HIDDEN)
WINDOW_4D_SHAPE = (-1, WINDOW_H, WINDOW_W, HIDDEN)
WINDOW_3D_SHAPE = (-1, WINDOW_H * WINDOW_W, HIDDEN)
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
        addmm_s0: tl.constexpr,
        addmm_s1: tl.constexpr,
        residual_s0: tl.constexpr,
        residual_s1: tl.constexpr,
        residual_s2: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        side_s0: tl.constexpr,
        side_s1: tl.constexpr,
        side_s2: tl.constexpr,
        rows_total: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        hidden: tl.constexpr,
        window_h: tl.constexpr,
        window_w: tl.constexpr,
        grid_w: tl.constexpr,
        shift: tl.constexpr,
        keep_prob: tl.constexpr,
        seed_index: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = (row < rows_total) & (cols < hidden)

        inner_w = row % window_w
        tmp = row // window_w
        inner_h = tmp % window_h
        tmp = tmp // window_h
        window_col = tmp % grid_w
        tmp = tmp // grid_w
        window_row = tmp % (height // window_h)
        batch = tmp // (height // window_h)

        shifted_h = window_row * window_h + inner_h
        shifted_w = window_col * window_w + inner_w
        src_h = (shifted_h + shift) % height
        src_w = (shifted_w + shift) % width
        spatial = src_h * width + src_w
        src_row = batch * height * width + spatial

        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, batch.to(tl.uint32)) < keep_prob
        drop_scale = tl.where(keep, 1.0 / keep_prob, 0.0)

        addmm = tl.load(
            addmm_ptr + src_row * addmm_s0 + cols * addmm_s1,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + batch * residual_s0 + spatial * residual_s1 + cols * residual_s2,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + addmm * drop_scale

        x_for_reduce = tl.where(cols < hidden, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(cols < hidden, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=cols < hidden,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=cols < hidden,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + row * out_s0 + cols * out_s1, y, mask=mask)
        tl.store(
            side_ptr + batch * side_s0 + src_h * side_s1 + src_w * side_s2,
            invstd / hidden,
            mask=row < rows_total,
        )


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
    addmm_t = _require_tensor("addmm_19", addmm, ADDM_INPUT_SHAPE, torch.float32)
    seeds_t = _require_tensor("inductor_seeds_default", seeds, SEEDS_SHAPE, torch.int64)
    residual_t = _require_tensor("view_136", residual, RESIDUAL_SHAPE, torch.float32)
    weight_t = _require_tensor("primals_84", weight, AFFINE_SHAPE, torch.float32)
    bias_t = _require_tensor("primals_85", bias, AFFINE_SHAPE, torch.float32)

    device = addmm_t.device
    for name, tensor in (
        ("inductor_seeds_default", seeds_t),
        ("view_136", residual_t),
        ("primals_84", weight_t),
        ("primals_85", bias_t),
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
    actual_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2, shape3, shape4, shape5))
    if actual_shapes != expected_shapes:
        raise ValueError(f"shape parameters {actual_shapes} != {expected_shapes}")

    return addmm_t, seeds_t, residual_t, weight_t, bias_t


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
    reshape_default = torch.ops.aten.reshape.default(addmm, shape0)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default([BATCH, 1, 1], seed, "rand")
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    drop_scale = torch.ops.aten.div.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        KEEP_PROB,
    )
    add_tensor = torch.ops.aten.add.Tensor(residual, reshape_default * drop_scale)
    reshape_default_1 = torch.ops.aten.reshape.default(add_tensor, shape1)
    variance, mean = torch.ops.aten.var_mean.correction(
        reshape_default_1,
        [3],
        correction=0,
        keepdim=True,
    )
    rsqrt = torch.ops.aten.rsqrt.default(variance + EPS)
    y = (reshape_default_1 - mean) * rsqrt * weight + bias
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
    windows = torch.ops.aten.reshape.default(shifted, shape2)
    permuted = torch.ops.aten.permute.default(windows, [0, 1, 3, 2, 4, 5])
    contiguous = torch.ops.aten.clone.default(permuted, memory_format=torch.contiguous_format)
    out = torch.ops.aten.reshape.default(
        torch.ops.aten.reshape.default(
            torch.ops.aten.reshape.default(contiguous, shape3),
            shape4,
        ),
        shape5,
    )
    return out, rsqrt / HIDDEN


def oracle_forward(inputs):
    """Run the complete seeded drop-path residual LayerNorm shifted-window scope."""
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
    _swin_droppath_shifted_window_layernorm_kernel[(ROWS,)](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        output,
        side_output,
        addmm.stride(0),
        addmm.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        output.stride(0),
        output.stride(1),
        side_output.stride(0),
        side_output.stride(1),
        side_output.stride(2),
        ROWS,
        HEIGHT,
        WIDTH,
        HIDDEN,
        WINDOW_H,
        WINDOW_W,
        GRID_W,
        SHIFT,
        KEEP_PROB,
        SEED_INDEX,
        EPS,
        BLOCK_H,
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
