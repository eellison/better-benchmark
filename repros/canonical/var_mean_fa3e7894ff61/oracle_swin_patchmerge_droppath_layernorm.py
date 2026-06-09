"""Gap diagnosis (classification: NEW_PATTERN): this oracle fuses the complete Swin drop-path residual patch-merge LayerNorm scope, including seeded Inductor RNG, residual add, 2x2 patch gather/contiguous flatten into 2048 channels, var_mean, rsqrt, affine, final [6272,2048] output, and sibling rsqrt/2048 output, whereas Inductor currently schedules the captured graph as separate RNG/drop-path pointwise, layout clone/reshape, generic var_mean LayerNorm, affine, final view, and side-output division kernels; Inductor cannot do this today because its scheduler/codegen pattern library has no patch-merge drop-path LayerNorm template that sinks the 2x2 gather/clone layout and seed-indexed broadcast mask into the normalization row schedule while preserving the saved inverse-std side output; the fix is NEW_PATTERN: add a guarded Swin patch-merge drop-path LayerNorm lowering that maps each output row/channel directly to source spatial rows, replays Inductor RNG, and emits both outputs from one row kernel."""
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
IN_CHANNELS = 512
PATCH_H = 7
PATCH_W = 7
OUT_CHANNELS = 2048
ROWS = BATCH * PATCH_H * PATCH_W
EPS = 1.0e-5
KEEP_PROB = 0.9086956530809402
SEED_INDEX = 41
BLOCK_C = 2048

ADDM_SHAPE = (BATCH * SPATIAL, IN_CHANNELS)
SEEDS_SHAPE = (46,)
RESIDUAL_SHAPE = (BATCH, SPATIAL, IN_CHANNELS)
AFFINE_SHAPE = (OUT_CHANNELS,)
RESHAPE_3D = (BATCH, SPATIAL, IN_CHANNELS)
RESHAPE_4D = (BATCH, HEIGHT, WIDTH, IN_CHANNELS)
PATCH_VIEW_SHAPE = (BATCH, PATCH_H, 2, PATCH_W, 2, IN_CHANNELS)
NORM_SHAPE = (BATCH, PATCH_H, PATCH_W, OUT_CHANNELS)
OUTPUT_SHAPE = (ROWS, OUT_CHANNELS)
OUTPUT_STRIDE = (OUT_CHANNELS, 1)
SIDE_SHAPE = (BATCH, PATCH_H, PATCH_W, 1)
SIDE_STRIDE = (PATCH_H * PATCH_W, PATCH_W, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_patchmerge_droppath_layernorm_kernel(
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
        height: tl.constexpr,
        width: tl.constexpr,
        in_channels: tl.constexpr,
        out_channels: tl.constexpr,
        patch_w: tl.constexpr,
        keep_prob: tl.constexpr,
        seed_index: tl.constexpr,
        eps: tl.constexpr,
        block_c: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_c)
        col_mask = cols < out_channels
        mask = col_mask

        patch_col = row % patch_w
        tmp = row // patch_w
        patch_row = tmp % (height // 2)
        batch = tmp // (height // 2)

        patch_lane = cols // in_channels
        channel = cols - patch_lane * in_channels
        src_h = patch_row * 2 + (patch_lane & 1)
        src_w = patch_col * 2 + (patch_lane >> 1)
        spatial = src_h * width + src_w
        src_row = batch * height * width + spatial

        seed = tl.load(seeds_ptr + seed_index)
        keep = tl.rand(seed, batch.to(tl.uint32)) < keep_prob
        drop_scale = tl.where(keep, 1.0 / keep_prob, 0.0)

        addmm = tl.load(
            addmm_ptr + src_row * addmm_s0 + channel * addmm_s1,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + batch * residual_s0 + spatial * residual_s1 + channel * residual_s2,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = residual + addmm * drop_scale

        x_for_reduce = tl.where(col_mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / out_channels
        centered = x - mean
        variance = tl.sum(tl.where(col_mask, centered * centered, 0.0), axis=0) / out_channels
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

        tl.store(out_ptr + row * out_s0 + cols * out_s1, y, mask=mask)
        tl.store(
            side_ptr + batch * side_s0 + patch_row * side_s1 + patch_col * side_s2,
            invstd / out_channels,
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
    addmm_t = _require_tensor("addmm_87", addmm, ADDM_SHAPE, torch.float32)
    seeds_t = _require_tensor("inductor_seeds_default", seeds, SEEDS_SHAPE, torch.int64)
    residual_t = _require_tensor("view_596", residual, RESIDUAL_SHAPE, torch.float32)
    weight_t = _require_tensor("primals_331", weight, AFFINE_SHAPE, torch.float32)
    bias_t = _require_tensor("primals_332", bias, AFFINE_SHAPE, torch.float32)

    device = addmm_t.device
    for name, tensor in (
        ("inductor_seeds_default", seeds_t),
        ("view_596", residual_t),
        ("primals_331", weight_t),
        ("primals_332", bias_t),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} is on {tensor.device}, expected {device}")

    expected_shapes = (
        RESHAPE_3D,
        RESHAPE_4D,
        PATCH_VIEW_SHAPE,
        NORM_SHAPE,
        OUTPUT_SHAPE,
    )
    actual_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2, shape3, shape4))
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
    ) = inputs
    reshape_default = torch.ops.aten.reshape.default(addmm, shape0)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default([BATCH, 1, 1], seed, "rand")
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    drop_scale = torch.ops.aten.div.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        KEEP_PROB,
    )
    dropped = torch.ops.aten.mul.Tensor(reshape_default, drop_scale)
    add_tensor = torch.ops.aten.add.Tensor(residual, dropped)
    reshape_default_1 = torch.ops.aten.reshape.default(add_tensor, shape1)
    reshape_default_2 = torch.ops.aten.reshape.default(reshape_default_1, shape2)
    permuted = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 4, 2, 5])
    contiguous = torch.ops.aten.clone.default(permuted, memory_format=torch.contiguous_format)
    reshape_default_3 = torch.ops.aten.reshape.default(contiguous, shape3)
    variance, mean = torch.ops.aten.var_mean.correction(
        reshape_default_3,
        [3],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(variance + EPS)
    y = (reshape_default_3 - mean) * invstd * weight + bias
    out = torch.ops.aten.reshape.default(y, shape4)
    return out, invstd / OUT_CHANNELS


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([46], i64), T([128, 196, 512], f32), T([2048], f32), T([2048], f32), S([128, 196, 512]), S([128, 14, 14, 512]), S([128, 7, 2, 7, 2, 512]), S([128, 7, 7, 2048]), S([6272, 2048]))")
def oracle_forward(inputs):
    """Run the complete seeded drop-path residual patch-merge LayerNorm scope."""
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
    _swin_patchmerge_droppath_layernorm_kernel[(ROWS,)](
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
        HEIGHT,
        WIDTH,
        IN_CHANNELS,
        OUT_CHANNELS,
        PATCH_W,
        KEEP_PROB,
        SEED_INDEX,
        EPS,
        BLOCK_C,
        num_warps=8,
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
