"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Swin dropout-residual LayerNorm plus shifted-window partition scope, including seed-index 39 Inductor dropout, the fp32 hidden-size-512 population var_mean, affine LayerNorm, the live rsqrt/512 side output, and the final contiguous [25088,512] window layout, while inlining the tiny batch RNG and writing the affine values directly into the post-permute/clone layout; Inductor currently materializes the RNG vector, stores mean/variance side buffers, writes an indexed [128,14,14,512] intermediate, then launches a separate layout clone; Inductor cannot do this today because the norm-template and pointwise scheduler do not sink the structured index plus window-partition clone into the normalization epilogue or inline the batchwise RNG producer; the fix is SCHEDULER_FUSION: teach the normalization/pointwise scheduler to fuse seeded batch dropout, affine LayerNorm side outputs, advanced indexing, and fixed window-partition layout stores into one scheduled epilogue around the row-statistics kernel."""
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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BATCH = 128
HEIGHT = 14
WIDTH = 14
WINDOW = 7
WINDOW_BLOCKS = 2
HIDDEN = 512
SPATIAL = HEIGHT * WIDTH
ROWS = BATCH * SPATIAL
OUTPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, SPATIAL, HIDDEN)
SIDE_SHAPE = (BATCH, HEIGHT, WIDTH, 1)
FOLD_SHAPE = (BATCH, WINDOW_BLOCKS, WINDOW, WINDOW_BLOCKS, WINDOW, HIDDEN)
WINDOW_SHAPE = (-1, WINDOW, WINDOW, HIDDEN)
WINDOW_FLAT_SHAPE = (-1, WINDOW * WINDOW, HIDDEN)
SEED_COUNT = 46
SEED_INDEX = 39
KEEP_PROB = 0.9130434766411781
DROPOUT_SCALE = 1.0952380971809903
EPS = 1.0e-5
BLOCK_H = 512
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _swin_dropout_layernorm_stats_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        mean_ptr,
        side_ptr,
        hidden: tl.constexpr,
        spatial: tl.constexpr,
        seed_index: tl.constexpr,
        keep_prob: tl.constexpr,
        dropout_scale: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        rows = row_ids[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask_1d = row_ids < total_rows
        row_mask = row_mask_1d[:, None]
        offsets = rows * hidden + cols

        addmm = tl.load(
            addmm_ptr + offsets,
            mask=row_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + offsets,
            mask=row_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        seed = tl.load(seeds_ptr + seed_index)
        batch = row_ids // spatial
        random = tl.rand(seed, batch.to(tl.uint32))
        scale = tl.where(random < keep_prob, dropout_scale, 0.0)[:, None]
        x = residual + addmm * scale

        x_for_reduce = tl.where(row_mask, x, 0.0)
        sum_x = tl.sum(x_for_reduce, axis=1)
        mean_1d = sum_x / hidden
        centered = tl.where(row_mask, x - mean_1d[:, None], 0.0)
        variance = tl.sum(centered * centered, axis=1) / hidden
        invstd = tl.rsqrt(variance + eps)

        tl.store(mean_ptr + row_ids, mean_1d, mask=row_mask_1d)
        tl.store(side_ptr + row_ids, invstd / hidden, mask=row_mask_1d)

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 2}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 4}, num_warps=4, num_stages=3),
            triton.Config({"ROW_BLOCK": 8}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _swin_window_affine_kernel(
        fmod_ptr,
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        side_ptr,
        out_ptr,
        hidden: tl.constexpr,
        spatial: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        window: tl.constexpr,
        seed_index: tl.constexpr,
        keep_prob: tl.constexpr,
        dropout_scale: tl.constexpr,
        block_h: tl.constexpr,
        total_rows: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        out_rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        cols = tl.arange(0, block_h)[None, :]
        row_mask_1d = out_rows < total_rows
        row_mask = row_mask_1d[:, None]
        out_offsets = out_rows[:, None] * hidden + cols

        batch = out_rows // spatial
        row_in_batch = out_rows - batch * spatial
        inner_w = row_in_batch % window
        tmp0 = row_in_batch // window
        inner_h = tmp0 % window
        tmp1 = tmp0 // window
        block_w = tmp1 % 2
        block_h = tmp1 // 2
        indexed_h_pos = block_h * window + inner_h
        indexed_w_pos = block_w * window + inner_w

        raw_h = tl.load(fmod_ptr + indexed_h_pos, mask=row_mask_1d, other=0)
        raw_w = tl.load(fmod_ptr + indexed_w_pos, mask=row_mask_1d, other=0)
        src_h = tl.where(raw_h < 0, raw_h + height, raw_h)
        src_w = tl.where(raw_w < 0, raw_w + width, raw_w)
        tl.device_assert((0 <= src_h) & (src_h < height), "index out of bounds: 0 <= src_h < 14")
        tl.device_assert((0 <= src_w) & (src_w < width), "index out of bounds: 0 <= src_w < 14")

        source_rows = batch * spatial + src_h * width + src_w
        source_offsets = source_rows[:, None] * hidden + cols

        addmm = tl.load(
            addmm_ptr + source_offsets,
            mask=row_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        residual = tl.load(
            residual_ptr + source_offsets,
            mask=row_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, batch.to(tl.uint32))
        scale = tl.where(random < keep_prob, dropout_scale, 0.0)[:, None]
        x = residual + addmm * scale

        mean = tl.load(mean_ptr + source_rows, mask=row_mask_1d, other=0.0)[:, None]
        invstd = tl.load(side_ptr + source_rows, mask=row_mask_1d, other=0.0)[:, None] * hidden
        weight = tl.load(weight_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(bias_ptr + cols, eviction_policy="evict_last").to(tl.float32)
        out = (x - mean) * invstd * weight + bias

        tl.store(out_ptr + out_offsets, out, mask=row_mask)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    addmm = _require_tensor("addmm_83", inputs[0], OUTPUT_SHAPE, torch.float32)
    seeds = _require_tensor("inductor_seeds_default", inputs[1], (SEED_COUNT,), torch.int64)
    residual = _require_tensor("view_568", inputs[2], RESIDUAL_SHAPE, torch.float32)
    weight = _require_tensor("primals_316", inputs[3], (HIDDEN,), torch.float32)
    bias = _require_tensor("primals_317", inputs[4], (HIDDEN,), torch.float32)
    fmod = _require_tensor("fmod_8", inputs[5], (HEIGHT,), torch.int64)

    expected_shapes = (
        RESIDUAL_SHAPE,
        SIDE_SHAPE[:-1] + (HIDDEN,),
        FOLD_SHAPE,
        WINDOW_SHAPE,
        WINDOW_FLAT_SHAPE,
        OUTPUT_SHAPE,
    )
    for index, expected in enumerate(expected_shapes, start=6):
        if _shape_tuple(inputs[index]) != expected:
            raise ValueError(f"unexpected shape parameter {index}: {inputs[index]!r}")

    device = addmm.device
    for tensor in (seeds, residual, weight, bias, fmod):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")
    return addmm, seeds, residual, weight, bias, fmod


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    addmm, seeds, residual, weight, bias, fmod = _validate_inputs(inputs)
    shape0 = _shape_tuple(inputs[6])
    shape1 = _shape_tuple(inputs[7])
    shape2 = _shape_tuple(inputs[8])
    shape3 = _shape_tuple(inputs[9])
    shape4 = _shape_tuple(inputs[10])
    shape5 = _shape_tuple(inputs[11])

    x = torch.ops.aten.reshape.default(addmm, shape0)
    seed = torch.ops.prims.inductor_lookup_seed.default(seeds, SEED_INDEX)
    random = torch.ops.prims.inductor_random.default([BATCH, 1, 1], seed, "rand")
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    dropped = x * keep.to(torch.float32) / KEEP_PROB
    x = residual + dropped
    x = torch.ops.aten.reshape.default(x, shape1)
    var, mean = torch.ops.aten.var_mean.correction(x, [3], correction=0, keepdim=True)
    invstd = torch.ops.aten.rsqrt.default(var + EPS)
    y = (x - mean) * invstd * weight + bias
    y = torch.ops.aten.index.Tensor(y, [None, fmod])
    y = torch.ops.aten.index.Tensor(y, [None, None, fmod])
    y = torch.ops.aten.reshape.default(y, shape2)
    y = torch.ops.aten.permute.default(y, [0, 1, 3, 2, 4, 5])
    y = torch.ops.aten.clone.default(y, memory_format=torch.contiguous_format)
    y = torch.ops.aten.reshape.default(y, shape3)
    y = torch.ops.aten.reshape.default(y, shape4)
    y = torch.ops.aten.reshape.default(y, shape5)
    return y, invstd / HIDDEN


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([46], i64), T([128, 196, 512], f32), T([512], f32), T([512], f32), T([14], i64, gen=Index(14)), S([128, 196, 512]), S([128, 14, 14, 512]), S([128, 2, 7, 2, 7, 512]), S([-1, 7, 7, 512]), S([-1, 49, 512]), S([25088, 512]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward stochastic LayerNorm and window-partition computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same two tensors: contiguous float32[25088,512] and float32[128,14,14,1].
    """
    addmm, seeds, residual, weight, bias, fmod = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        (HIDDEN, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    side = torch.empty_strided(
        SIDE_SHAPE,
        (SPATIAL, WIDTH, 1, 1),
        device=addmm.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (ROWS,),
        (1,),
        device=addmm.device,
        dtype=torch.float32,
    )

    stats_grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_dropout_layernorm_stats_kernel[stats_grid](
        addmm,
        seeds,
        residual,
        mean,
        side,
        hidden=HIDDEN,
        spatial=SPATIAL,
        seed_index=SEED_INDEX,
        keep_prob=KEEP_PROB,
        dropout_scale=DROPOUT_SCALE,
        eps=EPS,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )

    output_grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_window_affine_kernel[output_grid](
        fmod,
        addmm,
        seeds,
        residual,
        weight,
        bias,
        mean,
        side,
        output,
        hidden=HIDDEN,
        spatial=SPATIAL,
        height=HEIGHT,
        width=WIDTH,
        window=WINDOW,
        seed_index=SEED_INDEX,
        keep_prob=KEEP_PROB,
        dropout_scale=DROPOUT_SCALE,
        block_h=BLOCK_H,
        total_rows=ROWS,
    )
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

    # Report if stochastic ops detected in source
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")
        if args.no_skip_stochastic:
            print("NOTE: --no-skip-stochastic requested, so RNG-dependent values will be compared")

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
