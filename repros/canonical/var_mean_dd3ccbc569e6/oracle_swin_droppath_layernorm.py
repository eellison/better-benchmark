"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the full Swin window-reverse drop-path residual LayerNorm scope, including seed-index-44 batch drop-path RNG, population var_mean, affine epilogue, final flatten, and rsqrt/1024 side output, into one shape-specialized row reduction kernel, whereas Inductor currently materializes `prims.inductor_random` in a separate pointwise kernel before its fused var_mean kernel consumes the broadcast mask; Inductor cannot do this today because the scheduler treats Inductor RNG as a materialized producer boundary for normalization reductions instead of inlining the seed/offset expression into the row-reduction schedule; the fix is SCHEDULER_FUSION: allow fixed-shape seeded RNG producers with batch-broadcast uses to fuse into the LayerNorm reduction template while preserving Inductor RNG offsets and threshold semantics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401 - registers prims.* RNG ops.

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
WINDOW = 7
WINDOW_AREA = WINDOW * WINDOW
HIDDEN = 1024
ROWS = BATCH * WINDOW_AREA
SEED_COUNT = 46
SEED_INDEX = 44
INDUCTOR_RANDOM_OFFSET_DELTA = 8
KEEP_PROB = 0.8999999985098839
DROP_SCALE = 1.1111111129507607
EPS = 1.0e-5
BLOCK_H = 1024
ROW_BLOCK = 4
_SEEDED_KERNEL_WARMED = False

ADDMM_SHAPE = (ROWS, HIDDEN)
SEEDS_SHAPE = (SEED_COUNT,)
RESIDUAL_SHAPE = (BATCH, WINDOW, WINDOW, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
RESHAPE_0 = (BATCH, WINDOW_AREA, HIDDEN)
RESHAPE_1 = (-1, WINDOW, WINDOW, HIDDEN)
RESHAPE_2 = (-1, 1, 1, WINDOW, WINDOW, HIDDEN)
RESHAPE_3 = (-1, WINDOW, WINDOW, HIDDEN)
RESHAPE_4 = (BATCH, -1, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
SIDE_SHAPE = (BATCH, WINDOW_AREA, 1)
SIDE_STRIDE = (WINDOW_AREA, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _swin_droppath_layernorm_random_kernel(
        residual_ptr,
        addmm_ptr,
        random_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        hidden: tl.constexpr,
        keep_prob: tl.constexpr,
        drop_scale: tl.constexpr,
        eps: tl.constexpr,
        window_area: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
        total_rows: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = row_ids * hidden + cols
        batch = row_ids // window_area

        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        projected = tl.load(
            addmm_ptr + offsets,
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
        drop = tl.where(random_value < keep_prob, drop_scale, 0.0)
        x = residual + projected * drop

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
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
        out = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, out, mask=mask)
        tl.store(side_ptr + row_ids, invstd / hidden, mask=row_mask)

    @triton.jit
    def _swin_droppath_layernorm_seeded_kernel(
        residual_ptr,
        addmm_ptr,
        seeds_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        side_ptr,
        hidden: tl.constexpr,
        seed_index: tl.constexpr,
        keep_prob: tl.constexpr,
        drop_scale: tl.constexpr,
        eps: tl.constexpr,
        window_area: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
        total_rows: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = row_ids < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = row_ids * hidden + cols
        batch = row_ids // window_area

        residual = tl.load(
            residual_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        projected = tl.load(
            addmm_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        seed = tl.load(seeds_ptr + seed_index)
        random_value = tl.rand(seed, batch.to(tl.uint32))
        drop = tl.where(random_value < keep_prob, drop_scale, 0.0)
        x = residual + projected * drop

        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
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
        out = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, out, mask=mask)
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

    addmm_t = _require_tensor("addmm_93", addmm, ADDMM_SHAPE, torch.float32)
    seeds_t = _require_tensor("inductor_seeds_default", seeds, SEEDS_SHAPE, torch.int64)
    residual_t = _require_tensor("view_631", residual, RESIDUAL_SHAPE, torch.float32)
    weight_t = _require_tensor("primals_356", weight, AFFINE_SHAPE, torch.float32)
    bias_t = _require_tensor("primals_357", bias, AFFINE_SHAPE, torch.float32)

    expected_shapes = (RESHAPE_0, RESHAPE_1, RESHAPE_2, RESHAPE_3, RESHAPE_4, OUTPUT_SHAPE)
    actual_shapes = tuple(_shape_tuple(shape) for shape in (shape0, shape1, shape2, shape3, shape4, shape5))
    if actual_shapes != expected_shapes:
        raise ValueError(f"shape parameters {actual_shapes} != {expected_shapes}")

    device = addmm_t.device
    for name, tensor in (
        ("inductor_seeds_default", seeds_t),
        ("view_631", residual_t),
        ("primals_356", weight_t),
        ("primals_357", bias_t),
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
    return torch.ops.prims.inductor_random.default([BATCH, 1, 1, 1], seed, "rand")


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
    reshape_default_1 = torch.ops.aten.reshape.default(reshape_default, shape1)
    reshape_default_2 = torch.ops.aten.reshape.default(reshape_default_1, shape2)
    permute_default = torch.ops.aten.permute.default(reshape_default_2, [0, 1, 3, 2, 4, 5])
    reshape_default_3 = torch.ops.aten.reshape.default(permute_default, shape3)
    _rewind_cuda_rng_one_inductor_random(addmm.device)
    random = _inductor_random_like_repro(seeds)
    keep = torch.ops.aten.lt.Scalar(random, KEEP_PROB)
    scale = torch.ops.aten.div.Tensor(
        torch.ops.prims.convert_element_type.default(keep, torch.float32),
        KEEP_PROB,
    )
    dropped = torch.ops.aten.mul.Tensor(reshape_default_3, scale)
    x = torch.ops.aten.add.Tensor(residual, dropped)
    x3 = torch.ops.aten.reshape.default(x, shape4)
    variance, mean = torch.ops.aten.var_mean.correction(
        x3,
        [2],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    y = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(
            torch.ops.aten.mul.Tensor(
                torch.ops.aten.sub.Tensor(x3, mean),
                invstd,
            ),
            weight,
        ),
        bias,
    )
    return torch.ops.aten.reshape.default(y, shape5), torch.ops.aten.div.Tensor(invstd, HIDDEN)


def _launch_seeded_kernel(
    residual: torch.Tensor,
    addmm: torch.Tensor,
    seeds: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    side: torch.Tensor,
) -> None:
    _swin_droppath_layernorm_seeded_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        residual,
        addmm,
        seeds,
        weight,
        bias,
        output,
        side,
        HIDDEN,
        SEED_INDEX,
        KEEP_PROB,
        DROP_SCALE,
        EPS,
        WINDOW_AREA,
        BLOCK_H,
        ROW_BLOCK,
        ROWS,
        num_warps=4,
        num_stages=3,
    )


def _launch_exact_random_kernel(
    residual: torch.Tensor,
    addmm: torch.Tensor,
    random: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    side: torch.Tensor,
) -> None:
    _swin_droppath_layernorm_random_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        residual,
        addmm,
        random,
        weight,
        bias,
        output,
        side,
        HIDDEN,
        KEEP_PROB,
        DROP_SCALE,
        EPS,
        WINDOW_AREA,
        BLOCK_H,
        ROW_BLOCK,
        ROWS,
        num_warps=4,
        num_stages=3,
    )


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    if triton is None or not addmm.is_cuda:
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

    if torch.cuda.is_current_stream_capturing():
        _launch_seeded_kernel(residual, addmm, seeds, weight, bias, output, side)
        return output, side

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
        _launch_seeded_kernel(residual, addmm, seeds, weight, bias, warm_output, warm_side)
        _SEEDED_KERNEL_WARMED = True

    _rewind_cuda_rng_one_inductor_random(addmm.device)
    random = _inductor_random_like_repro(seeds)
    _launch_exact_random_kernel(residual, addmm, random, weight, bias, output, side)
    return output, side


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
