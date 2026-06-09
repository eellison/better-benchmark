"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full Swin dropout-residual-LayerNorm-spatial-mean scope with a two-kernel Triton schedule that carries only per-row mean and invstd/1024 between the normalization stats pass and the final 7x7 affine mean, whereas tuned Inductor already measures in the same envelope for the seeded dropout producer, row var_mean normalization, affine epilogue, invstd side output, and following spatial mean; Inductor cannot be assigned a confirmed local fusion or canonicalization gap from this repro because the remaining work is dominated by required RNG, activation and affine reads, fixed-width row reductions, side-output stores, and final mean stores rather than avoidable materialization; the fix is BANDWIDTH_BOUND: record this as at floor unless broader norm-template, memory-traffic, or launch-overhead improvements move the baseline. Exact stochastic equality is not established for prims.inductor_random, so the floor status is not_true_floor."""
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
    get_shape_key,
    has_stochastic_ops,
)


BATCH = 128
SPATIAL = 49
HEIGHT = 7
WIDTH = 7
HIDDEN = 1024
ROWS = BATCH * SPATIAL
SEED_INDEX = 45
SEED_COUNT = 46
KEEP_PROB = 0.8999999985098839
DROPOUT_SCALE = 1.1111111129507602
EPS = 1.0e-5
BLOCK_H = 1024
MEAN_BLOCK_C = 256

ADDMM_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (BATCH, SPATIAL, HIDDEN)
INPUT_VIEW_SHAPE = (BATCH, SPATIAL, HIDDEN)
NORM_VIEW_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
OUTPUT0_SHAPE = (BATCH, HIDDEN)
OUTPUT0_STRIDE = (HIDDEN, 1)
OUTPUT1_SHAPE = (BATCH, HEIGHT, WIDTH, 1)
OUTPUT1_STRIDE = (SPATIAL, WIDTH, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _row_stats_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        mean_ptr,
        invstd_div_ptr,
        hidden: tl.constexpr,
        spatial: tl.constexpr,
        seed_index: tl.constexpr,
        keep_prob: tl.constexpr,
        dropout_scale: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        batch = row // spatial
        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, batch.to(tl.uint32))
        keep_scale = tl.where(random < keep_prob, dropout_scale, 0.0)

        projected = tl.load(
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
        x = residual + projected * keep_scale
        x = tl.where(mask, x, 0.0)

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        tl.store(mean_ptr + row, mean)
        tl.store(invstd_div_ptr + row, invstd / hidden)

    @triton.jit
    def _affine_spatial_mean_kernel(
        addmm_ptr,
        seeds_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        invstd_div_ptr,
        out_ptr,
        hidden: tl.constexpr,
        spatial: tl.constexpr,
        seed_index: tl.constexpr,
        keep_prob: tl.constexpr,
        dropout_scale: tl.constexpr,
        block_c: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.program_id(1) * block_c + tl.arange(0, block_c)
        col_mask = cols < hidden

        seed = tl.load(seeds_ptr + seed_index)
        random = tl.rand(seed, batch.to(tl.uint32))
        keep_scale = tl.where(random < keep_prob, dropout_scale, 0.0)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        acc = tl.zeros((block_c,), tl.float32)

        for spatial_idx in tl.range(0, spatial):
            row = batch * spatial + spatial_idx
            offsets = row * hidden + cols
            projected = tl.load(
                addmm_ptr + offsets,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            residual = tl.load(
                residual_ptr + offsets,
                mask=col_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            mean = tl.load(mean_ptr + row).to(tl.float32)
            invstd = tl.load(invstd_div_ptr + row).to(tl.float32) * hidden
            x = residual + projected * keep_scale
            y = (x - mean) * invstd
            acc += y * weight + bias

        tl.store(out_ptr + batch * hidden + cols, acc / spatial, mask=col_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _expect_tensor(
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
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm = _expect_tensor("addmm_95", inputs[0], ADDMM_SHAPE, torch.float32)
    seeds = _expect_tensor("inductor_seeds", inputs[1], (SEED_COUNT,), torch.int64)
    residual = _expect_tensor("view_650", inputs[2], RESIDUAL_SHAPE, torch.float32)
    weight = _expect_tensor("arg361_1", inputs[3], (HIDDEN,), torch.float32)
    bias = _expect_tensor("arg362_1", inputs[4], (HIDDEN,), torch.float32)

    if _shape_tuple(inputs[5]) != INPUT_VIEW_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {inputs[5]!r}")
    if _shape_tuple(inputs[6]) != NORM_VIEW_SHAPE:
        raise ValueError(f"unexpected norm view shape parameter: {inputs[6]!r}")

    device = addmm.device
    for tensor in (seeds, residual, weight, bias):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")
    return addmm, seeds, residual, weight, bias


@oracle_impl(hardware="H100", shapes="(T([6272, 1024], f32), T([46], i64), T([128, 49, 1024], f32), T([1024], f32), T([1024], f32), S([128, 49, 1024]), S([128, 7, 7, 1024]))")
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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dropout_layernorm_spatial_mean.py")

    addmm, seeds, residual, weight, bias = _validate_inputs(inputs)
    mean = torch.empty_strided((ROWS,), (1,), device=addmm.device, dtype=torch.float32)
    invstd_div = torch.empty_strided(
        OUTPUT1_SHAPE,
        OUTPUT1_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        OUTPUT0_SHAPE,
        OUTPUT0_STRIDE,
        device=addmm.device,
        dtype=torch.float32,
    )

    _row_stats_kernel[(ROWS,)](
        addmm,
        seeds,
        residual,
        mean,
        invstd_div,
        hidden=HIDDEN,
        spatial=SPATIAL,
        seed_index=SEED_INDEX,
        keep_prob=KEEP_PROB,
        dropout_scale=DROPOUT_SCALE,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=1,
    )
    _affine_spatial_mean_kernel[(BATCH, triton.cdiv(HIDDEN, MEAN_BLOCK_C))](
        addmm,
        seeds,
        residual,
        weight,
        bias,
        mean,
        invstd_div,
        output,
        hidden=HIDDEN,
        spatial=SPATIAL,
        seed_index=SEED_INDEX,
        keep_prob=KEEP_PROB,
        dropout_scale=DROPOUT_SCALE,
        block_c=MEAN_BLOCK_C,
        num_warps=4,
        num_stages=1,
    )
    return output, invstd_div


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
