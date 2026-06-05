"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle streams each ALBERT GELU-tanh/layernorm-backward row group once, shares the row-local reductions across the derivative epilogue, materializes the returned transpose backing tensor, and cooperatively accumulates all three f32[128] column reductions, whereas Inductor currently schedules the GELU derivative algebra, row reductions, sibling column reductions, transpose side output, and final side-output reduction as separate generic kernels around materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K template for dependent row reductions that also writes a transpose side output and finalizes several compatible small column reductions; the fix is COOPERATIVE_SPLIT_K: add an Inductor multi-output reduction template that carries row-local reduction scalars into the epilogue, writes view-equivalent transpose backing storage, and reduces shared column partials without rereading the materialized side output."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
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


ROWS = 4096
CHANNELS = 128
ROWS_PER_GROUP = 16
BLOCK_R = 8
FINALIZE_BLOCK_C = 16
SQRT_2_OVER_PI = 0.7978845608028654
GELU_TANH_CUBE = 0.044715


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _row_group_gelu_ln_store_partials_kernel(
        mm_ptr,
        weight_ptr,
        gelu_in_ptr,
        mean_ptr,
        rstd_ptr,
        side_base_ptr,
        partials_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        SQRT_2_OVER_PI_: tl.constexpr,
        GELU_TANH_CUBE_: tl.constexpr,
        ROWS_PER_GROUP_: tl.constexpr,
        BLOCK_R_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        group = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C_)
        col_mask = cols < CHANNELS_
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        acc_x_normed = tl.zeros((BLOCK_C_,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_C_,), dtype=tl.float32)
        acc_side = tl.zeros((BLOCK_C_,), dtype=tl.float32)

        for local_start in tl.range(0, ROWS_PER_GROUP_, BLOCK_R_):
            rows = group * ROWS_PER_GROUP_ + local_start + tl.arange(0, BLOCK_R_)
            row_mask = rows < ROWS_
            mask = row_mask[:, None] & col_mask[None, :]
            offsets = rows[:, None] * CHANNELS_ + cols[None, :]

            x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            gelu_in = tl.load(gelu_in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
            rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

            inner = gelu_in + GELU_TANH_CUBE_ * gelu_in * gelu_in * gelu_in
            tanh_arg = SQRT_2_OVER_PI_ * inner
            tanh_val = libdevice.tanh(tanh_arg)
            one_plus_tanh = tanh_val + 1.0
            gelu = 0.5 * gelu_in * one_plus_tanh
            normed = (gelu - mean[:, None]) * rstd[:, None]

            weighted = x * weight[None, :]
            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(mask, weighted * normed, 0.0), axis=1)
            ln_grad = (rstd[:, None] / CHANNELS_) * (
                weighted * CHANNELS_ - row_sum[:, None] - normed * row_dot[:, None]
            )

            tanh_grad = 1.0 - tanh_val * tanh_val
            cubic_grad = 1.0 + 3.0 * GELU_TANH_CUBE_ * gelu_in * gelu_in
            gelu_grad = 0.5 * one_plus_tanh + (
                0.5 * gelu_in * tanh_grad * SQRT_2_OVER_PI_ * cubic_grad
            )
            side = ln_grad * gelu_grad

            tl.store(side_base_ptr + offsets, side, mask=mask)
            acc_x_normed += tl.sum(tl.where(mask, x * normed, 0.0), axis=0)
            acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
            acc_side += tl.sum(tl.where(mask, side, 0.0), axis=0)

        partial_base = group * 3 * CHANNELS_ + cols
        tl.store(partials_ptr + partial_base, acc_x_normed, mask=col_mask)
        tl.store(partials_ptr + partial_base + CHANNELS_, acc_x, mask=col_mask)
        tl.store(partials_ptr + partial_base + 2 * CHANNELS_, acc_side, mask=col_mask)

    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out_x_normed_ptr,
        out_x_ptr,
        out_side_ptr,
        NUM_GROUPS: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_G_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        groups = tl.arange(0, BLOCK_G_)
        col_mask = cols < CHANNELS_
        group_mask = groups < NUM_GROUPS
        mask = group_mask[:, None] & col_mask[None, :]
        offsets = groups[:, None] * 3 * CHANNELS_ + cols[None, :]

        x_normed = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(partials_ptr + offsets + CHANNELS_, mask=mask, other=0.0).to(tl.float32)
        side = tl.load(partials_ptr + offsets + 2 * CHANNELS_, mask=mask, other=0.0).to(tl.float32)

        tl.store(out_x_normed_ptr + cols, tl.sum(x_normed, axis=0), mask=col_mask)
        tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
        tl.store(out_side_ptr + cols, tl.sum(side, axis=0), mask=col_mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 9:
        raise ValueError(f"expected 9 inputs, got {len(inputs)}")

    mm, weight, gelu_in, mean, rstd, shape0, shape1, shape2, shape3 = inputs
    tensors = (mm, weight, gelu_in, mean, rstd)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("first five repro inputs must be tensors")
    if tuple(mm.shape) != (ROWS, CHANNELS):
        raise ValueError(f"mm shape {tuple(mm.shape)} != {(ROWS, CHANNELS)}")
    if tuple(weight.shape) != (CHANNELS,):
        raise ValueError(f"weight shape {tuple(weight.shape)} != {(CHANNELS,)}")
    if tuple(gelu_in.shape) != (ROWS, CHANNELS):
        raise ValueError(f"gelu_in shape {tuple(gelu_in.shape)} != {(ROWS, CHANNELS)}")
    if tuple(mean.shape) != (8, 512, 1) or tuple(rstd.shape) != (8, 512, 1):
        raise ValueError("mean and rstd must both have shape (8, 512, 1)")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("all tensor inputs must be torch.float32")
    if any(not x.is_cuda for x in tensors):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("all tensor inputs must be contiguous")
    if list(shape0) != [8, 512, CHANNELS] or list(shape1) != [8, 512, CHANNELS]:
        raise ValueError(f"unexpected leading view shape params: {shape0!r}, {shape1!r}")
    if list(shape2) != [ROWS, CHANNELS] or list(shape3) != [CHANNELS]:
        raise ValueError(f"unexpected trailing view shape params: {shape2!r}, {shape3!r}")

    return tensors


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same four outputs: two f32[128] source reductions, the f32[128,4096]
    transposed side output, and the final f32[128] side-output reduction.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    mm, weight, gelu_in, mean, rstd = _validate_inputs(inputs)
    device = mm.device
    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)

    out_x_normed = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    side_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)
    out_side = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    partials = torch.empty((num_groups, 3, CHANNELS), device=device, dtype=torch.float32)

    _row_group_gelu_ln_store_partials_kernel[(num_groups,)](
        mm,
        weight,
        gelu_in,
        mean,
        rstd,
        side_base,
        partials,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        SQRT_2_OVER_PI_=SQRT_2_OVER_PI,
        GELU_TANH_CUBE_=GELU_TANH_CUBE,
        ROWS_PER_GROUP_=ROWS_PER_GROUP,
        BLOCK_R_=BLOCK_R,
        BLOCK_C_=CHANNELS,
        num_warps=4,
    )
    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINALIZE_BLOCK_C),)](
        partials,
        out_x_normed,
        out_x,
        out_side,
        NUM_GROUPS=num_groups,
        CHANNELS_=CHANNELS,
        BLOCK_G_=triton.next_power_of_2(num_groups),
        BLOCK_C_=FINALIZE_BLOCK_C,
        num_warps=8,
    )
    return out_x_normed, out_x, side_base.permute(1, 0), out_side


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
