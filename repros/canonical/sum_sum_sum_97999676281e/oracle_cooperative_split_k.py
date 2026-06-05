"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full DeiT layer-norm-backward/projection return tuple by row-tiling the `[25344, 768]` producer, sharing each row's hidden-dimension reductions, writing the required non-contiguous `[768, 25344]` transpose side output, and cooperatively finalizing the two source column reductions plus the side-output column reduction, whereas Inductor currently schedules the row reductions, dependent pointwise layer-norm-backward epilogue, transpose side-output materialization, and sibling column reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local reduction scalars live while also emitting a layout-changing side output and compatible column partials; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer/finalizer schedule that fuses layer-norm-backward epilogues, writes view-equivalent transpose backing storage, and finalizes all sibling column reductions together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

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
TOKENS = 198
ROWS = BATCH * TOKENS
CHANNELS = 768
ROWS_PER_GROUP = 16
BLOCK_R = 2
FINALIZE_BLOCK_CHANNELS = 2


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _ln_tail_store_partials_kernel(
        mm_ptr,
        weight_ptr,
        rhs_ptr,
        scale_ptr,
        residual_ptr,
        side_base_ptr,
        partials_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        ROWS_PER_GROUP_: tl.constexpr,
        BLOCK_R_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        group = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C_)
        col_mask = cols < CHANNELS_
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        acc_x_rhs = tl.zeros((BLOCK_C_,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_C_,), dtype=tl.float32)
        acc_side = tl.zeros((BLOCK_C_,), dtype=tl.float32)

        for local_start in tl.range(0, ROWS_PER_GROUP_, BLOCK_R_):
            rows = group * ROWS_PER_GROUP_ + local_start + tl.arange(0, BLOCK_R_)
            row_mask = rows < ROWS_
            mask = row_mask[:, None] & col_mask[None, :]
            offsets = rows[:, None] * CHANNELS_ + cols[None, :]

            x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

            weighted = x * weight[None, :]
            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
            side = residual + scale[:, None] * (
                weighted * CHANNELS_ - row_sum[:, None] - rhs * row_dot[:, None]
            )

            tl.store(side_base_ptr + offsets, side, mask=mask)
            acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
            acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
            acc_side += tl.sum(tl.where(mask, side, 0.0), axis=0)

        partial_base = group * 3 * CHANNELS_ + cols
        tl.store(partials_ptr + partial_base, acc_x_rhs, mask=col_mask)
        tl.store(partials_ptr + partial_base + CHANNELS_, acc_x, mask=col_mask)
        tl.store(partials_ptr + partial_base + 2 * CHANNELS_, acc_side, mask=col_mask)

    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out_x_rhs_ptr,
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

        x_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(partials_ptr + offsets + CHANNELS_, mask=mask, other=0.0).to(
            tl.float32
        )
        side = tl.load(partials_ptr + offsets + 2 * CHANNELS_, mask=mask, other=0.0).to(
            tl.float32
        )

        tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
        tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
        tl.store(out_side_ptr + cols, tl.sum(side, axis=0), mask=col_mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")

    (
        mm_94,
        primals_13,
        mul_2,
        div_25,
        add_132,
        shape0,
        shape1,
        shape2,
    ) = inputs
    tensors = (mm_94, primals_13, mul_2, div_25, add_132)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("first five repro inputs must be tensors")

    if tuple(mm_94.shape) != (ROWS, CHANNELS):
        raise ValueError(f"mm_94 shape {tuple(mm_94.shape)} != {(ROWS, CHANNELS)}")
    if tuple(primals_13.shape) != (CHANNELS,):
        raise ValueError(
            f"primals_13 shape {tuple(primals_13.shape)} != {(CHANNELS,)}"
        )
    if tuple(mul_2.shape) != (BATCH, TOKENS, CHANNELS):
        raise ValueError(
            f"mul_2 shape {tuple(mul_2.shape)} != {(BATCH, TOKENS, CHANNELS)}"
        )
    if tuple(div_25.shape) != (BATCH, TOKENS, 1):
        raise ValueError(f"div_25 shape {tuple(div_25.shape)} != {(BATCH, TOKENS, 1)}")
    if tuple(add_132.shape) != (BATCH, TOKENS, CHANNELS):
        raise ValueError(
            f"add_132 shape {tuple(add_132.shape)} != {(BATCH, TOKENS, CHANNELS)}"
        )
    if any(value.dtype != torch.float32 for value in tensors):
        raise TypeError("all tensor inputs must be torch.float32")
    if list(shape0) != [BATCH, TOKENS, CHANNELS]:
        raise ValueError(f"unexpected first reshape shape param: {shape0!r}")
    if list(shape1) != [ROWS, CHANNELS] or list(shape2) != [CHANNELS]:
        raise ValueError(f"unexpected trailing shape params: {shape1!r}, {shape2!r}")

    return tensors


def _oracle_torch(inputs: tuple[Any, ...] | list[Any]):
    mm_94, primals_13, mul_2, div_25, add_132 = _validate_inputs(inputs)

    x = mm_94.reshape(BATCH, TOKENS, CHANNELS)
    weighted = x * primals_13
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * mul_2).sum(dim=2, keepdim=True)
    grad = div_25 * (weighted * CHANNELS - row_sum - mul_2 * row_dot)
    side = add_132 + grad
    side_flat = side.reshape(ROWS, CHANNELS)

    return (
        (x * mul_2).sum(dim=(0, 1)),
        x.sum(dim=(0, 1)),
        side_flat.permute(1, 0),
        side_flat.sum(dim=0),
    )


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same four outputs: two f32[768] source reductions, the f32[768,25344]
    transposed side output, and the final f32[768] side-output reduction.
    """
    mm_94, primals_13, mul_2, div_25, add_132 = _validate_inputs(inputs)
    if triton is None or mm_94.device.type != "cuda":
        return _oracle_torch(inputs)

    mm = mm_94.reshape(ROWS, CHANNELS)
    weight = primals_13
    rhs = mul_2.reshape(ROWS, CHANNELS)
    scale = div_25.reshape(ROWS)
    residual = add_132.reshape(ROWS, CHANNELS)
    if not (
        mm.is_contiguous()
        and weight.is_contiguous()
        and rhs.is_contiguous()
        and scale.is_contiguous()
        and residual.is_contiguous()
    ):
        raise ValueError("Triton oracle expects contiguous tensor inputs")

    device = mm.device
    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)
    block_c = triton.next_power_of_2(CHANNELS)

    out_x_rhs = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    side_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)
    out_side = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    partials = torch.empty((num_groups, 3, CHANNELS), device=device, dtype=torch.float32)

    _ln_tail_store_partials_kernel[(num_groups,)](
        mm,
        weight,
        rhs,
        scale,
        residual,
        side_base,
        partials,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        ROWS_PER_GROUP_=ROWS_PER_GROUP,
        BLOCK_R_=BLOCK_R,
        BLOCK_C_=block_c,
        num_warps=4,
    )
    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINALIZE_BLOCK_CHANNELS),)](
        partials,
        out_x_rhs,
        out_x,
        out_side,
        NUM_GROUPS=num_groups,
        CHANNELS_=CHANNELS,
        BLOCK_G_=triton.next_power_of_2(num_groups),
        BLOCK_C_=FINALIZE_BLOCK_CHANNELS,
        num_warps=8,
    )
    return out_x_rhs, out_x, side_base.permute(1, 0), out_side


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
