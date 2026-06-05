"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Blenderbot dropout plus layer-norm backward tail by streaming `mm_2`, the dropout mask/source producer, and the residual once per row block, keeping the hidden-dimension layer-norm sums local while accumulating all three returned `[2560]` reductions and materializing the returned `[2560, 4096]` transpose view, whereas Inductor currently schedules the mask/dropout source, hidden-dimension sums, column reductions, dropout epilogue, permute side output, and downstream output sum as separate generic kernels with materialized intermediates; Inductor cannot do this today because its scheduler/codegen cannot form a coordinated cross-axis multi-output reduction with row-local reductions, column partial finalization, and a dependent materialized side output from the same producer; the fix is COOPERATIVE_SPLIT_K: add a split-K dependent multi-output reduction template that shares row scalars, column partials, and direct side-output stores across this normalization/dropout backward pattern."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
CHANNELS = 2560
DROPOUT_SCALE = 1.1111111111111112
INV_CHANNELS = 1.0 / CHANNELS
ROW_SPLIT = 8
ROW_BLOCK_C = 4096
FINALIZE_BLOCK_C = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _row_block_full_scope_kernel(
        x_ptr,
        weight_ptr,
        source_base_ptr,
        keep_ptr,
        residual_source_ptr,
        shift_ptr,
        scale_ptr,
        residual_ptr,
        out_base_ptr,
        partial_x_m2_ptr,
        partial_x_ptr,
        partial_out_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        INV_CHANNELS_: tl.constexpr,
        ROW_SPLIT_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        col_mask = cols < CHANNELS_
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        acc_x_m2 = tl.zeros([BLOCK_C], dtype=tl.float32)
        acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)
        acc_out = tl.zeros([BLOCK_C], dtype=tl.float32)

        for local_row in tl.range(0, ROW_SPLIT_):
            row = row_block * ROW_SPLIT_ + local_row
            row_mask = row < ROWS_
            mask = row_mask & col_mask
            offsets = row * CHANNELS_ + cols

            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            source_base = tl.load(source_base_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
            residual_source = tl.load(residual_source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            shift = tl.load(shift_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

            source = residual_source + keep * source_base * DROPOUT_SCALE_
            m2 = (source - shift) * scale
            weighted = x * weight

            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(mask, weighted * m2, 0.0), axis=0)

            ln_tail = scale * INV_CHANNELS_ * (
                weighted * CHANNELS_ - row_sum - m2 * row_dot
            )
            out = (residual + ln_tail) * keep * DROPOUT_SCALE_

            tl.store(out_base_ptr + offsets, out, mask=mask)
            acc_x_m2 += tl.where(mask, x * m2, 0.0)
            acc_x += tl.where(mask, x, 0.0)
            acc_out += tl.where(mask, out, 0.0)

        partial_offsets = row_block * CHANNELS_ + cols
        tl.store(partial_x_m2_ptr + partial_offsets, acc_x_m2, mask=col_mask)
        tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)
        tl.store(partial_out_ptr + partial_offsets, acc_out, mask=col_mask)


    @triton.jit
    def _finalize_column_partials_kernel(
        partial_x_m2_ptr,
        partial_x_ptr,
        partial_out_ptr,
        out_x_m2_ptr,
        out_x_ptr,
        out_sum_out_ptr,
        NUM_ROW_BLOCKS: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_ROW_BLOCKS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
        col_mask = cols < CHANNELS_
        block_mask = blocks < NUM_ROW_BLOCKS
        mask = block_mask[:, None] & col_mask[None, :]
        offsets = blocks[:, None] * CHANNELS_ + cols[None, :]

        x_m2 = tl.load(partial_x_m2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        tl.store(out_x_m2_ptr + cols, tl.sum(x_m2, axis=0), mask=col_mask)
        tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
        tl.store(out_sum_out_ptr + cols, tl.sum(out, axis=0), mask=col_mask)


def _validate_inputs(inputs):
    if len(inputs) != 12:
        raise ValueError(f"expected 12 inputs, got {len(inputs)}")

    (
        mm_2,
        arg6_1,
        arg20_1,
        arg21_1,
        arg1_1,
        arg22_1,
        arg23_1,
        arg28_1,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
    ) = inputs

    expected = {
        "mm_2": (mm_2, (ROWS, CHANNELS), torch.float32),
        "arg6_1": (arg6_1, (CHANNELS,), torch.float32),
        "arg20_1": (arg20_1, (ROWS, CHANNELS), torch.float32),
        "arg21_1": (arg21_1, (BATCH, SEQ, CHANNELS), torch.bool),
        "arg1_1": (arg1_1, (BATCH, SEQ, CHANNELS), torch.float32),
        "arg22_1": (arg22_1, (BATCH, SEQ, 1), torch.float32),
        "arg23_1": (arg23_1, (BATCH, SEQ, 1), torch.float32),
        "arg28_1": (arg28_1, (BATCH, SEQ, CHANNELS), torch.float32),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )

    if list(shape_param_0) != [BATCH, SEQ, CHANNELS]:
        raise ValueError(f"unexpected first view shape parameter: {shape_param_0}")
    if list(shape_param_1) != [BATCH, SEQ, CHANNELS]:
        raise ValueError(f"unexpected second view shape parameter: {shape_param_1}")
    if list(shape_param_2) != [ROWS, CHANNELS]:
        raise ValueError(f"unexpected third view shape parameter: {shape_param_2}")
    if list(shape_param_3) != [CHANNELS]:
        raise ValueError(f"unexpected final view shape parameter: {shape_param_3}")


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
        raise RuntimeError("triton is not available")

    _validate_inputs(inputs)
    (
        mm_2,
        arg6_1,
        arg20_1,
        arg21_1,
        arg1_1,
        arg22_1,
        arg23_1,
        arg28_1,
        *_shape_params,
    ) = inputs

    num_row_blocks = triton.cdiv(ROWS, ROW_SPLIT)
    partial_x_m2 = torch.empty((num_row_blocks, CHANNELS), device=mm_2.device, dtype=torch.float32)
    partial_x = torch.empty((num_row_blocks, CHANNELS), device=mm_2.device, dtype=torch.float32)
    partial_out = torch.empty((num_row_blocks, CHANNELS), device=mm_2.device, dtype=torch.float32)
    out_base = torch.empty((ROWS, CHANNELS), device=mm_2.device, dtype=torch.float32)

    _row_block_full_scope_kernel[(num_row_blocks,)](
        mm_2,
        arg6_1,
        arg20_1,
        arg21_1,
        arg1_1,
        arg22_1,
        arg23_1,
        arg28_1,
        out_base,
        partial_x_m2,
        partial_x,
        partial_out,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        INV_CHANNELS_=INV_CHANNELS,
        ROW_SPLIT_=ROW_SPLIT,
        BLOCK_C=ROW_BLOCK_C,
        num_warps=16,
    )

    out_x_m2 = torch.empty((CHANNELS,), device=mm_2.device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=mm_2.device, dtype=torch.float32)
    out_sum_out = torch.empty((CHANNELS,), device=mm_2.device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(CHANNELS, FINALIZE_BLOCK_C),)](
        partial_x_m2,
        partial_x,
        partial_out,
        out_x_m2,
        out_x,
        out_sum_out,
        NUM_ROW_BLOCKS=num_row_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=FINALIZE_BLOCK_C,
        num_warps=8,
    )

    return out_x_m2, out_x, out_base.permute(1, 0), out_sum_out


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
