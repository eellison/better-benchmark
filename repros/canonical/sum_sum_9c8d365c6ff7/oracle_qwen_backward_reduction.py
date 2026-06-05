"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Qwen RMSNorm-backward-like `Repro.forward` tuple by row-tiling the `[2048, 1024]` producer, sharing each row's hidden-dimension sum for the returned transposed bf16 side output while accumulating partial per-channel sums for the sibling bf16 `[1024]` reduction, whereas Inductor currently schedules the row reduction, column reduction, bf16 casts, and transposed-layout side output as generic producer/consumer work over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that keeps row-local scalars, a live materialized side output, and compatible column partials in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: add a row-tiled multi-output reduction lowering that fuses the row-summary epilogue with side-output stores and finalizes channel partial reductions together."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


ROWS = 2048
CHANNELS = 1024
ROW_TILE = 16
BLOCK_CHANNELS = 1024
FINAL_BLOCK_TILES = 128
FINAL_BLOCK_CHANNELS = 16


if triton is not None:

    @triton.jit
    def _row_tile_kernel(
        mm_ptr,
        weight_ptr,
        arg873_ptr,
        scale_ptr,
        out_base_ptr,
        partial_sum_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        tile = tl.program_id(0)
        rows = tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, BLOCK_CHANNELS_)
        row_mask = rows < ROWS_
        col_mask = cols < CHANNELS_
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * CHANNELS_ + cols[None, :]

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        arg873 = tl.load(arg873_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        # The captured graph stores bf16 intermediates before reusing them in
        # f32 reductions, so preserve those roundings explicitly.
        weighted = (mm * weight[None, :]).to(tl.bfloat16).to(tl.float32)
        scaled_arg = (arg873 * scale[:, None]).to(tl.bfloat16).to(tl.float32)
        prod_for_col_sum = (mm * scaled_arg).to(tl.bfloat16).to(tl.float32)

        row_sum = tl.sum(tl.where(mask, weighted * arg873, 0.0), axis=1)
        scale2 = scale * scale
        correction = ((-0.5 * row_sum * scale2 * scale) / CHANNELS_)[:, None]
        out = weighted * scale[:, None] + correction * (2.0 * arg873)

        tl.store(out_base_ptr + offsets, out, mask=mask)

        partial_offsets = tile * CHANNELS_ + cols
        tl.store(
            partial_sum_ptr + partial_offsets,
            tl.sum(tl.where(mask, prod_for_col_sum, 0.0), axis=0),
            mask=col_mask,
        )

    @triton.jit
    def _finalize_partials_kernel(
        partial_sum_ptr,
        out_sum_ptr,
        NUM_ROW_TILES: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = (tiles[:, None] < NUM_ROW_TILES) & (cols[None, :] < CHANNELS_)
        offsets = tiles[:, None] * CHANNELS_ + cols[None, :]
        values = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_sum_ptr + cols, tl.sum(values, axis=0), mask=cols < CHANNELS_)


def oracle_triton(
    mm_1: torch.Tensor,
    arg311_1: torch.Tensor,
    arg873_1: torch.Tensor,
    arg874_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full graph in a row producer plus column-partial finalizer."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if mm_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    assert mm_1.shape == (ROWS, CHANNELS)
    assert arg311_1.shape == (CHANNELS,)
    assert arg873_1.shape == (4, 512, CHANNELS)
    assert arg874_1.shape == (4, 512, 1)
    assert mm_1.is_contiguous()
    assert arg311_1.is_contiguous()
    assert arg873_1.is_contiguous()

    num_row_tiles = triton.cdiv(ROWS, ROW_TILE)
    out_sum = torch.empty((CHANNELS,), device=mm_1.device, dtype=torch.bfloat16)
    out_base = torch.empty((ROWS, CHANNELS), device=mm_1.device, dtype=torch.bfloat16)
    partial_sum = torch.empty(
        (num_row_tiles, CHANNELS),
        device=mm_1.device,
        dtype=torch.float32,
    )

    _row_tile_kernel[(num_row_tiles,)](
        mm_1,
        arg311_1,
        arg873_1,
        arg874_1,
        out_base,
        partial_sum,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        BLOCK_ROWS=ROW_TILE,
        BLOCK_CHANNELS_=BLOCK_CHANNELS,
        num_warps=8,
    )

    _finalize_partials_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_sum,
        out_sum,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS_=FINAL_BLOCK_CHANNELS,
        num_warps=4,
    )

    return out_sum, out_base.permute(1, 0)


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    (
        mm_1,
        arg311_1,
        arg873_1,
        arg874_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    return oracle_triton(mm_1, arg311_1, arg873_1, arg874_1)


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
