"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Longformer layout assembly by treating the iota/as_strided index_put, crop, pad, view, permute, clone, final transpose, and sibling hidden-dimension sum as one structured overlapping layout scatter-reduce from the original f32[96,768,64] input, whereas Inductor currently materializes the zero-filled f32[2359296] index_put result and schedules the downstream layout chain and reduction as generic separate work; Inductor cannot do this today because scheduler/codegen does not recognize this fixed Longformer overlapping iota index pattern as a direct cropped output-layout producer with a reduction epilogue; the fix is SCATTER_REDUCE: add a structured iota/as_strided index_put lowering that writes the final transposed storage layout directly and accumulates compatible hidden reductions without materializing the dense intermediate."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

SOURCE_BLOCKS = 96
LAYOUT_BLOCKS = 24
OVERLAP_BLOCKS = 4
TOKENS = 1024
BATCH = 2
HEADS = 12
HIDDEN = 768
K = 64
ROWS = TOKENS * BATCH
CROP_START = 256
SOURCE_BLOCK_STRIDE = HIDDEN * K
TILE_ROWS = 16
TILE_HIDDEN = 32
NUM_ROW_TILES = (ROWS + TILE_ROWS - 1) // TILE_ROWS


if triton is not None and tl is not None:

    @triton.jit
    def _layout_and_partial_reduce_kernel(
        bmm_ptr,
        transposed_storage_ptr,
        partial_ptr,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        K_: tl.constexpr,
        HEADS_: tl.constexpr,
        SOURCE_BLOCK_STRIDE_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        hidden_tile = tl.program_id(1)

        rows = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        hidden = hidden_tile * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)[None, :]
        active = (rows < ROWS_) & (hidden < HIDDEN_)

        token = rows // 2
        batch = rows - token * 2
        head = hidden // K_
        k = hidden - head * K_
        layout_block = batch * HEADS_ + head

        c0 = token + 256
        c1 = token
        c2 = token - 256
        c3 = token - 512

        offset0 = ((layout_block * 4 + 0) * HIDDEN_ + c0) * K_ + k
        offset1 = ((layout_block * 4 + 1) * HIDDEN_ + c1) * K_ + k
        offset2 = ((layout_block * 4 + 2) * HIDDEN_ + c2) * K_ + k
        offset3 = ((layout_block * 4 + 3) * HIDDEN_ + c3) * K_ + k

        values = tl.load(
            bmm_ptr + offset0,
            mask=active & (token < 512),
            other=0.0,
        ).to(tl.float32)
        values += tl.load(
            bmm_ptr + offset1,
            mask=active & (token < 768),
            other=0.0,
        ).to(tl.float32)
        values += tl.load(
            bmm_ptr + offset2,
            mask=active & (token >= 256),
            other=0.0,
        ).to(tl.float32)
        values += tl.load(
            bmm_ptr + offset3,
            mask=active & (token >= 512),
            other=0.0,
        ).to(tl.float32)

        tl.store(transposed_storage_ptr + rows * HIDDEN_ + hidden, values, mask=active)

        partial = tl.sum(tl.where(rows < ROWS_, values, 0.0), axis=0)
        hidden_cols = hidden_tile * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
        tl.store(
            partial_ptr + row_tile * HIDDEN_ + hidden_cols,
            partial,
            mask=hidden_cols < HIDDEN_,
        )

    @triton.jit
    def _finalize_sum_kernel(
        partial_ptr,
        sum_ptr,
        NUM_ROW_TILES_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        hidden = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        active = tiles < NUM_ROW_TILES_
        values = tl.load(
            partial_ptr + tiles * HIDDEN_ + hidden,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        tl.store(sum_ptr + hidden, tl.sum(values, axis=0))


def _check_shape_params(
    shape_param_0,
    shape_param_1,
    shape_param_2,
    shape_param_3,
    shape_param_4,
    shape_param_5,
) -> None:
    assert list(shape_param_0) == [LAYOUT_BLOCKS, OVERLAP_BLOCKS, HIDDEN, K, 1]
    assert list(shape_param_1) == [LAYOUT_BLOCKS * OVERLAP_BLOCKS * HIDDEN * K]
    assert list(shape_param_2) == [BATCH, HEADS, TOKENS, K]
    assert list(shape_param_3) == [TOKENS, BATCH, HIDDEN]
    assert list(shape_param_4) == [HIDDEN]
    assert list(shape_param_5) == [ROWS, HIDDEN]


def _check_bmm(bmm: torch.Tensor) -> None:
    assert bmm.is_cuda
    assert bmm.shape == (SOURCE_BLOCKS, HIDDEN, K)
    assert bmm.dtype == torch.float32
    assert bmm.is_contiguous()


@oracle_impl(hardware="H100", shapes="(T([96, 768, 64], f32), S([24, 4, 768, 64, 1]), S([4718592]), S([2, 12, 1024, 64]), S([1024, 2, 768]), S([768]), S([2048, 768]))")
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
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        bmm,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
        shape_param_4,
        shape_param_5,
    ) = inputs
    _check_shape_params(
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
        shape_param_4,
        shape_param_5,
    )
    _check_bmm(bmm)

    transposed = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=bmm.device,
        dtype=torch.float32,
    )
    partial = torch.empty(
        (NUM_ROW_TILES, HIDDEN),
        device=bmm.device,
        dtype=torch.float32,
    )
    summed = torch.empty((HIDDEN,), device=bmm.device, dtype=torch.float32)

    _layout_and_partial_reduce_kernel[
        (NUM_ROW_TILES, triton.cdiv(HIDDEN, TILE_HIDDEN))
    ](
        bmm,
        transposed,
        partial,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        K_=K,
        HEADS_=HEADS,
        SOURCE_BLOCK_STRIDE_=SOURCE_BLOCK_STRIDE,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_HIDDEN=TILE_HIDDEN,
        num_warps=8,
    )
    _finalize_sum_kernel[(HIDDEN,)](
        partial,
        summed,
        NUM_ROW_TILES_=NUM_ROW_TILES,
        HIDDEN_=HIDDEN,
        BLOCK_TILES=triton.next_power_of_2(NUM_ROW_TILES),
        num_warps=4,
    )
    return summed, transposed


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
