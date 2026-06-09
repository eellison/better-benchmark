"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Longformer sliding-window backward reduction, dropout/padding masks, triangular edge masks, structured chunk scatter, and final skewed output directly from the original row sources with one row-reduction kernel and one gather-mask-scatter-elimination kernel, whereas Inductor materializes the zero-padded views, clone/copy slice updates, slice_scatter/select_scatter tree, and final layout conversion as generic tensor work around the reduction; Inductor cannot do this today because scheduler/codegen does not recognize this structured sliding-window scatter/reduce and skew pattern as a source-space gather with disjoint scatter domains; the fix is SCATTER_REDUCE: add a structured Longformer sliding-window backward lowering that computes row sums once and maps source chunks directly into the final skewed output without materialized scatter intermediates."""
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


N_BATCH = 2
N_HEADS = 12
N_CHUNKS = 4
LOCAL_TOKENS = 256
WINDOW = 513
BMM_COLS = 768
OUT_GROUPS = 72
OUT_ROWS = 512
OUT_COLS = 512
OUT_ROW_NUMEL = OUT_ROWS * OUT_COLS
TOTAL_ROWS = N_BATCH * N_HEADS * N_CHUNKS * LOCAL_TOKENS
TOTAL_OUT = OUT_GROUPS * OUT_ROW_NUMEL
DROPOUT_SCALE = 1.1111111111111112

BLOCK_WINDOW = 1024
BLOCK_OUT_COLS = 256


if triton is not None:

    @triton.jit
    def _row_sums_kernel(
        bmm_ptr,
        dropout_mask_ptr,
        padding_mask_ptr,
        softmax_ptr,
        sums_ptr,
        BLOCK_WINDOW_: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets_j = tl.arange(0, BLOCK_WINDOW_)
        j_mask = offsets_j < 513

        local_i = row % 256
        chunk_tmp = row // 256
        chunk = chunk_tmp % 4
        head_batch = chunk_tmp // 4
        batch = head_batch // 12
        head = head_batch - batch * 12
        token = chunk * 256 + local_i

        bmm_base = ((head_batch * 4 + chunk) * 256 + local_i) * 768
        attn_base = ((batch * 1024 + token) * 12 + head) * 513
        active_token = tl.load(padding_mask_ptr + batch * 1024 + token) == 0

        bmm = tl.load(
            bmm_ptr + bmm_base + local_i + offsets_j,
            mask=j_mask,
            other=0.0,
        ).to(tl.float32)
        keep = tl.load(
            dropout_mask_ptr + attn_base + offsets_j,
            mask=j_mask,
            other=0,
        ).to(tl.float32)
        softmax = tl.load(
            softmax_ptr + attn_base + offsets_j,
            mask=j_mask,
            other=0.0,
        ).to(tl.float32)

        grad = tl.where(j_mask & active_token, bmm * keep * 1.1111111111111112, 0.0)
        row_sum = tl.sum(grad * softmax, axis=0)
        tl.store(sums_ptr + row, row_sum)

    @triton.jit
    def _final_output_kernel(
        bmm_ptr,
        dropout_mask_ptr,
        padding_mask_ptr,
        softmax_ptr,
        upper_mask_ptr,
        lower_mask_ptr,
        sums_ptr,
        out_ptr,
        BLOCK_OUT_COLS_: tl.constexpr,
    ):
        out_group = tl.program_id(0)
        out_row = tl.program_id(1)
        col_block = tl.program_id(2)
        offsets_col = col_block * BLOCK_OUT_COLS_ + tl.arange(0, BLOCK_OUT_COLS_)
        col_mask = offsets_col < 512

        head_batch = out_group // 3
        out_channel = out_group - head_batch * 3
        batch = head_batch // 12
        head = head_batch - batch * 12

        # Equivalent to viewing a [512, 513] scatter buffer as [513, 512]
        # and cropping the last skew row.
        scatter_row = tl.where(offsets_col >= out_row, out_row, out_row - 1)
        scatter_col = tl.where(
            offsets_col >= out_row,
            offsets_col - out_row,
            513 + offsets_col - out_row,
        )

        valid_right = (scatter_row < 256) & (scatter_col < 257)
        valid_left_edge = (
            (out_channel == 0)
            & (scatter_row < 255)
            & (scatter_col >= 258)
        )
        valid_lower = (
            (scatter_row >= 255)
            & (scatter_row < 511)
            & (scatter_col >= 257)
        )
        valid_upper_edge = (
            (out_channel == 2)
            & (scatter_row >= 256)
            & (scatter_col < 257)
        )
        valid = col_mask & (valid_right | valid_left_edge | valid_lower | valid_upper_edge)

        src_chunk = tl.zeros((BLOCK_OUT_COLS_,), dtype=tl.int64)
        src_i = tl.zeros((BLOCK_OUT_COLS_,), dtype=tl.int64)
        src_j = tl.zeros((BLOCK_OUT_COLS_,), dtype=tl.int64)

        src_chunk = tl.where(valid_right, out_channel, src_chunk)
        src_i = tl.where(valid_right, scatter_row, src_i)
        src_j = tl.where(valid_right, scatter_col + 256, src_j)

        src_chunk = tl.where(valid_left_edge, 0, src_chunk)
        src_i = tl.where(valid_left_edge, scatter_row + 1, src_i)
        src_j = tl.where(valid_left_edge, scatter_col - 257, src_j)

        src_chunk = tl.where(valid_lower, out_channel + 1, src_chunk)
        src_i = tl.where(valid_lower, scatter_row - 255, src_i)
        src_j = tl.where(valid_lower, scatter_col - 257, src_j)

        src_chunk = tl.where(valid_upper_edge, 3, src_chunk)
        src_i = tl.where(valid_upper_edge, scatter_row - 256, src_i)
        src_j = tl.where(valid_upper_edge, scatter_col + 256, src_j)

        src_token = src_chunk * 256 + src_i
        sum_row = (head_batch * 4 + src_chunk) * 256 + src_i
        bmm_base = ((head_batch * 4 + src_chunk) * 256 + src_i) * 768
        attn_base = ((batch * 1024 + src_token) * 12 + head) * 513

        active_token = tl.load(
            padding_mask_ptr + batch * 1024 + src_token,
            mask=valid,
            other=1,
        ) == 0
        bmm = tl.load(
            bmm_ptr + bmm_base + src_i + src_j,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        keep = tl.load(
            dropout_mask_ptr + attn_base + src_j,
            mask=valid,
            other=0,
        ).to(tl.float32)
        softmax = tl.load(
            softmax_ptr + attn_base + src_j,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        row_sum = tl.load(sums_ptr + sum_row, mask=valid, other=0.0).to(tl.float32)

        grad = tl.where(valid & active_token, bmm * keep * 1.1111111111111112, 0.0)
        value = softmax * (grad - row_sum)

        lower_region = (src_chunk == 0) & (src_j < 257)
        upper_region = (src_chunk == 3) & (src_j >= 256)
        lower_j = tl.where(lower_region, src_j, 0)
        upper_j = tl.where(upper_region, src_j - 256, 0)
        lower_mask = tl.load(
            lower_mask_ptr + src_i * 257 + lower_j,
            mask=valid & lower_region,
            other=0.0,
        ) != 0.0
        upper_mask = tl.load(
            upper_mask_ptr + src_i * 257 + upper_j,
            mask=valid & upper_region,
            other=0.0,
        ) != 0.0

        value = tl.where(valid & ~(lower_mask | upper_mask), value, 0.0)
        out_offsets = out_group * 262144 + out_row * 512 + offsets_col
        tl.store(out_ptr + out_offsets, value, mask=col_mask)


@oracle_impl(hardware="H100", shapes="(T([96, 256, 768], f32), T([2, 1024, 12, 513], b8), T([2, 1024], b8), T([2, 1024, 12, 513], f32), T([1, 256, 1, 257], f32), T([1, 256, 1, 257], f32), S([24, 4, 256, 768, 1]), S([24, 4, 196864]), S([24, 4, 256, 770]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 256, 12, 257]), S([24, 4, 256, 513]), S([2, 12, 1024, 513]), S([24, 4, 256, 513]), S([2, 256, 12, 257]), S([24, 4, 256, 513]), S([24, 3, 513, 512]), S([24, 3, 512, 512, 1]), S([72, 512, 512]))")
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
        raise RuntimeError("Triton is required for this oracle")

    bmm_1, arg222_1, arg3_1, arg221_1, arg99_1, arg98_1, *_ = inputs
    if bmm_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if tuple(bmm_1.shape) != (96, 256, 768):
        raise ValueError(f"unexpected bmm_1 shape: {tuple(bmm_1.shape)}")
    if tuple(arg222_1.shape) != (2, 1024, 12, 513):
        raise ValueError(f"unexpected arg222_1 shape: {tuple(arg222_1.shape)}")
    if tuple(arg3_1.shape) != (2, 1024):
        raise ValueError(f"unexpected arg3_1 shape: {tuple(arg3_1.shape)}")
    if tuple(arg221_1.shape) != (2, 1024, 12, 513):
        raise ValueError(f"unexpected arg221_1 shape: {tuple(arg221_1.shape)}")
    if tuple(arg99_1.shape) != (1, 256, 1, 257):
        raise ValueError(f"unexpected arg99_1 shape: {tuple(arg99_1.shape)}")
    if tuple(arg98_1.shape) != (1, 256, 1, 257):
        raise ValueError(f"unexpected arg98_1 shape: {tuple(arg98_1.shape)}")

    row_sums = torch.empty((TOTAL_ROWS,), device=bmm_1.device, dtype=torch.float32)
    out = torch.empty((OUT_GROUPS, OUT_ROWS, OUT_COLS), device=bmm_1.device, dtype=torch.float32)

    _row_sums_kernel[(TOTAL_ROWS,)](
        bmm_1,
        arg222_1,
        arg3_1,
        arg221_1,
        row_sums,
        BLOCK_WINDOW_=BLOCK_WINDOW,
        num_warps=8,
    )
    _final_output_kernel[(OUT_GROUPS, OUT_ROWS, triton.cdiv(OUT_COLS, BLOCK_OUT_COLS))](
        bmm_1,
        arg222_1,
        arg3_1,
        arg221_1,
        arg99_1,
        arg98_1,
        row_sums,
        out,
        BLOCK_OUT_COLS_=BLOCK_OUT_COLS,
        num_warps=4,
    )
    return out


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
