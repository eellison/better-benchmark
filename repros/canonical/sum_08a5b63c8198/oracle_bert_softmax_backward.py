"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT attention softmax-backward row update returned by Repro.forward, including the dropout-mask scale on the BMM gradient, broadcast attention mask substitution, libdevice f32 exp normalization, row sum, tl.fma softmax-gradient epilogue, second masked fill, final divide-by-8 ordering, and contiguous f32[1536,128,128] output; Inductor already lowers this full scope as one fused persistent row-reduction kernel, so this fixed-shape row oracle is a bandwidth/math floor check rather than evidence for a missing scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, or new-pattern transformation; the fix is BANDWIDTH_BOUND: record as at_floor unless this full-scope oracle materially beats torch.compile under the shared CUDAGraph harness."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


ROWS = 128 * 12 * 128
COLS = 128
HEADS = 12
ROWS_PER_BLOCK = 4
OUT_SHAPE = (1536, 128, 128)
OUT_STRIDE = (16384, 128, 1)
VIEW_SHAPE = (128, 12, 128, 128)
VIEW_STRIDE = (196608, 16384, 128, 1)
ROW_SCALAR_STRIDE = (1536, 128, 1, 1)
EQ_STRIDE = (16384, 16384, 128, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def oracle_kernel(
        bmm_ptr,
        dropout_mask_ptr,
        score_ptr,
        eq_ptr,
        row_shift_ptr,
        row_denom_ptr,
        full_ptr,
        out_ptr,
        ROWS_PER_BLOCK_: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HEADS_: tl.constexpr,
        EQ_BATCH_STRIDE_: tl.constexpr,
        EQ_QUERY_STRIDE_: tl.constexpr,
        ROW_SCALAR_BATCH_STRIDE_: tl.constexpr,
        ROW_SCALAR_HEAD_STRIDE_: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS_PER_BLOCK_ + tl.arange(0, ROWS_PER_BLOCK_)[:, None]
        cols = tl.arange(0, BLOCK_N)[None, :]

        batch = rows // (HEADS_ * BLOCK_N)
        head_q = rows - batch * (HEADS_ * BLOCK_N)
        head = head_q // BLOCK_N
        query = head_q - head * BLOCK_N

        row_offsets = rows * BLOCK_N + cols
        eq_offsets = batch * EQ_BATCH_STRIDE_ + query * EQ_QUERY_STRIDE_ + cols
        row_scalar_offsets = batch * ROW_SCALAR_BATCH_STRIDE_ + head * ROW_SCALAR_HEAD_STRIDE_ + query

        bmm = tl.load(bmm_ptr + row_offsets, eviction_policy="evict_first").to(tl.float32)
        keep = tl.load(dropout_mask_ptr + row_offsets, eviction_policy="evict_first").to(tl.float32)
        eq = tl.load(eq_ptr + eq_offsets, eviction_policy="evict_first")
        scores = tl.load(score_ptr + row_offsets, eviction_policy="evict_first").to(tl.float32)
        row_shift = tl.load(row_shift_ptr + row_scalar_offsets).to(tl.float32)
        row_denom = tl.load(row_denom_ptr + row_scalar_offsets).to(tl.float32)
        full_value = tl.load(full_ptr + 0).to(tl.float32)

        dropout_scaled_bmm = bmm * (keep * 1.1111111111111112)
        divided_scores = scores * 0.125
        masked_scores = tl.where(eq, -1000000000.0, divided_scores)
        probs = libdevice.exp(masked_scores - row_shift) / row_denom
        product = dropout_scaled_bmm * probs
        row_sum = tl.sum(product, 1)[:, None].to(tl.float32)
        grad = tl.fma(-probs, row_sum, product)
        selected = tl.where(eq, full_value, grad)
        out = selected * 0.125

        tl.store(out_ptr + row_offsets, out)


def _validate_inputs(
    bmm_45: torch.Tensor,
    arg108_1: torch.Tensor,
    arg105_1: torch.Tensor,
    eq_1: torch.Tensor,
    arg106_1: torch.Tensor,
    arg107_1: torch.Tensor,
    full_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> None:
    assert bmm_45.is_cuda
    assert arg108_1.is_cuda
    assert arg105_1.is_cuda
    assert eq_1.is_cuda
    assert arg106_1.is_cuda
    assert arg107_1.is_cuda
    assert full_1.is_cuda
    assert bmm_45.dtype is torch.float32 and tuple(bmm_45.shape) == OUT_SHAPE
    assert arg108_1.dtype is torch.bool and tuple(arg108_1.shape) == VIEW_SHAPE
    assert arg105_1.dtype is torch.float32 and tuple(arg105_1.shape) == OUT_SHAPE
    assert eq_1.dtype is torch.bool and tuple(eq_1.shape) == (128, 1, 128, 128)
    assert arg106_1.dtype is torch.float32 and tuple(arg106_1.shape) == (128, 12, 128, 1)
    assert arg107_1.dtype is torch.float32 and tuple(arg107_1.shape) == (128, 12, 128, 1)
    assert full_1.dtype is torch.float32 and tuple(full_1.shape) == ()
    assert bmm_45.stride() == OUT_STRIDE
    assert arg108_1.stride() == VIEW_STRIDE
    assert arg105_1.stride() == OUT_STRIDE
    assert eq_1.stride() == EQ_STRIDE
    assert arg106_1.stride() == ROW_SCALAR_STRIDE
    assert arg107_1.stride() == ROW_SCALAR_STRIDE
    assert tuple(_shape_param_0) == VIEW_SHAPE
    assert tuple(_shape_param_1) == VIEW_SHAPE
    assert tuple(_shape_param_2) == OUT_SHAPE


@oracle_impl(hardware="H100", shapes="(T([1536, 128, 128], f32), T([128, 12, 128, 128], b8), T([1536, 128, 128], f32), T([128, 1, 128, 128], b8), T([128, 12, 128, 1], f32), T([128, 12, 128, 1], f32), T([], f32), S([128, 12, 128, 128]), S([128, 12, 128, 128]), S([1536, 128, 128]))")
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
        raise RuntimeError("triton is required for this oracle")

    (
        bmm_45,
        arg108_1,
        arg105_1,
        eq_1,
        arg106_1,
        arg107_1,
        full_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    _validate_inputs(
        bmm_45,
        arg108_1,
        arg105_1,
        eq_1,
        arg106_1,
        arg107_1,
        full_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )

    out = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=bmm_45.device, dtype=torch.float32)
    grid = (ROWS // ROWS_PER_BLOCK,)
    oracle_kernel[grid](
        bmm_45,
        arg108_1,
        arg105_1,
        eq_1,
        arg106_1,
        arg107_1,
        full_1,
        out,
        ROWS_PER_BLOCK_=ROWS_PER_BLOCK,
        BLOCK_N=COLS,
        HEADS_=HEADS,
        EQ_BATCH_STRIDE_=EQ_STRIDE[0],
        EQ_QUERY_STRIDE_=EQ_STRIDE[2],
        ROW_SCALAR_BATCH_STRIDE_=ROW_SCALAR_STRIDE[0],
        ROW_SCALAR_HEAD_STRIDE_=ROW_SCALAR_STRIDE[1],
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
