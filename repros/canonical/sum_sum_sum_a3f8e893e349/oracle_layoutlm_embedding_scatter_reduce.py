"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete LayoutLM layernorm/dropout backward return tuple, including both hidden-column reductions, all indexed embedding-gradient accumulators, and the vocabulary-gradient add into `mm_1[:-2]`, by recomputing the shared row producer once and scattering it directly, whereas Inductor currently materializes the rowwise producer, masks, generic `index_put(accumulate=True)` scatter intermediates, pairwise scatter adds, and final slice add as separate scheduled work; Inductor cannot do this today because scheduler/codegen has no multi-destination embedding scatter-reduce template that keeps the row-local reductions live while feeding several duplicate-index accumulators and a dense base-add output; the fix is SCATTER_REDUCE: add a gather-mask-reduce lowering for embedding backward patterns that fuses the row algebra, hidden reductions, masked scatter-adds, and base vocabulary add into direct output-space kernels."""
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


BATCH = 32
SEQ = 512
ROWS = BATCH * SEQ
HIDDEN = 768
SEGMENT_ROWS = 2
POSITION_ROWS = 512
TABLE_ROWS = 1024
VOCAB_ROWS = 30522
DROPOUT_SCALE = 1.1111111111111112
BLOCK_H = 1024
INIT_BLOCK = 1024


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        mm1_ptr,
        sum_saved_ptr,
        sum_plain_ptr,
        out_segment_ptr,
        out_table_a_ptr,
        out_table_b_ptr,
        out_table_c_ptr,
        out_table_d_ptr,
        out_position_ptr,
        out_vocab_ptr,
        HIDDEN_: tl.constexpr,
        SEGMENT_ROWS_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        TABLE_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        zeros = tl.zeros([BLOCK_], dtype=tl.float32)

        sum_saved_size = HIDDEN_
        sum_plain_base = sum_saved_size
        out_segment_base = sum_plain_base + HIDDEN_
        out_segment_size = SEGMENT_ROWS_ * HIDDEN_
        out_table_a_base = out_segment_base + out_segment_size
        out_table_size = TABLE_ROWS_ * HIDDEN_
        out_table_b_base = out_table_a_base + out_table_size
        out_table_c_base = out_table_b_base + out_table_size
        out_table_d_base = out_table_c_base + out_table_size
        out_position_base = out_table_d_base + out_table_size
        out_position_size = POSITION_ROWS_ * HIDDEN_
        out_vocab_base = out_position_base + out_position_size
        out_vocab_size = VOCAB_ROWS_ * HIDDEN_

        mask0 = offsets < sum_saved_size
        tl.store(sum_saved_ptr + offsets, zeros, mask=mask0)

        off1 = offsets - sum_plain_base
        mask1 = (off1 >= 0) & (off1 < HIDDEN_)
        tl.store(sum_plain_ptr + off1, zeros, mask=mask1)

        off2 = offsets - out_segment_base
        mask2 = (off2 >= 0) & (off2 < out_segment_size)
        tl.store(out_segment_ptr + off2, zeros, mask=mask2)

        off3 = offsets - out_table_a_base
        mask3 = (off3 >= 0) & (off3 < out_table_size)
        tl.store(out_table_a_ptr + off3, zeros, mask=mask3)

        off4 = offsets - out_table_b_base
        mask4 = (off4 >= 0) & (off4 < out_table_size)
        tl.store(out_table_b_ptr + off4, zeros, mask=mask4)

        off5 = offsets - out_table_c_base
        mask5 = (off5 >= 0) & (off5 < out_table_size)
        tl.store(out_table_c_ptr + off5, zeros, mask=mask5)

        off6 = offsets - out_table_d_base
        mask6 = (off6 >= 0) & (off6 < out_table_size)
        tl.store(out_table_d_ptr + off6, zeros, mask=mask6)

        off7 = offsets - out_position_base
        mask7 = (off7 >= 0) & (off7 < out_position_size)
        tl.store(out_position_ptr + off7, zeros, mask=mask7)

        off8 = offsets - out_vocab_base
        mask8 = (off8 >= 0) & (off8 < out_vocab_size)
        vocab_base = tl.load(mm1_ptr + off8, mask=mask8, other=0.0).to(tl.float32)
        tl.store(out_vocab_ptr + off8, vocab_base, mask=mask8)

    @triton.jit
    def _layoutlm_scatter_reduce_kernel(
        mm142_ptr,
        mul290_ptr,
        mm144_ptr,
        mm146_ptr,
        dropout_mask_ptr,
        gamma_ptr,
        saved_ptr,
        row_scale_ptr,
        full_ptr,
        segment_idx_ptr,
        table_a_idx_ptr,
        table_b_idx_ptr,
        table_c0_idx_ptr,
        table_d0_idx_ptr,
        table_c1_idx_ptr,
        table_d1_idx_ptr,
        position_idx_ptr,
        vocab_idx_ptr,
        sum_saved_ptr,
        sum_plain_ptr,
        out_segment_ptr,
        out_table_a_ptr,
        out_table_b_ptr,
        out_table_c_ptr,
        out_table_d_ptr,
        out_position_ptr,
        out_vocab_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        SEGMENT_ROWS_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        TABLE_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        hidden_mask = h < HIDDEN_
        offsets = row * HIDDEN_ + h

        base = tl.load(mul290_ptr + offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        base += tl.load(mm142_ptr + offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        base += tl.load(mm144_ptr + offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        base += tl.load(mm146_ptr + offsets, mask=hidden_mask, other=0.0).to(tl.float32)

        keep = tl.load(dropout_mask_ptr + offsets, mask=hidden_mask, other=0).to(tl.float32)
        produced = base * keep * DROPOUT_SCALE_
        saved = tl.load(saved_ptr + offsets, mask=hidden_mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + h, mask=hidden_mask, other=0.0).to(tl.float32)
        scaled = produced * gamma

        row_sum = tl.sum(tl.where(hidden_mask, scaled, 0.0), axis=0)
        row_dot = tl.sum(tl.where(hidden_mask, scaled * saved, 0.0), axis=0)
        row_scale = tl.load(row_scale_ptr + row).to(tl.float32)
        dx = row_scale * (scaled * HIDDEN_ - row_sum - saved * row_dot)

        tl.atomic_add(sum_saved_ptr + h, produced * saved, sem="relaxed", mask=hidden_mask)
        tl.atomic_add(sum_plain_ptr + h, produced, sem="relaxed", mask=hidden_mask)

        full_value = tl.load(full_ptr).to(tl.float32)

        segment_raw = tl.load(segment_idx_ptr + row).to(tl.int64)
        segment_idx = tl.where(segment_raw < 0, segment_raw + SEGMENT_ROWS_, segment_raw)
        segment_valid = (segment_idx >= 0) & (segment_idx < SEGMENT_ROWS_)
        tl.atomic_add(
            out_segment_ptr + segment_idx * HIDDEN_ + h,
            dx,
            sem="relaxed",
            mask=hidden_mask & segment_valid,
        )

        table_a_raw = tl.load(table_a_idx_ptr + row).to(tl.int64)
        table_a_idx = tl.where(table_a_raw < 0, table_a_raw + TABLE_ROWS_, table_a_raw)
        table_a_val = tl.where(table_a_raw == -1, full_value, dx)
        table_a_valid = (table_a_idx >= 0) & (table_a_idx < TABLE_ROWS_)
        tl.atomic_add(
            out_table_a_ptr + table_a_idx * HIDDEN_ + h,
            table_a_val,
            sem="relaxed",
            mask=hidden_mask & table_a_valid,
        )

        table_b_raw = tl.load(table_b_idx_ptr + row).to(tl.int64)
        table_b_idx = tl.where(table_b_raw < 0, table_b_raw + TABLE_ROWS_, table_b_raw)
        table_b_val = tl.where(table_b_raw == -1, full_value, dx)
        table_b_valid = (table_b_idx >= 0) & (table_b_idx < TABLE_ROWS_)
        tl.atomic_add(
            out_table_b_ptr + table_b_idx * HIDDEN_ + h,
            table_b_val,
            sem="relaxed",
            mask=hidden_mask & table_b_valid,
        )

        table_c0_raw = tl.load(table_c0_idx_ptr + row).to(tl.int64)
        table_c0_idx = tl.where(table_c0_raw < 0, table_c0_raw + TABLE_ROWS_, table_c0_raw)
        table_c0_val = tl.where(table_c0_raw == -1, full_value, dx)
        table_c0_valid = (table_c0_idx >= 0) & (table_c0_idx < TABLE_ROWS_)
        tl.atomic_add(
            out_table_c_ptr + table_c0_idx * HIDDEN_ + h,
            table_c0_val,
            sem="relaxed",
            mask=hidden_mask & table_c0_valid,
        )

        table_d0_raw = tl.load(table_d0_idx_ptr + row).to(tl.int64)
        table_d0_idx = tl.where(table_d0_raw < 0, table_d0_raw + TABLE_ROWS_, table_d0_raw)
        table_d0_val = tl.where(table_d0_raw == -1, full_value, dx)
        table_d0_valid = (table_d0_idx >= 0) & (table_d0_idx < TABLE_ROWS_)
        tl.atomic_add(
            out_table_d_ptr + table_d0_idx * HIDDEN_ + h,
            table_d0_val,
            sem="relaxed",
            mask=hidden_mask & table_d0_valid,
        )

        table_c1_raw = tl.load(table_c1_idx_ptr + row).to(tl.int64)
        table_c1_idx = tl.where(table_c1_raw < 0, table_c1_raw + TABLE_ROWS_, table_c1_raw)
        table_c1_val = tl.where(table_c1_raw == -1, full_value, dx)
        table_c1_valid = (table_c1_idx >= 0) & (table_c1_idx < TABLE_ROWS_)
        tl.atomic_add(
            out_table_c_ptr + table_c1_idx * HIDDEN_ + h,
            table_c1_val,
            sem="relaxed",
            mask=hidden_mask & table_c1_valid,
        )

        table_d1_raw = tl.load(table_d1_idx_ptr + row).to(tl.int64)
        table_d1_idx = tl.where(table_d1_raw < 0, table_d1_raw + TABLE_ROWS_, table_d1_raw)
        table_d1_val = tl.where(table_d1_raw == -1, full_value, dx)
        table_d1_valid = (table_d1_idx >= 0) & (table_d1_idx < TABLE_ROWS_)
        tl.atomic_add(
            out_table_d_ptr + table_d1_idx * HIDDEN_ + h,
            table_d1_val,
            sem="relaxed",
            mask=hidden_mask & table_d1_valid,
        )

        seq = row % SEQ_
        batch = row // SEQ_
        position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
        position_idx = tl.where(
            position_raw < 0,
            position_raw + POSITION_ROWS_,
            position_raw,
        )
        position_val = tl.where(
            position_raw == -1,
            tl.where(batch == 0, full_value, 0.0),
            dx,
        )
        position_valid = (position_idx >= 0) & (position_idx < POSITION_ROWS_)
        tl.atomic_add(
            out_position_ptr + position_idx * HIDDEN_ + h,
            position_val,
            sem="relaxed",
            mask=hidden_mask & position_valid,
        )

        vocab_raw = tl.load(vocab_idx_ptr + row).to(tl.int64)
        vocab_idx = tl.where(vocab_raw < 0, vocab_raw + VOCAB_ROWS_, vocab_raw)
        vocab_val = tl.where(vocab_raw == 0, full_value, dx)
        vocab_valid = (vocab_idx >= 0) & (vocab_idx < VOCAB_ROWS_)
        tl.atomic_add(
            out_vocab_ptr + vocab_idx * HIDDEN_ + h,
            vocab_val,
            sem="relaxed",
            mask=hidden_mask & vocab_valid,
        )


@oracle_impl(hardware="H100", shapes="(T([30524, 768], f32), T([16384, 768], f32), T([32, 512, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([32, 512, 768], b8), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([], f32), T([32, 512], i64, gen=Index(2)), T([32, 512], i64, gen=Index(1024)), T([32, 512], i64, gen=Index(1024)), T([32, 512], i64, gen=Index(1024)), T([32, 512], i64, gen=Index(1024)), T([32, 512], i64, gen=Index(1024)), T([32, 512], i64, gen=Index(1024)), T([1, 512], i64, gen=Index(512)), T([32, 512], i64, gen=Index(30522)), S([32, 512, 768]), S([32, 512, 768]), S([32, 512, 768]))")
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
        mm_1,
        mm_142,
        mul_290,
        mm_144,
        mm_146,
        arg111_1,
        arg3_1,
        arg110_1,
        arg337_1,
        full_1,
        arg103_1,
        arg109_1,
        arg108_1,
        arg107_1,
        arg106_1,
        arg105_1,
        arg104_1,
        arg2_1,
        arg0_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    if mm_1.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 0: {_shape_param_0}")
    if list(_shape_param_1) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 1: {_shape_param_1}")
    if list(_shape_param_2) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 2: {_shape_param_2}")

    assert mm_1.shape == (VOCAB_ROWS + 2, HIDDEN) and mm_1.is_contiguous()
    assert mm_142.shape == (ROWS, HIDDEN) and mm_142.is_contiguous()
    assert mul_290.shape == (BATCH, SEQ, HIDDEN) and mul_290.is_contiguous()
    assert mm_144.shape == (ROWS, HIDDEN) and mm_144.is_contiguous()
    assert mm_146.shape == (ROWS, HIDDEN) and mm_146.is_contiguous()
    assert arg111_1.shape == (BATCH, SEQ, HIDDEN) and arg111_1.is_contiguous()
    assert arg3_1.shape == (HIDDEN,) and arg3_1.is_contiguous()
    assert arg110_1.shape == (BATCH, SEQ, HIDDEN) and arg110_1.is_contiguous()
    assert arg337_1.shape == (BATCH, SEQ, 1) and arg337_1.is_contiguous()
    assert full_1.shape == ()
    assert arg103_1.shape == (BATCH, SEQ) and arg103_1.is_contiguous()
    assert arg109_1.shape == (BATCH, SEQ) and arg109_1.is_contiguous()
    assert arg108_1.shape == (BATCH, SEQ) and arg108_1.is_contiguous()
    assert arg107_1.shape == (BATCH, SEQ) and arg107_1.is_contiguous()
    assert arg106_1.shape == (BATCH, SEQ) and arg106_1.is_contiguous()
    assert arg105_1.shape == (BATCH, SEQ) and arg105_1.is_contiguous()
    assert arg104_1.shape == (BATCH, SEQ) and arg104_1.is_contiguous()
    assert arg2_1.shape == (1, SEQ) and arg2_1.is_contiguous()
    assert arg0_1.shape == (BATCH, SEQ) and arg0_1.is_contiguous()

    device = mm_1.device
    sum_saved = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_plain = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_segment = torch.empty((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_table_a = torch.empty((TABLE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_table_b = torch.empty((TABLE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_table_c = torch.empty((TABLE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_table_d = torch.empty((TABLE_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    total_init_elems = (
        2 * HIDDEN
        + SEGMENT_ROWS * HIDDEN
        + 4 * TABLE_ROWS * HIDDEN
        + POSITION_ROWS * HIDDEN
        + VOCAB_ROWS * HIDDEN
    )
    _init_outputs_kernel[(triton.cdiv(total_init_elems, INIT_BLOCK),)](
        mm_1,
        sum_saved,
        sum_plain,
        out_segment,
        out_table_a,
        out_table_b,
        out_table_c,
        out_table_d,
        out_position,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        POSITION_ROWS_=POSITION_ROWS,
        TABLE_ROWS_=TABLE_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_=INIT_BLOCK,
        num_warps=4,
    )

    _layoutlm_scatter_reduce_kernel[(ROWS,)](
        mm_142,
        mul_290,
        mm_144,
        mm_146,
        arg111_1,
        arg3_1,
        arg110_1,
        arg337_1,
        full_1,
        arg103_1,
        arg109_1,
        arg108_1,
        arg107_1,
        arg106_1,
        arg105_1,
        arg104_1,
        arg2_1,
        arg0_1,
        sum_saved,
        sum_plain,
        out_segment,
        out_table_a,
        out_table_b,
        out_table_c,
        out_table_d,
        out_position,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        POSITION_ROWS_=POSITION_ROWS,
        TABLE_ROWS_=TABLE_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=BLOCK_H,
        num_warps=8,
    )

    return (
        sum_saved,
        sum_plain,
        out_segment,
        out_table_a,
        out_table_b,
        out_table_c,
        out_table_d,
        out_position,
        out_vocab,
    )


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
