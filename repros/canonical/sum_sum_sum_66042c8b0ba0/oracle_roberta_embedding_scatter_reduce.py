"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full RoBERTa embedding/layernorm-backward row producer once, including the two hidden reductions and all three `index_put(accumulate=True)` scatter-add outputs with the `mm_1` vocabulary add, whereas Inductor currently materializes the `[4,512,768]` producer and schedules the sibling sums, sentinel `where` masks, zero fills, three indexed accumulators, and vocabulary add as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that keeps row-local layernorm reduction scalars live while feeding multiple indexed accumulator destinations of different sizes plus sibling hidden reductions; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that fuses the rowwise producer with hidden reductions, sentinel guards, indexed atomic accumulation, and vocabulary-gradient accumulation for the full return tuple."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
POSITION_ROWS = 514
SEGMENT_ROWS = 1
VOCAB_ROWS = 250002
DROPOUT_SCALE = 1.1111111111111112
INIT_BLOCK = 1024
HIDDEN_BLOCK = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        mm1_ptr,
        out_sum_weighted_xhat_ptr,
        out_sum_weighted_ptr,
        out_position_ptr,
        out_segment_ptr,
        out_vocab_ptr,
        TOTAL_POSITION_: tl.constexpr,
        TOTAL_SEGMENT_: tl.constexpr,
        TOTAL_VOCAB_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        zeros = tl.zeros([BLOCK_], dtype=tl.float32)

        hidden_mask = offsets < HIDDEN_
        tl.store(out_sum_weighted_xhat_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(out_sum_weighted_ptr + offsets, zeros, mask=hidden_mask)

        position_mask = offsets < TOTAL_POSITION_
        tl.store(out_position_ptr + offsets, zeros, mask=position_mask)

        segment_mask = offsets < TOTAL_SEGMENT_
        tl.store(out_segment_ptr + offsets, zeros, mask=segment_mask)

        vocab_mask = offsets < TOTAL_VOCAB_
        vocab_value = tl.load(mm1_ptr + offsets, mask=vocab_mask, other=0.0)
        tl.store(out_vocab_ptr + offsets, vocab_value, mask=vocab_mask)

    @triton.jit
    def _row_scatter_reduce_kernel(
        mm142_ptr,
        mul345_ptr,
        mm144_ptr,
        mm146_ptr,
        dropout_mask_ptr,
        weight_ptr,
        saved_x_ptr,
        row_scale_ptr,
        position_index_ptr,
        full_ptr,
        segment_index_ptr,
        token_index_ptr,
        out_sum_weighted_xhat_ptr,
        out_sum_weighted_ptr,
        out_position_ptr,
        out_segment_ptr,
        out_vocab_ptr,
        HIDDEN_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        SEGMENT_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        mask = h < HIDDEN_
        offsets = row * HIDDEN_ + h

        source = (
            tl.load(mul345_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm142_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm144_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm146_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        weighted_pre = source * keep * DROPOUT_SCALE_
        weight = tl.load(weight_ptr + h, mask=mask, other=0.0).to(tl.float32)
        saved_x = tl.load(saved_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weighted = weighted_pre * weight

        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, weighted * saved_x, 0.0), axis=0)
        scale = tl.load(row_scale_ptr + row).to(tl.float32)
        grad = scale * (weighted * 768.0 - row_sum - saved_x * row_dot)

        tl.atomic_add(
            out_sum_weighted_xhat_ptr + h,
            weighted_pre * saved_x,
            sem="relaxed",
            mask=mask,
        )
        tl.atomic_add(
            out_sum_weighted_ptr + h,
            weighted_pre,
            sem="relaxed",
            mask=mask,
        )

        full_value = tl.load(full_ptr).to(tl.float32)

        position_raw = tl.load(position_index_ptr + row).to(tl.int64)
        position_wrapped = tl.where(
            position_raw < 0,
            position_raw + POSITION_ROWS_,
            position_raw,
        )
        position_value = tl.where(position_raw == 1, full_value, grad)
        tl.atomic_add(
            out_position_ptr + position_wrapped * HIDDEN_ + h,
            position_value,
            sem="relaxed",
            mask=mask
            & (position_wrapped >= 0)
            & (position_wrapped < POSITION_ROWS_),
        )

        segment_raw = tl.load(segment_index_ptr + row).to(tl.int64)
        segment_wrapped = tl.where(
            segment_raw < 0,
            segment_raw + SEGMENT_ROWS_,
            segment_raw,
        )
        segment_value = tl.where(segment_raw == -1, full_value, grad)
        tl.atomic_add(
            out_segment_ptr + segment_wrapped * HIDDEN_ + h,
            segment_value,
            sem="relaxed",
            mask=mask
            & (segment_wrapped >= 0)
            & (segment_wrapped < SEGMENT_ROWS_),
        )

        token_raw = tl.load(token_index_ptr + row).to(tl.int64)
        token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
        token_value = tl.where(token_raw == 1, full_value, grad)
        tl.atomic_add(
            out_vocab_ptr + token_wrapped * HIDDEN_ + h,
            token_value,
            sem="relaxed",
            mask=mask & (token_wrapped >= 0) & (token_wrapped < VOCAB_ROWS_),
        )


def oracle_roberta_embedding_scatter_reduce(
    mm_142: torch.Tensor,
    mul_345: torch.Tensor,
    mm_144: torch.Tensor,
    mm_146: torch.Tensor,
    arg105_1: torch.Tensor,
    arg2_1: torch.Tensor,
    arg104_1: torch.Tensor,
    arg323_1: torch.Tensor,
    arg102_1: torch.Tensor,
    full_1: torch.Tensor,
    arg103_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm_142.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 0: {_shape_param_0}")
    if list(_shape_param_1) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 1: {_shape_param_1}")
    if list(_shape_param_2) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 2: {_shape_param_2}")

    assert mm_142.shape == (ROWS, HIDDEN)
    assert mul_345.shape == (BATCH, SEQ, HIDDEN)
    assert mm_144.shape == (ROWS, HIDDEN)
    assert mm_146.shape == (ROWS, HIDDEN)
    assert arg105_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg2_1.shape == (HIDDEN,)
    assert arg104_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg323_1.shape == (BATCH, SEQ, 1)
    assert arg102_1.shape == (BATCH, SEQ)
    assert full_1.shape == ()
    assert arg103_1.shape == (BATCH, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm_1.shape == (VOCAB_ROWS, HIDDEN)

    device = mm_142.device
    out_sum_weighted_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_weighted = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_segment = torch.empty((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    total_position = POSITION_ROWS * HIDDEN
    total_segment = SEGMENT_ROWS * HIDDEN
    total_vocab = VOCAB_ROWS * HIDDEN
    _init_outputs_kernel[(triton.cdiv(total_vocab, INIT_BLOCK),)](
        mm_1,
        out_sum_weighted_xhat,
        out_sum_weighted,
        out_position,
        out_segment,
        out_vocab,
        TOTAL_POSITION_=total_position,
        TOTAL_SEGMENT_=total_segment,
        TOTAL_VOCAB_=total_vocab,
        HIDDEN_=HIDDEN,
        BLOCK_=INIT_BLOCK,
        num_warps=8,
    )

    _row_scatter_reduce_kernel[(ROWS,)](
        mm_142,
        mul_345,
        mm_144,
        mm_146,
        arg105_1,
        arg2_1,
        arg104_1,
        arg323_1,
        arg102_1,
        full_1,
        arg103_1,
        arg0_1,
        out_sum_weighted_xhat,
        out_sum_weighted,
        out_position,
        out_segment,
        out_vocab,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=HIDDEN_BLOCK,
        num_warps=8,
    )
    return (
        out_sum_weighted_xhat,
        out_sum_weighted,
        out_position,
        out_segment,
        out_vocab,
    )


@oracle_impl(hardware="H100", shapes="(T([2048, 768], f32), T([4, 512, 768], f32), T([2048, 768], f32), T([2048, 768], f32), T([4, 512, 768], b8), T([768], f32), T([4, 512, 768], f32), T([4, 512, 1], f32), T([4, 512], i64, gen=Index(514)), T([], f32), T([4, 512], i64, gen=Index(1)), T([4, 512], i64, gen=Index(250002)), T([250002, 768], f32), S([4, 512, 768]), S([4, 512, 768]), S([4, 512, 768]))")
def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    return oracle_roberta_embedding_scatter_reduce(*inputs)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
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
