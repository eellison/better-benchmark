"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete ALBERT embedding-backward return tuple by fusing the shared rowwise hidden producer, both `[128]` sibling reductions, and all three duplicate-index `index_put(accumulate=True)` scatter-add outputs, whereas Inductor currently materializes the `[8, 512, 128]` producer and schedules the two reductions plus the 512-row, 2-row, and 30000-row scatter outputs as separate generic kernels; Inductor cannot do this today because scheduler/codegen has no structured scatter-reduce template that shares one row-local reduction producer across multiple indexed accumulation destinations and sibling column reductions; the fix is SCATTER_REDUCE: add an embedding-backward scatter-reduce lowering that keeps the row algebra in registers, emits the hidden-column reductions, and atomically accumulates every indexed embedding-gradient output directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401

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


REPRO_ID = "sum_sum_sum_8ffe5090cf87"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
SEQ = 512
HIDDEN = 128
ROWS = BATCH * SEQ
POSITION_ROWS = 512
SEGMENT_ROWS = 2
VOCAB_ROWS = 30000
HIDDEN_BLOCK = 128
FINAL_BLOCK_H = 16


if triton is not None:

    @triton.jit
    def _init_scatter_outputs_kernel(
        mm1_ptr,
        position_out_ptr,
        segment_out_ptr,
        vocab_out_ptr,
        TOTAL_POSITION: tl.constexpr,
        TOTAL_SEGMENT: tl.constexpr,
        TOTAL_VOCAB: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

        vocab_mask = offset < TOTAL_VOCAB
        vocab_value = tl.load(mm1_ptr + offset, mask=vocab_mask, other=0.0).to(
            tl.float32
        )
        tl.store(vocab_out_ptr + offset, vocab_value, mask=vocab_mask)

        position_mask = offset < TOTAL_POSITION
        tl.store(
            position_out_ptr + offset,
            tl.zeros([BLOCK], dtype=tl.float32),
            mask=position_mask,
        )

        segment_mask = offset < TOTAL_SEGMENT
        tl.store(
            segment_out_ptr + offset,
            tl.zeros([BLOCK], dtype=tl.float32),
            mask=segment_mask,
        )

    @triton.jit
    def _seq_scatter_reduce_kernel(
        mm148_ptr,
        gamma_ptr,
        saved_ptr,
        row_scale_ptr,
        position_idx_ptr,
        full_ptr,
        segment_idx_ptr,
        token_idx_ptr,
        partial_sum_saved_weighted_ptr,
        partial_sum_saved_ptr,
        position_out_ptr,
        segment_out_ptr,
        vocab_out_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        BATCH_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        SEGMENT_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)
        hidden_mask = h < HIDDEN_

        gamma = tl.load(gamma_ptr + h, mask=hidden_mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        acc_sum_saved_weighted = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_sum_saved = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc_position_grad = tl.zeros([BLOCK_H], dtype=tl.float32)

        for b in tl.static_range(0, BATCH_):
            row = b * SEQ_ + seq
            offsets = row * HIDDEN_ + h

            saved = tl.load(mm148_ptr + offsets, mask=hidden_mask, other=0.0).to(
                tl.float32
            )
            weighted_saved = saved * gamma
            norm_input = tl.load(saved_ptr + offsets, mask=hidden_mask, other=0.0).to(
                tl.float32
            )
            row_sum = tl.sum(tl.where(hidden_mask, weighted_saved, 0.0), axis=0)
            row_dot = tl.sum(
                tl.where(hidden_mask, weighted_saved * norm_input, 0.0), axis=0
            )
            scale = tl.load(row_scale_ptr + row).to(tl.float32)
            grad = scale * (weighted_saved * 128.0 - row_sum - norm_input * row_dot)

            acc_sum_saved_weighted += tl.where(
                hidden_mask, saved * norm_input, 0.0
            )
            acc_sum_saved += tl.where(hidden_mask, saved, 0.0)
            acc_position_grad += tl.where(hidden_mask, grad, 0.0)

            token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
            token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
            token_value = tl.where(token_raw == 0, full_value, grad)
            tl.atomic_add(
                vocab_out_ptr + token_wrapped * HIDDEN_ + h,
                token_value,
                sem="relaxed",
                mask=hidden_mask
                & (token_wrapped >= 0)
                & (token_wrapped < VOCAB_ROWS_),
            )

        position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
        position_wrapped = tl.where(
            position_raw < 0, position_raw + POSITION_ROWS_, position_raw
        )
        position_value = tl.where(position_raw == -1, full_value, acc_position_grad)
        tl.atomic_add(
            position_out_ptr + position_wrapped * HIDDEN_ + h,
            position_value,
            sem="relaxed",
            mask=hidden_mask
            & (position_wrapped >= 0)
            & (position_wrapped < POSITION_ROWS_),
        )

        segment_raw = tl.load(segment_idx_ptr + seq).to(tl.int64)
        segment_wrapped = tl.where(
            segment_raw < 0, segment_raw + SEGMENT_ROWS_, segment_raw
        )
        segment_value = tl.where(
            segment_raw == -1, full_value * BATCH_, acc_position_grad
        )
        tl.atomic_add(
            segment_out_ptr + segment_wrapped * HIDDEN_ + h,
            segment_value,
            sem="relaxed",
            mask=hidden_mask
            & (segment_wrapped >= 0)
            & (segment_wrapped < SEGMENT_ROWS_),
        )

        partial_offsets = seq * HIDDEN_ + h
        tl.store(
            partial_sum_saved_weighted_ptr + partial_offsets,
            acc_sum_saved_weighted,
            mask=hidden_mask,
        )
        tl.store(
            partial_sum_saved_ptr + partial_offsets,
            acc_sum_saved,
            mask=hidden_mask,
        )

    @triton.jit
    def _finalize_hidden_sums_kernel(
        partial_sum_saved_weighted_ptr,
        partial_sum_saved_ptr,
        out_sum_saved_weighted_ptr,
        out_sum_saved_ptr,
        SEQ_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
        n = tl.arange(0, BLOCK_N)
        mask = (n[:, None] < SEQ_) & (h[None, :] < HIDDEN_)
        offsets = n[:, None] * HIDDEN_ + h[None, :]

        sum_saved_weighted = tl.load(
            partial_sum_saved_weighted_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        sum_saved = tl.load(
            partial_sum_saved_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)

        out_mask = h < HIDDEN_
        tl.store(
            out_sum_saved_weighted_ptr + h,
            tl.sum(sum_saved_weighted, axis=0),
            mask=out_mask,
        )
        tl.store(out_sum_saved_ptr + h, tl.sum(sum_saved, axis=0), mask=out_mask)


def get_inputs() -> tuple[Any, ...]:
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR)


def oracle_embedding_scatter_reduce(
    mm_148: torch.Tensor,
    arg3_1: torch.Tensor,
    arg17_1: torch.Tensor,
    arg184_1: torch.Tensor,
    arg1_1: torch.Tensor,
    full_1: torch.Tensor,
    arg16_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if mm_148.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if list(_shape_param_0) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected shape parameter 0: {_shape_param_0}")
    if list(_shape_param_1) != [BATCH, SEQ]:
        raise ValueError(f"unexpected shape parameter 1: {_shape_param_1}")

    mm_148 = mm_148.contiguous()
    arg3_1 = arg3_1.contiguous()
    arg17_1 = arg17_1.contiguous()
    arg184_1 = arg184_1.contiguous()
    arg1_1 = arg1_1.contiguous()
    full_1 = full_1.contiguous()
    arg16_1 = arg16_1.contiguous()
    arg0_1 = arg0_1.contiguous()
    mm_1 = mm_1.contiguous()

    assert mm_148.shape == (ROWS, HIDDEN)
    assert arg3_1.shape == (HIDDEN,)
    assert arg17_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg184_1.shape == (BATCH, SEQ, 1)
    assert arg1_1.shape == (1, SEQ)
    assert full_1.shape == ()
    assert arg16_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm_1.shape == (VOCAB_ROWS, HIDDEN)

    device = mm_148.device
    partial_sum_saved_weighted = torch.empty(
        (SEQ, HIDDEN), device=device, dtype=torch.float32
    )
    partial_sum_saved = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    out_sum_saved_weighted = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_sum_saved = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty(
        (POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32
    )
    segment_out = torch.empty(
        (SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32
    )
    vocab_out = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    init_block = 1024
    total_position = POSITION_ROWS * HIDDEN
    total_segment = SEGMENT_ROWS * HIDDEN
    total_vocab = VOCAB_ROWS * HIDDEN
    _init_scatter_outputs_kernel[(triton.cdiv(total_vocab, init_block),)](
        mm_1,
        position_out,
        segment_out,
        vocab_out,
        TOTAL_POSITION=total_position,
        TOTAL_SEGMENT=total_segment,
        TOTAL_VOCAB=total_vocab,
        BLOCK=init_block,
        num_warps=8,
    )

    _seq_scatter_reduce_kernel[(SEQ,)](
        mm_148,
        arg3_1,
        arg17_1,
        arg184_1,
        arg1_1,
        full_1,
        arg16_1,
        arg0_1,
        partial_sum_saved_weighted,
        partial_sum_saved,
        position_out,
        segment_out,
        vocab_out,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BATCH_=BATCH,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_H=HIDDEN_BLOCK,
        num_warps=4,
    )

    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_H),)](
        partial_sum_saved_weighted,
        partial_sum_saved,
        out_sum_saved_weighted,
        out_sum_saved,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BLOCK_N=triton.next_power_of_2(SEQ),
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=8,
    )

    return (
        out_sum_saved_weighted,
        out_sum_saved,
        position_out,
        segment_out,
        vocab_out,
    )


@oracle_impl(hardware="H100", shapes="(T([4096, 128], f32), T([128], f32), T([8, 512, 128], f32), T([8, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([1, 512], i64, gen=Index(2)), T([8, 512], i64, gen=Index(30000)), T([30000, 128], f32), S([8, 512, 128]), S([8, 512]))")
def oracle_forward(inputs: tuple[Any, ...]):
    return oracle_embedding_scatter_reduce(*inputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check", action="store_true", help="Verify correctness against eager Repro"
    )
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument(
        "--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check"
    )
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt"
    )
    parser.add_argument(
        "--show-hw", action="store_true", help="Print GPU hardware info and exit"
    )
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(
                    f"WARNING: oracle is slower than compile "
                    f"(ratio={result['ratio']:.3f}x)"
                )


if __name__ == "__main__":
    main()
