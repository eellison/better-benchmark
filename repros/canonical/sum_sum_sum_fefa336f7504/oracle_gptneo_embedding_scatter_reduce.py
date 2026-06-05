"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete GPT-Neo layernorm-backward embedding-gradient return tuple, including the two hidden-column reductions, position-embedding scatter-add, and token-embedding scatter-add added to `mm[:50257]`, whereas Inductor currently materializes the rowwise layernorm-backward producer and schedules the sibling reductions plus duplicate-index `index_put(accumulate=True)` outputs as separate generic kernels; Inductor cannot do this today because scheduler/codegen lacks an embedding-backward scatter-reduce lowering that keeps row-local reduction scalars live while feeding multiple indexed accumulator destinations and sibling column reductions; the fix is SCATTER_REDUCE: add a structured embedding-backward scatter-reduce template that computes each row producer once, accumulates hidden reductions, and scatters position and token gradients directly into their dense destinations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
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

BATCH = 32
SEQ = 128
HIDDEN = 2048
ROWS = BATCH * SEQ
VOCAB = 50257
POSITION_ROWS = 2048
BLOCK_HIDDEN = 2048
INIT_BLOCK = 1024
FINAL_BLOCK_H = 32


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _init_embedding_outputs_kernel(
        mm_ptr,
        position_out_ptr,
        vocab_out_ptr,
        total_vocab: tl.constexpr,
        total_position: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        vocab_mask = offsets < total_vocab
        position_mask = offsets < total_position
        values = tl.load(mm_ptr + offsets, mask=vocab_mask, other=0.0)
        tl.store(vocab_out_ptr + offsets, values, mask=vocab_mask)
        tl.store(position_out_ptr + offsets, tl.zeros((BLOCK,), dtype=tl.float32), mask=position_mask)

    @triton.jit
    def _sequence_layernorm_scatter_kernel(
        mm285_ptr,
        mm287_ptr,
        mm289_ptr,
        gamma_ptr,
        saved_ptr,
        position_base_ptr,
        mean_ptr,
        invstd_ptr,
        residual_grad_ptr,
        position_idx_ptr,
        fill_value_ptr,
        token_idx_ptr,
        partial_xhat_sum_ptr,
        partial_plain_sum_ptr,
        position_out_ptr,
        vocab_out_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        BATCH_: tl.constexpr,
        VOCAB_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.arange(0, BLOCK_H)
        h_mask = h < HIDDEN_

        gamma = tl.load(gamma_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
        position_base = tl.load(
            position_base_ptr + seq * HIDDEN_ + h,
            mask=h_mask,
            other=0.0,
        ).to(tl.float32)
        fill_value = tl.load(fill_value_ptr).to(tl.float32)

        acc_xhat_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc_plain_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
        acc_position = tl.zeros((BLOCK_H,), dtype=tl.float32)

        for batch in tl.range(0, BATCH_):
            row = batch * SEQ_ + seq
            offsets = row * HIDDEN_ + h

            x = (
                tl.load(mm285_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
                + tl.load(mm287_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
                + tl.load(mm289_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            )
            mean = tl.load(mean_ptr + row).to(tl.float32)
            invstd = tl.load(invstd_ptr + row).to(tl.float32)
            saved = tl.load(saved_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            normalized = (saved + position_base - mean) * invstd
            weighted = x * gamma

            row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(h_mask, weighted * normalized, 0.0), axis=0)
            ln_grad = (invstd / HIDDEN_) * (weighted * HIDDEN_ - row_sum - normalized * row_dot)
            residual_grad = tl.load(
                residual_grad_ptr + offsets,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            embedding_grad = residual_grad + ln_grad

            acc_xhat_sum += tl.where(h_mask, x * normalized, 0.0)
            acc_plain_sum += tl.where(h_mask, x, 0.0)
            acc_position += tl.where(h_mask, embedding_grad, 0.0)

            token_raw = tl.load(token_idx_ptr + row).to(tl.int64)
            token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_, token_raw)
            token_valid = (token_wrapped >= 0) & (token_wrapped < VOCAB_)
            token_value = tl.where(token_raw == -1, fill_value, embedding_grad)
            tl.atomic_add(
                vocab_out_ptr + token_wrapped * HIDDEN_ + h,
                token_value,
                sem="relaxed",
                mask=h_mask & token_valid,
            )

        position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
        position_wrapped = tl.where(position_raw < 0, position_raw + POSITION_ROWS_, position_raw)
        position_valid = (position_wrapped >= 0) & (position_wrapped < POSITION_ROWS_)
        position_value = tl.where(position_raw == -1, fill_value, acc_position)
        tl.atomic_add(
            position_out_ptr + position_wrapped * HIDDEN_ + h,
            position_value,
            sem="relaxed",
            mask=h_mask & position_valid,
        )

        partial_offsets = seq * HIDDEN_ + h
        tl.store(partial_xhat_sum_ptr + partial_offsets, acc_xhat_sum, mask=h_mask)
        tl.store(partial_plain_sum_ptr + partial_offsets, acc_plain_sum, mask=h_mask)

    @triton.jit
    def _finalize_hidden_sums_kernel(
        partial_xhat_sum_ptr,
        partial_plain_sum_ptr,
        out_xhat_sum_ptr,
        out_plain_sum_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
        seq = tl.arange(0, BLOCK_S)
        mask = (seq[:, None] < SEQ_) & (h[None, :] < HIDDEN_)
        offsets = seq[:, None] * HIDDEN_ + h[None, :]

        xhat_values = tl.load(partial_xhat_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        plain_values = tl.load(partial_plain_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out_mask = h < HIDDEN_
        tl.store(out_xhat_sum_ptr + h, tl.sum(xhat_values, axis=0), mask=out_mask)
        tl.store(out_plain_sum_ptr + h, tl.sum(plain_values, axis=0), mask=out_mask)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} expected shape={shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise ValueError(f"{name} expected dtype={dtype}, got {value.dtype}")
    if value.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous for this shape-specialized oracle")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 16:
        raise ValueError(f"expected 16 inputs, got {len(inputs)}")

    (
        mm,
        mm_285,
        mm_287,
        mm_289,
        arg2_1,
        arg219_1,
        arg221_1,
        arg222_1,
        arg223_1,
        add_189,
        arg220_1,
        full_1,
        arg0_1,
        shape0,
        shape1,
        shape2,
    ) = inputs

    if list(shape0) != [BATCH, SEQ, HIDDEN] or list(shape1) != [BATCH, SEQ, HIDDEN] or list(shape2) != [BATCH, SEQ, HIDDEN]:
        raise ValueError(f"unexpected view shape params: {shape0}, {shape1}, {shape2}")

    tensors = (
        _require_tensor("mm", mm, (VOCAB + 3, HIDDEN), torch.float32),
        _require_tensor("mm_285", mm_285, (ROWS, HIDDEN), torch.float32),
        _require_tensor("mm_287", mm_287, (ROWS, HIDDEN), torch.float32),
        _require_tensor("mm_289", mm_289, (ROWS, HIDDEN), torch.float32),
        _require_tensor("arg2_1", arg2_1, (HIDDEN,), torch.float32),
        _require_tensor("arg219_1", arg219_1, (BATCH, SEQ, HIDDEN), torch.float32),
        _require_tensor("arg221_1", arg221_1, (1, SEQ, HIDDEN), torch.float32),
        _require_tensor("arg222_1", arg222_1, (BATCH, SEQ, 1), torch.float32),
        _require_tensor("arg223_1", arg223_1, (BATCH, SEQ, 1), torch.float32),
        _require_tensor("add_189", add_189, (BATCH, SEQ, HIDDEN), torch.float32),
        _require_tensor("arg220_1", arg220_1, (1, SEQ), torch.int64),
        _require_tensor("full_1", full_1, (), torch.float32),
        _require_tensor("arg0_1", arg0_1, (BATCH, SEQ), torch.int64),
    )
    return tensors


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full GPT-Neo layernorm-backward and embedding scatter-reduce scope."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        mm,
        mm_285,
        mm_287,
        mm_289,
        arg2_1,
        arg219_1,
        arg221_1,
        arg222_1,
        arg223_1,
        add_189,
        arg220_1,
        full_1,
        arg0_1,
    ) = _validate_inputs(inputs)

    device = mm.device
    out_xhat_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_plain_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = torch.empty((VOCAB, HIDDEN), device=device, dtype=torch.float32)
    partial_xhat_sum = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)
    partial_plain_sum = torch.empty((SEQ, HIDDEN), device=device, dtype=torch.float32)

    total_vocab = VOCAB * HIDDEN
    total_position = POSITION_ROWS * HIDDEN
    _init_embedding_outputs_kernel[(triton.cdiv(total_vocab, INIT_BLOCK),)](
        mm,
        position_out,
        vocab_out,
        total_vocab=total_vocab,
        total_position=total_position,
        BLOCK=INIT_BLOCK,
        num_warps=4,
    )
    _sequence_layernorm_scatter_kernel[(SEQ,)](
        mm_285,
        mm_287,
        mm_289,
        arg2_1,
        arg219_1,
        arg221_1,
        arg222_1,
        arg223_1,
        add_189,
        arg220_1,
        full_1,
        arg0_1,
        partial_xhat_sum,
        partial_plain_sum,
        position_out,
        vocab_out,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BATCH_=BATCH,
        VOCAB_=VOCAB,
        POSITION_ROWS_=POSITION_ROWS,
        BLOCK_H=BLOCK_HIDDEN,
        num_warps=8,
    )
    _finalize_hidden_sums_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_H),)](
        partial_xhat_sum,
        partial_plain_sum,
        out_xhat_sum,
        out_plain_sum,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BLOCK_S=SEQ,
        BLOCK_H=FINAL_BLOCK_H,
        num_warps=4,
    )

    return out_xhat_sum, out_plain_sum, position_out, vocab_out


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
