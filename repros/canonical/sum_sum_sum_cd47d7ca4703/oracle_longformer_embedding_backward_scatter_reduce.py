"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer backward-like scope, including both hidden-column reductions and all three duplicate-index accumulate=True embedding scatter-add outputs with the captured mask/where semantics, whereas Inductor currently schedules the rowwise producer, reductions, masks, and generic index_put scatter-adds as separate kernels; Inductor cannot do this today because scheduler/codegen has no embedding-backward scatter-reduce template that keeps the row-local hidden reductions live while feeding multiple sibling reductions and indexed accumulator destinations; the fix is SCATTER_REDUCE: add a multi-destination embedding-backward scatter-reduce lowering that emits the row math once, accumulates hidden-column reductions, and atomically scatters each masked embedding gradient directly into its destination row."""
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


HIDDEN = 768
BATCH = 8
SEQ = 1024
POSITION_ROWS = 4098
VOCAB_ROWS = 50265
BLOCK_H = 1024
ZERO_BLOCK = 1024


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _zero_outputs_kernel(
        sum_arg4_ptr,
        sum_plain_ptr,
        scatter_one_ptr,
        scatter_position_ptr,
        scatter_vocab_ptr,
        HIDDEN_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        BLOCK_N_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        zero = tl.zeros([BLOCK_N_], dtype=tl.float32)

        sum_arg4_size = HIDDEN_
        sum_plain_base = sum_arg4_size
        scatter_one_base = sum_plain_base + HIDDEN_
        scatter_position_base = scatter_one_base + HIDDEN_
        scatter_position_size = POSITION_ROWS_ * HIDDEN_
        scatter_vocab_base = scatter_position_base + scatter_position_size
        total = scatter_vocab_base + VOCAB_ROWS_ * HIDDEN_

        mask0 = offsets < sum_arg4_size
        tl.store(sum_arg4_ptr + offsets, zero, mask=mask0)

        off1 = offsets - sum_plain_base
        mask1 = (off1 >= 0) & (off1 < HIDDEN_)
        tl.store(sum_plain_ptr + off1, zero, mask=mask1)

        off2 = offsets - scatter_one_base
        mask2 = (off2 >= 0) & (off2 < HIDDEN_)
        tl.store(scatter_one_ptr + off2, zero, mask=mask2)

        off3 = offsets - scatter_position_base
        mask3 = (off3 >= 0) & (off3 < scatter_position_size)
        tl.store(scatter_position_ptr + off3, zero, mask=mask3)

        off4 = offsets - scatter_vocab_base
        mask4 = (off4 >= 0) & (offsets < total)
        tl.store(scatter_vocab_ptr + off4, zero, mask=mask4)

    @triton.jit
    def _longformer_embedding_bwd_kernel(
        mask_ptr,
        grad_ptr,
        weight_ptr,
        saved_ptr,
        row_scale_ptr,
        one_idx_ptr,
        position_idx_ptr,
        vocab_idx_ptr,
        sum_arg4_ptr,
        sum_plain_ptr,
        scatter_one_ptr,
        scatter_position_ptr,
        scatter_vocab_ptr,
        HIDDEN_: tl.constexpr,
        BATCH_: tl.constexpr,
        SEQ_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        seq = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        h_mask = h < HIDDEN_
        weight = tl.load(weight_ptr + h, mask=h_mask, other=0.0).to(tl.float32)

        acc_sum_arg4 = tl.zeros([BLOCK_H_], dtype=tl.float32)
        acc_sum_plain = tl.zeros([BLOCK_H_], dtype=tl.float32)
        acc_scatter_one = tl.zeros([BLOCK_H_], dtype=tl.float32)

        for batch in tl.static_range(0, BATCH_):
            row = batch * SEQ_ + seq
            offsets = row * HIDDEN_ + h

            active = tl.load(mask_ptr + offsets, mask=h_mask, other=0).to(tl.float32)
            grad = tl.load(grad_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)
            saved = tl.load(saved_ptr + offsets, mask=h_mask, other=0.0).to(tl.float32)

            masked_grad = grad * active
            weighted = masked_grad * weight
            row_sum = tl.sum(tl.where(h_mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(h_mask, weighted * saved, 0.0), axis=0)
            scale = tl.load(row_scale_ptr + row).to(tl.float32)
            scatter_value = scale * (weighted * HIDDEN_ - row_sum - saved * row_dot)

            acc_sum_arg4 += tl.where(h_mask, masked_grad * saved, 0.0)
            acc_sum_plain += tl.where(h_mask, masked_grad, 0.0)

            one_raw = tl.load(one_idx_ptr + row).to(tl.int64)
            one_wrapped = tl.where(one_raw < 0, one_raw + 1, one_raw)
            acc_scatter_one += tl.where((one_wrapped == 0) & h_mask, scatter_value, 0.0)

            position_raw = tl.load(position_idx_ptr + row).to(tl.int64)
            position_wrapped = tl.where(
                position_raw < 0,
                position_raw + POSITION_ROWS_,
                position_raw,
            )
            position_valid = (
                h_mask
                & (position_raw != 1)
                & (position_wrapped >= 0)
                & (position_wrapped < POSITION_ROWS_)
            )
            tl.atomic_add(
                scatter_position_ptr + position_wrapped * HIDDEN_ + h,
                scatter_value,
                sem="relaxed",
                mask=position_valid,
            )

            vocab_raw = tl.load(vocab_idx_ptr + row).to(tl.int64)
            vocab_wrapped = tl.where(vocab_raw < 0, vocab_raw + VOCAB_ROWS_, vocab_raw)
            vocab_valid = (
                h_mask
                & (vocab_raw != 1)
                & (vocab_wrapped >= 0)
                & (vocab_wrapped < VOCAB_ROWS_)
            )
            tl.atomic_add(
                scatter_vocab_ptr + vocab_wrapped * HIDDEN_ + h,
                scatter_value,
                sem="relaxed",
                mask=vocab_valid,
            )

        tl.atomic_add(sum_arg4_ptr + h, acc_sum_arg4, sem="relaxed", mask=h_mask)
        tl.atomic_add(sum_plain_ptr + h, acc_sum_plain, sem="relaxed", mask=h_mask)
        tl.atomic_add(scatter_one_ptr + h, acc_scatter_one, sem="relaxed", mask=h_mask)


@oracle_impl(hardware="H100", shapes="(T([8, 1024, 768], b8), T([8, 1024, 768], f32), T([768], f32), T([8, 1024, 768], f32), T([8, 1024, 1], f32), T([8, 1024], i64, gen=Index(1)), T([8, 1024], i64, gen=Index(4098)), T([8, 1024], i64, gen=Index(50265)))")
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
        mask,
        grad,
        weight,
        saved,
        row_scale,
        one_idx,
        position_idx,
        vocab_idx,
    ) = inputs
    if mask.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    assert mask.shape == (BATCH, SEQ, HIDDEN)
    assert grad.shape == (BATCH, SEQ, HIDDEN)
    assert weight.shape == (HIDDEN,)
    assert saved.shape == (BATCH, SEQ, HIDDEN)
    assert row_scale.shape == (BATCH, SEQ, 1)
    assert one_idx.shape == (BATCH, SEQ)
    assert position_idx.shape == (BATCH, SEQ)
    assert vocab_idx.shape == (BATCH, SEQ)

    device = grad.device
    sum_arg4 = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_plain = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    scatter_one = torch.empty((1, HIDDEN), device=device, dtype=torch.float32)
    scatter_position = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    scatter_vocab = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    total_zero_elems = (3 + POSITION_ROWS + VOCAB_ROWS) * HIDDEN
    _zero_outputs_kernel[(triton.cdiv(total_zero_elems, ZERO_BLOCK),)](
        sum_arg4,
        sum_plain,
        scatter_one,
        scatter_position,
        scatter_vocab,
        HIDDEN_=HIDDEN,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_N_=ZERO_BLOCK,
        num_warps=4,
    )
    _longformer_embedding_bwd_kernel[(SEQ,)](
        mask,
        grad,
        weight,
        saved,
        row_scale,
        one_idx,
        position_idx,
        vocab_idx,
        sum_arg4,
        sum_plain,
        scatter_one,
        scatter_position,
        scatter_vocab,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        SEQ_=SEQ,
        POSITION_ROWS_=POSITION_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        BLOCK_H_=BLOCK_H,
        num_warps=8,
    )
    return sum_arg4, sum_plain, scatter_one, scatter_position, scatter_vocab


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
