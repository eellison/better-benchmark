"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DistilBERT layernorm-backward/dropout return tuple, including both hidden-column reductions, the position-embedding scatter-add, and the vocabulary-gradient add into `mm_1[:30522]`, by recomputing the row producer in Triton and scattering directly into the returned dense outputs, whereas Inductor currently materializes the `[256,128,768]` rowwise producer and schedules the sibling reductions, sentinel `where` masks, zero fills, generic `index_put(accumulate=True)` buffers, and final base add as separate kernels; Inductor lacks an embedding-backward scatter-reduce template that could express this directly, but template `bench_oracle()` timing shows the full-scope direct-scatter oracle is only at floor rather than a meaningful speedup; the fix is BANDWIDTH_BOUND: no performance fix is justified for this repro, though a structured embedding scatter-reduce lowering may still simplify related cases where the scatter work is not dominated by memory traffic."""
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
    has_stochastic_ops,
)


BATCH = 256
SEQ = 128
HIDDEN = 768
ROWS = BATCH * SEQ
VOCAB_ROWS = 30522
POSITION_ROWS = 512
DROPOUT_SCALE = 1.1111111111111112
INV_HIDDEN = 1.0 / HIDDEN

ROW_BLOCK_H = 1024
INIT_BLOCK = 1024
SEQ_SCATTER_BLOCK_B = 16
SEQ_SCATTER_BLOCK_C = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _init_tiled_outputs_kernel(
        mm1_ptr,
        out_weighted_xhat_ptr,
        out_weighted_ptr,
        position_out_ptr,
        vocab_out_ptr,
        POSITION_ELEMS_: tl.constexpr,
        VOCAB_ELEMS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        zeros = tl.zeros([BLOCK_], dtype=tl.float32)

        hidden_mask = offsets < HIDDEN_
        tl.store(out_weighted_xhat_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(out_weighted_ptr + offsets, zeros, mask=hidden_mask)

        position_offsets = offsets - HIDDEN_
        position_mask = (position_offsets >= 0) & (position_offsets < POSITION_ELEMS_)
        tl.store(position_out_ptr + position_offsets, zeros, mask=position_mask)

        vocab_offsets = offsets - HIDDEN_ - POSITION_ELEMS_
        vocab_mask = (vocab_offsets >= 0) & (vocab_offsets < VOCAB_ELEMS_)
        vocab_values = tl.load(mm1_ptr + vocab_offsets, mask=vocab_mask, other=0.0).to(tl.float32)
        tl.store(vocab_out_ptr + vocab_offsets, vocab_values, mask=vocab_mask)

    @triton.jit
    def _row_stats_kernel(
        mm70_ptr,
        mul167_ptr,
        mm72_ptr,
        mm74_ptr,
        dropout_mask_ptr,
        weight_ptr,
        saved_ptr,
        position_saved_ptr,
        mean_ptr,
        invstd_ptr,
        row_sum_ptr,
        row_dot_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
    ):
        row = tl.program_id(0)
        h = tl.arange(0, BLOCK_H_)
        mask = h < HIDDEN_
        seq = row - (row // SEQ_) * SEQ_
        offsets = row * HIDDEN_ + h
        position_offsets = seq * HIDDEN_ + h

        source = tl.load(mul167_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        source += tl.load(mm70_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        source += tl.load(mm72_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        source += tl.load(mm74_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
        weighted_pre = source * keep * DROPOUT_SCALE_
        weight = tl.load(weight_ptr + h, mask=mask, other=0.0).to(tl.float32)
        weighted = weighted_pre * weight

        mean = tl.load(mean_ptr + row).to(tl.float32)
        invstd = tl.load(invstd_ptr + row).to(tl.float32)
        saved = tl.load(saved_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        position_saved = tl.load(position_saved_ptr + position_offsets, mask=mask, other=0.0).to(tl.float32)
        xhat = (saved + position_saved - mean) * invstd

        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
        row_dot = tl.sum(tl.where(mask, weighted * xhat, 0.0), axis=0)
        tl.store(row_sum_ptr + row, row_sum)
        tl.store(row_dot_ptr + row, row_dot)

    @triton.jit
    def _seq_scatter_reduce_kernel(
        mm70_ptr,
        mul167_ptr,
        mm72_ptr,
        mm74_ptr,
        dropout_mask_ptr,
        weight_ptr,
        saved_ptr,
        position_saved_ptr,
        mean_ptr,
        invstd_ptr,
        position_idx_ptr,
        full_ptr,
        token_idx_ptr,
        row_sum_ptr,
        row_dot_ptr,
        out_weighted_xhat_ptr,
        out_weighted_ptr,
        position_out_ptr,
        vocab_out_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        BATCH_: tl.constexpr,
        VOCAB_ROWS_: tl.constexpr,
        POSITION_ROWS_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        INV_HIDDEN_: tl.constexpr,
        BLOCK_B_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        seq = tl.program_id(1)
        pid_b = tl.program_id(2)
        batch_offsets = pid_b * BLOCK_B_ + tl.arange(0, BLOCK_B_)
        cols = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        rows = batch_offsets * SEQ_ + seq
        active = (batch_offsets[:, None] < BATCH_) & (cols[None, :] < HIDDEN_)
        row_active = batch_offsets < BATCH_

        offsets = rows[:, None] * HIDDEN_ + cols[None, :]
        position_saved_offsets = seq * HIDDEN_ + cols

        source = tl.load(mm70_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        source += tl.load(mul167_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        source += tl.load(mm72_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        source += tl.load(mm74_ptr + offsets, mask=active, other=0.0).to(tl.float32)

        keep = tl.load(dropout_mask_ptr + offsets, mask=active, other=0).to(tl.float32)
        weighted_pre = source * keep * DROPOUT_SCALE_
        weight = tl.load(weight_ptr + cols, mask=cols < HIDDEN_, other=0.0).to(tl.float32)
        weighted = weighted_pre * weight[None, :]

        mean = tl.load(mean_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
        saved = tl.load(saved_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        position_saved = tl.load(
            position_saved_ptr + position_saved_offsets,
            mask=cols < HIDDEN_,
            other=0.0,
        ).to(tl.float32)
        xhat = (saved + position_saved[None, :] - mean[:, None]) * invstd[:, None]

        row_sum = tl.load(row_sum_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
        row_dot = tl.load(row_dot_ptr + rows, mask=row_active, other=0.0).to(tl.float32)
        dx = (invstd[:, None] * INV_HIDDEN_) * (
            weighted * HIDDEN_ - row_sum[:, None] - xhat * row_dot[:, None]
        )
        dx = tl.where(active, dx, 0.0)

        channel_mask = cols < HIDDEN_
        tl.atomic_add(
            out_weighted_xhat_ptr + cols,
            tl.sum(tl.where(active, weighted_pre * xhat, 0.0), axis=0),
            sem="relaxed",
            mask=channel_mask,
        )
        tl.atomic_add(
            out_weighted_ptr + cols,
            tl.sum(tl.where(active, weighted_pre, 0.0), axis=0),
            sem="relaxed",
            mask=channel_mask,
        )

        full_value = tl.load(full_ptr).to(tl.float32)

        position_raw = tl.load(position_idx_ptr + seq).to(tl.int64)
        position_wrapped = tl.where(
            position_raw < 0,
            position_raw + POSITION_ROWS_,
            position_raw,
        )
        position_value = tl.where(
            position_raw == -1,
            tl.where(batch_offsets[:, None] == 0, full_value, 0.0),
            dx,
        )
        position_valid = (
            channel_mask
            & (position_wrapped >= 0)
            & (position_wrapped < POSITION_ROWS_)
        )
        tl.atomic_add(
            position_out_ptr + position_wrapped * HIDDEN_ + cols,
            tl.sum(tl.where(active, position_value, 0.0), axis=0),
            sem="relaxed",
            mask=position_valid,
        )

        token_raw = tl.load(token_idx_ptr + rows, mask=row_active, other=0).to(tl.int64)
        token_wrapped = tl.where(token_raw < 0, token_raw + VOCAB_ROWS_, token_raw)
        token_value = tl.where(token_raw[:, None] == 0, full_value, dx)
        token_valid = (
            active
            & (token_wrapped[:, None] >= 0)
            & (token_wrapped[:, None] < VOCAB_ROWS_)
        )
        tl.atomic_add(
            vocab_out_ptr + token_wrapped[:, None] * HIDDEN_ + cols[None, :],
            token_value,
            sem="relaxed",
            mask=token_valid,
        )

def _check_shape_params(shape0, shape1, shape2) -> None:
    expected = [BATCH, SEQ, HIDDEN]
    for idx, shape in enumerate((shape0, shape1, shape2)):
        if list(shape) != expected:
            raise ValueError(f"unexpected shape parameter {idx}: {shape}")


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
        mm_70,
        mul_167,
        mm_72,
        mm_74,
        arg59_1,
        arg3_1,
        arg55_1,
        arg56_1,
        arg57_1,
        arg58_1,
        arg2_1,
        full_1,
        arg0_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    if mm_1.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    _check_shape_params(_shape_param_0, _shape_param_1, _shape_param_2)

    assert mm_1.shape == (VOCAB_ROWS + 2, HIDDEN) and mm_1.is_contiguous()
    assert mm_70.shape == (ROWS, HIDDEN) and mm_70.is_contiguous()
    assert mul_167.shape == (BATCH, SEQ, HIDDEN) and mul_167.is_contiguous()
    assert mm_72.shape == (ROWS, HIDDEN) and mm_72.is_contiguous()
    assert mm_74.shape == (ROWS, HIDDEN) and mm_74.is_contiguous()
    assert arg59_1.shape == (BATCH, SEQ, HIDDEN) and arg59_1.is_contiguous()
    assert arg3_1.shape == (HIDDEN,) and arg3_1.is_contiguous()
    assert arg55_1.shape == (BATCH, SEQ, HIDDEN) and arg55_1.is_contiguous()
    assert arg56_1.shape == (1, SEQ, HIDDEN) and arg56_1.is_contiguous()
    assert arg57_1.shape == (BATCH, SEQ, 1) and arg57_1.is_contiguous()
    assert arg58_1.shape == (BATCH, SEQ, 1) and arg58_1.is_contiguous()
    assert arg2_1.shape == (1, POSITION_ROWS) and arg2_1.is_contiguous()
    assert full_1.shape == ()
    assert arg0_1.shape == (BATCH, SEQ) and arg0_1.is_contiguous()

    device = mm_1.device
    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    out_weighted_xhat = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_weighted = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    position_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    vocab_out = torch.empty((VOCAB_ROWS, HIDDEN), device=device, dtype=torch.float32)

    position_elems = POSITION_ROWS * HIDDEN
    vocab_elems = VOCAB_ROWS * HIDDEN
    _init_tiled_outputs_kernel[
        (triton.cdiv(HIDDEN + position_elems + vocab_elems, INIT_BLOCK),)
    ](
        mm_1,
        out_weighted_xhat,
        out_weighted,
        position_out,
        vocab_out,
        POSITION_ELEMS_=position_elems,
        VOCAB_ELEMS_=vocab_elems,
        HIDDEN_=HIDDEN,
        BLOCK_=INIT_BLOCK,
        num_warps=4,
    )

    _row_stats_kernel[(ROWS,)](
        mm_70,
        mul_167,
        mm_72,
        mm_74,
        arg59_1,
        arg3_1,
        arg55_1,
        arg56_1,
        arg57_1,
        arg58_1,
        row_sum,
        row_dot,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        BLOCK_H_=ROW_BLOCK_H,
        num_warps=8,
    )

    _seq_scatter_reduce_kernel[
        (triton.cdiv(HIDDEN, SEQ_SCATTER_BLOCK_C), SEQ, triton.cdiv(BATCH, SEQ_SCATTER_BLOCK_B))
    ](
        mm_70,
        mul_167,
        mm_72,
        mm_74,
        arg59_1,
        arg3_1,
        arg55_1,
        arg56_1,
        arg57_1,
        arg58_1,
        arg2_1,
        full_1,
        arg0_1,
        row_sum,
        row_dot,
        out_weighted_xhat,
        out_weighted,
        position_out,
        vocab_out,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        BATCH_=BATCH,
        VOCAB_ROWS_=VOCAB_ROWS,
        POSITION_ROWS_=POSITION_ROWS,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        INV_HIDDEN_=INV_HIDDEN,
        BLOCK_B_=SEQ_SCATTER_BLOCK_B,
        BLOCK_C_=SEQ_SCATTER_BLOCK_C,
        num_warps=4,
    )

    return out_weighted_xhat, out_weighted, position_out, vocab_out


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
