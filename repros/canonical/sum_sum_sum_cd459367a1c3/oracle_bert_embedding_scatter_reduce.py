"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full BERT embedding/layernorm-backward tuple as a structured Triton scatter-reduce: one initialization pass copies the sliced embedding weight and zeros the accumulator outputs, and one row-wise pass forms the dropout-scaled residual sum, performs the two per-token layernorm reductions, emits the two hidden reductions, and accumulates the three indexed outputs without materializing the `[32, 512, 768]` gradient intermediate or the generic `index_put(accumulate=True)` tensors. Inductor currently schedules the sibling reductions and duplicate-index `index_put` consumers as separate materialized operations, so it cannot keep the shared rowwise producer live across the indexed scatter reductions and dense reductions. The fix is SCATTER_REDUCE: add scheduler/codegen support for recognizing this embedding-backward gather/scatter-reduce pattern and generating a fused multi-output reduction with indexed accumulator epilogues."""
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


# --- Oracle kernel(s) ---

BATCH = 32
SEQ = 512
HIDDEN = 768
ROWS = BATCH * SEQ
VOCAB = 30522
MM1_ROWS = 30524
INIT_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _init_outputs_kernel(
        mm1_ptr,
        out_hidden0_ptr,
        out_hidden1_ptr,
        out_seq_ptr,
        out_two_ptr,
        out_vocab_ptr,
        TOTAL_VOCAB: tl.constexpr,
        TOTAL_SEQ: tl.constexpr,
        TOTAL_TWO: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)

        vocab_mask = offsets < TOTAL_VOCAB
        vocab_values = tl.load(mm1_ptr + offsets, mask=vocab_mask, other=0.0)
        tl.store(out_vocab_ptr + offsets, vocab_values, mask=vocab_mask)

        seq_mask = offsets < TOTAL_SEQ
        zeros = tl.zeros((BLOCK,), tl.float32)
        tl.store(out_seq_ptr + offsets, zeros, mask=seq_mask)

        two_mask = offsets < TOTAL_TWO
        tl.store(out_two_ptr + offsets, zeros, mask=two_mask)

        hidden_mask = offsets < HIDDEN_
        tl.store(out_hidden0_ptr + offsets, zeros, mask=hidden_mask)
        tl.store(out_hidden1_ptr + offsets, zeros, mask=hidden_mask)

    @triton.jit
    def oracle_kernel(
        mm142_ptr,
        mul345_ptr,
        mm144_ptr,
        mm146_ptr,
        keep_mask_ptr,
        weight_ptr,
        saved_x_ptr,
        row_scale_ptr,
        index_seq_ptr,
        full_ptr,
        index_two_ptr,
        token_index_ptr,
        out_hidden0_ptr,
        out_hidden1_ptr,
        out_seq_ptr,
        out_two_ptr,
        out_vocab_ptr,
        HIDDEN_: tl.constexpr,
        SEQ_: tl.constexpr,
        VOCAB_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        channels = tl.arange(0, BLOCK_C)
        channel_mask = channels < HIDDEN_
        base = row * HIDDEN_ + channels
        seq = row % SEQ_
        batch = row // SEQ_

        residual = (
            tl.load(mul345_ptr + base, mask=channel_mask, other=0.0).to(tl.float32)
            + tl.load(mm142_ptr + base, mask=channel_mask, other=0.0).to(tl.float32)
            + tl.load(mm144_ptr + base, mask=channel_mask, other=0.0).to(tl.float32)
            + tl.load(mm146_ptr + base, mask=channel_mask, other=0.0).to(tl.float32)
        )
        keep = tl.load(keep_mask_ptr + base, mask=channel_mask, other=0).to(tl.float32)
        dropout_scaled = residual * keep * 1.1111111111111112

        weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
        saved_x = tl.load(saved_x_ptr + base, mask=channel_mask, other=0.0).to(tl.float32)
        weighted = dropout_scaled * weight
        weighted_saved = weighted * saved_x
        row_sum = tl.sum(tl.where(channel_mask, weighted, 0.0), axis=0)
        row_weighted_sum = tl.sum(tl.where(channel_mask, weighted_saved, 0.0), axis=0)
        scale = tl.load(row_scale_ptr + row).to(tl.float32)
        grad = scale * (weighted * 768.0 - row_sum - saved_x * row_weighted_sum)

        tl.atomic_add(
            out_hidden0_ptr + channels,
            dropout_scaled * saved_x,
            sem="relaxed",
            mask=channel_mask,
        )
        tl.atomic_add(
            out_hidden1_ptr + channels,
            dropout_scaled,
            sem="relaxed",
            mask=channel_mask,
        )

        full_value = tl.load(full_ptr).to(tl.float32)

        seq_index = tl.load(index_seq_ptr + seq).to(tl.int64)
        seq_dest = tl.where(seq_index < 0, seq_index + SEQ_, seq_index)
        seq_value = tl.where(seq_index == -1, full_value, grad)
        seq_emit = (seq_index != -1) | (batch == 0)
        tl.atomic_add(
            out_seq_ptr + seq_dest * HIDDEN_ + channels,
            seq_value,
            sem="relaxed",
            mask=channel_mask & seq_emit & (seq_dest >= 0) & (seq_dest < SEQ_),
        )

        two_index = tl.load(index_two_ptr + seq).to(tl.int64)
        two_dest = tl.where(two_index < 0, two_index + 2, two_index)
        two_value = tl.where(two_index == -1, full_value, grad)
        tl.atomic_add(
            out_two_ptr + two_dest * HIDDEN_ + channels,
            two_value,
            sem="relaxed",
            mask=channel_mask & (two_dest >= 0) & (two_dest < 2),
        )

        token = tl.load(token_index_ptr + row).to(tl.int64)
        token_dest = tl.where(token < 0, token + VOCAB_, token)
        token_value = tl.where(token == 0, full_value, grad)
        tl.atomic_add(
            out_vocab_ptr + token_dest * HIDDEN_ + channels,
            token_value,
            sem="relaxed",
            mask=channel_mask & (token_dest >= 0) & (token_dest < VOCAB_),
        )


def _validate_inputs(inputs):
    (
        mm_1,
        mm_142,
        mul_345,
        mm_144,
        mm_146,
        arg105_1,
        arg3_1,
        arg104_1,
        arg323_1,
        arg1_1,
        full_1,
        arg103_1,
        arg0_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    expected_shapes = [
        (mm_1, (MM1_ROWS, HIDDEN), "mm_1"),
        (mm_142, (ROWS, HIDDEN), "mm_142"),
        (mul_345, (BATCH, SEQ, HIDDEN), "mul_345"),
        (mm_144, (ROWS, HIDDEN), "mm_144"),
        (mm_146, (ROWS, HIDDEN), "mm_146"),
        (arg105_1, (BATCH, SEQ, HIDDEN), "arg105_1"),
        (arg3_1, (HIDDEN,), "arg3_1"),
        (arg104_1, (BATCH, SEQ, HIDDEN), "arg104_1"),
        (arg323_1, (BATCH, SEQ, 1), "arg323_1"),
        (arg1_1, (1, SEQ), "arg1_1"),
        (full_1, (), "full_1"),
        (arg103_1, (1, SEQ), "arg103_1"),
        (arg0_1, (BATCH, SEQ), "arg0_1"),
    ]
    for tensor, shape, name in expected_shapes:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"unexpected {name} shape: got={tuple(tensor.shape)} expected={shape}")
    for shape_param in (_shape_param_0, _shape_param_1, _shape_param_2):
        if tuple(shape_param) != (BATCH, SEQ, HIDDEN):
            raise ValueError(f"unexpected hidden shape parameter: {shape_param}")
    if tuple(_shape_param_3) != (BATCH, SEQ):
        raise ValueError(f"unexpected index shape parameter: {_shape_param_3}")


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
        raise RuntimeError("triton is not available")
    (
        mm_1,
        mm_142,
        mul_345,
        mm_144,
        mm_146,
        arg105_1,
        arg3_1,
        arg104_1,
        arg323_1,
        arg1_1,
        full_1,
        arg103_1,
        arg0_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    if mm_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    _validate_inputs(inputs)

    out_hidden0 = torch.empty((HIDDEN,), device=mm_1.device, dtype=torch.float32)
    out_hidden1 = torch.empty((HIDDEN,), device=mm_1.device, dtype=torch.float32)
    out_seq = torch.empty((SEQ, HIDDEN), device=mm_1.device, dtype=torch.float32)
    out_two = torch.empty((2, HIDDEN), device=mm_1.device, dtype=torch.float32)
    out_vocab = torch.empty((VOCAB, HIDDEN), device=mm_1.device, dtype=torch.float32)

    total_vocab = VOCAB * HIDDEN
    total_seq = SEQ * HIDDEN
    total_two = 2 * HIDDEN
    _init_outputs_kernel[(triton.cdiv(total_vocab, INIT_BLOCK),)](
        mm_1,
        out_hidden0,
        out_hidden1,
        out_seq,
        out_two,
        out_vocab,
        TOTAL_VOCAB=total_vocab,
        TOTAL_SEQ=total_seq,
        TOTAL_TWO=total_two,
        HIDDEN_=HIDDEN,
        BLOCK=INIT_BLOCK,
        num_warps=8,
    )
    oracle_kernel[(ROWS,)](
        mm_142,
        mul_345,
        mm_144,
        mm_146,
        arg105_1,
        arg3_1,
        arg104_1,
        arg323_1,
        arg1_1,
        full_1,
        arg103_1,
        arg0_1,
        out_hidden0,
        out_hidden1,
        out_seq,
        out_two,
        out_vocab,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        VOCAB_=VOCAB,
        BLOCK_C=triton.next_power_of_2(HIDDEN),
        num_warps=8,
    )
    return (out_hidden0, out_hidden1, out_seq, out_two, out_vocab)


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
