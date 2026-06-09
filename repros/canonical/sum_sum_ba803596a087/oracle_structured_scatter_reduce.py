"""
Oracle for sum_sum_ba803596a087

Gap diagnosis:
  Classification: BANDWIDTH_BOUND
  What oracle does differently: Computes the full masked-LM gradient add, padded side output, and vocabulary reduction in Triton while algebraically removing the dense one-hot/iota materialization.
  What Inductor change would fix: Treat this as bandwidth-bound unless a scheduler fusion can share the large side-output producer with the column reduction without losing memory throughput.
"""
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

# Import shared oracle infrastructure (installed via pip install -e .)
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

BLOCK_N = 64
BLOCK_V = 128
REDUCE_BLOCKS = 32
STORE_WARPS = 4
REDUCE_WARPS = 4
FINAL_WARPS = 1

if triton is not None:

    @triton.jit
    def _store_and_partial_sum_kernel(
        arg1318_ptr,
        arg1122_ptr,
        labels_ptr,
        arg1123_ptr,
        arg1319_ptr,
        side_out_ptr,
        partials_ptr,
        N_ROWS: tl.constexpr,
        VOCAB: tl.constexpr,
        PADDED_VOCAB: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_V_: tl.constexpr,
    ):
        v_block = tl.program_id(0)
        n_block = tl.program_id(1)

        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        v_offsets = v_block * BLOCK_V_ + tl.arange(0, BLOCK_V_)

        row_mask = n_offsets < N_ROWS
        vocab_mask = v_offsets < VOCAB
        side_mask = v_offsets < PADDED_VOCAB

        labels = tl.load(labels_ptr + n_offsets, mask=row_mask, other=-100)
        valid_label = (labels != -100) & (labels >= 0) & (labels < VOCAB)
        scale = tl.load(arg1318_ptr) / tl.load(arg1122_ptr)

        in_offsets = n_offsets[:, None] * VOCAB + v_offsets[None, :]
        in_mask = row_mask[:, None] & vocab_mask[None, :]
        grad = tl.load(arg1123_ptr + in_offsets, mask=in_mask, other=0.0).to(tl.float32)
        residual = tl.load(arg1319_ptr + in_offsets, mask=in_mask, other=0.0).to(tl.float32)

        label_hit = valid_label[:, None] & (v_offsets[None, :] == labels[:, None])
        ce_grad = tl.where(valid_label[:, None], scale * grad, 0.0)
        ce_grad = ce_grad - tl.where(label_hit, scale, 0.0)
        value = residual + ce_grad

        side_offsets = n_offsets[:, None] * PADDED_VOCAB + v_offsets[None, :]
        side_value = tl.where(vocab_mask[None, :], value, 0.0)
        tl.store(side_out_ptr + side_offsets, side_value, mask=row_mask[:, None] & side_mask[None, :])

        partial = tl.sum(tl.where(in_mask, value, 0.0), axis=0)
        tl.store(
            partials_ptr + n_block * VOCAB + v_offsets,
            partial,
            mask=vocab_mask,
        )


    @triton.jit
    def _reduce_partials_stage1_kernel(
        partials_ptr,
        partials2_ptr,
        NUM_N_BLOCKS: tl.constexpr,
        VOCAB: tl.constexpr,
        BLOCK_R_: tl.constexpr,
        BLOCK_V_: tl.constexpr,
    ):
        v_block = tl.program_id(0)
        r_block = tl.program_id(1)

        r_offsets = r_block * BLOCK_R_ + tl.arange(0, BLOCK_R_)
        v_offsets = v_block * BLOCK_V_ + tl.arange(0, BLOCK_V_)
        mask = (r_offsets[:, None] < NUM_N_BLOCKS) & (v_offsets[None, :] < VOCAB)
        vals = tl.load(
            partials_ptr + r_offsets[:, None] * VOCAB + v_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        partial = tl.sum(vals, axis=0)
        tl.store(
            partials2_ptr + r_block * VOCAB + v_offsets,
            partial,
            mask=v_offsets < VOCAB,
        )


    @triton.jit
    def _reduce_partials_stage2_kernel(
        partials2_ptr,
        out_ptr,
        NUM_R_BLOCKS: tl.constexpr,
        VOCAB: tl.constexpr,
        BLOCK_R2_: tl.constexpr,
        BLOCK_V_: tl.constexpr,
    ):
        v_block = tl.program_id(0)

        r_offsets = tl.arange(0, BLOCK_R2_)
        v_offsets = v_block * BLOCK_V_ + tl.arange(0, BLOCK_V_)
        mask = (r_offsets[:, None] < NUM_R_BLOCKS) & (v_offsets[None, :] < VOCAB)
        vals = tl.load(
            partials2_ptr + r_offsets[:, None] * VOCAB + v_offsets[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        total = tl.sum(vals, axis=0)
        tl.store(out_ptr + v_offsets, total, mask=v_offsets < VOCAB)


@oracle_impl(hardware="H100", shapes="(T([], f32), T([], f32), T([256, 128], i64), T([32768, 30522], f32), T([256, 128, 30522], f32), S([1, 30522]), S([32768, 30522]), S([256, 128, 30522]), S([30522]), S([32768, 30522]))")
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
        arg1318_1,
        arg1122_1,
        arg582_1,
        arg1123_1,
        arg1319_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
    ) = inputs

    n_rows = arg1123_1.shape[0]
    vocab = arg1123_1.shape[1]
    padded_vocab = vocab + 2

    out0 = torch.empty((vocab,), device=arg1123_1.device, dtype=arg1123_1.dtype)
    out1 = torch.empty((n_rows, padded_vocab), device=arg1123_1.device, dtype=arg1123_1.dtype)

    num_n_blocks = triton.cdiv(n_rows, BLOCK_N)
    num_v_blocks = triton.cdiv(padded_vocab, BLOCK_V)
    partials = torch.empty((num_n_blocks, vocab), device=arg1123_1.device, dtype=torch.float32)

    _store_and_partial_sum_kernel[(num_v_blocks, num_n_blocks)](
        arg1318_1,
        arg1122_1,
        arg582_1,
        arg1123_1,
        arg1319_1,
        out1,
        partials,
        N_ROWS=n_rows,
        VOCAB=vocab,
        PADDED_VOCAB=padded_vocab,
        BLOCK_N_=BLOCK_N,
        BLOCK_V_=BLOCK_V,
        num_warps=STORE_WARPS,
    )

    num_r_blocks = triton.cdiv(num_n_blocks, REDUCE_BLOCKS)
    partials2 = torch.empty((num_r_blocks, vocab), device=arg1123_1.device, dtype=torch.float32)
    _reduce_partials_stage1_kernel[(triton.cdiv(vocab, BLOCK_V), num_r_blocks)](
        partials,
        partials2,
        NUM_N_BLOCKS=num_n_blocks,
        VOCAB=vocab,
        BLOCK_R_=REDUCE_BLOCKS,
        BLOCK_V_=BLOCK_V,
        num_warps=REDUCE_WARPS,
    )
    _reduce_partials_stage2_kernel[(triton.cdiv(vocab, BLOCK_V),)](
        partials2,
        out0,
        NUM_R_BLOCKS=num_r_blocks,
        VOCAB=vocab,
        BLOCK_R2_=triton.next_power_of_2(num_r_blocks),
        BLOCK_V_=BLOCK_V,
        num_warps=FINAL_WARPS,
    )

    return (out0, out1)


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
