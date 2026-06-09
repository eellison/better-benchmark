"""
Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full GPT-Neo attention-mask softmax materialization with one Triton program per `[128]` row, recomputing the causal/segment predicate from `unsqueeze` and `cumsum`, loading the sliced `arg8_1[:, :, :128, :128]` attention mask directly, applying the same two `where` masks to `bmm.view(32, 16, 128, 128)`, and normalizing the row without materializing either mask tensor or the masked-score intermediate. Inductor lowers the decomposed graph as generic mask construction, two large `where` operations, an add, and a softmax reduction over the materialized `[32, 16, 128, 128]` score tensor; it cannot currently see this as one masked softmax row template because the predicate is built from advanced indexing into `cumsum` plus a separate broadcasted attention mask slice. The fix is a NEW_PATTERN lowering for attention masked softmax that accepts structured mask predicates and emits a row kernel that recomputes cheap boolean masks at the point of use.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

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



REPRO_ID = "amax_sum_ca987107ccc7"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 16
SEQ = 128
ATTN_MASK_STRIDE = 2048
N_ROWS = BATCH * HEADS * SEQ

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _masked_softmax_rows_kernel(
    unsqueeze_ptr,
    cumsum_ptr,
    bmm_ptr,
    arg8_ptr,
    out_ptr,
    block_n: tl.constexpr,
    seq_len: tl.constexpr,
    n_heads: tl.constexpr,
    attn_mask_stride: tl.constexpr,
):
    row = tl.program_id(0)
    q = row % seq_len
    batch_head = row // seq_len
    batch = batch_head // n_heads

    cols = tl.arange(0, block_n)
    col_mask = cols < seq_len

    q_index = tl.load(unsqueeze_ptr + q)
    k_index = tl.load(unsqueeze_ptr + cols, mask=col_mask, other=0)

    causal_mask = k_index <= q_index
    q_segment = tl.load(cumsum_ptr + batch * seq_len + q_index)
    k_segment = tl.load(
        cumsum_ptr + batch * seq_len + k_index,
        mask=col_mask,
        other=q_segment + 1,
    )
    segment_mask = causal_mask & (k_segment == q_segment)

    score_offsets = row * seq_len + cols
    scores = tl.load(bmm_ptr + score_offsets, mask=col_mask, other=0.0).to(tl.float32)

    attn_mask = tl.load(
        arg8_ptr + q * attn_mask_stride + cols,
        mask=col_mask,
        other=0,
    ).to(tl.int1)

    min_f32 = tl.full((block_n,), -3.4028234663852886e38, tl.float32)
    zeros = tl.zeros((block_n,), tl.float32)
    masked_scores = tl.where(attn_mask, scores, min_f32) + tl.where(
        segment_mask,
        zeros,
        min_f32,
    )

    row_max = tl.max(masked_scores, axis=0)
    numer = tl.exp(masked_scores - row_max)
    denom = tl.sum(numer, axis=0)
    out = numer / denom

    tl.store(out_ptr + score_offsets, out, mask=col_mask)


def _launch_oracle(
    unsqueeze: torch.Tensor,
    cumsum: torch.Tensor,
    bmm: torch.Tensor,
    arg8_1: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert unsqueeze.is_cuda and cumsum.is_cuda and bmm.is_cuda and arg8_1.is_cuda
    assert unsqueeze.dtype == torch.int64 and unsqueeze.shape == (1, SEQ)
    assert cumsum.dtype == torch.int64 and cumsum.shape == (BATCH, SEQ)
    assert bmm.dtype == torch.float32 and bmm.shape == (BATCH * HEADS, SEQ, SEQ)
    assert arg8_1.dtype == torch.bool and arg8_1.shape == (1, 1, 2048, 2048)
    assert out.dtype == torch.float32 and out.shape == bmm.shape

    _masked_softmax_rows_kernel[(N_ROWS,)](
        unsqueeze,
        cumsum,
        bmm,
        arg8_1,
        out,
        block_n=block_n,
        seq_len=SEQ,
        n_heads=HEADS,
        attn_mask_stride=ATTN_MASK_STRIDE,
        num_warps=num_warps,
    )
    return out


def oracle_attention_masked_softmax(
    unsqueeze: torch.Tensor,
    cumsum: torch.Tensor,
    bmm: torch.Tensor,
    arg8_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    _shape_param_3=None,
    *,
    block_n: int = 128,
    num_warps: int = 4,
) -> torch.Tensor:
    out = torch.empty_strided(
        bmm.shape,
        bmm.stride(),
        device=bmm.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        unsqueeze,
        cumsum,
        bmm,
        arg8_1,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([1, 128], i64, gen=Index(128)), T([32, 128], i64, gen=Index(32)), T([512, 128, 128], f32), T([1, 1, 2048, 2048], b8), S([32, -1, 128, 128]), S([32, 16, 128, 128]), S([32, 16, 128, 128]), S([512, 128, 128]))")
def oracle_forward(inputs):
    return oracle_attention_masked_softmax(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
