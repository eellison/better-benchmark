"""
Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo infer masked-attention softmax returned by Repro.forward in one Triton row kernel, including the sliced attention mask, the generated causal arange predicate, the `cumsum` segment-equality predicate, the score masking, stable row softmax, and final `[512,128,128]` output layout, whereas Inductor currently lowers the decomposed slice/iota/index/where/add/amax/sub/exp/sum/div graph as generic mask construction plus a softmax over materialized masked scores; Inductor cannot do this today because its pattern library does not recognize an attention softmax whose predicate is assembled from generated aranges plus advanced indexing into `cumsum` and a separate broadcasted bool attention mask; the fix is NEW_PATTERN: add a guarded masked-attention-softmax lowering that recomputes cheap structured predicates inside the online row-softmax kernel and avoids materializing the mask and masked-score intermediates.
"""
from __future__ import annotations

import argparse
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 16
SEQ = 128
ATTN_MASK_STRIDE = 2048
N_ROWS = BATCH * HEADS * SEQ


def get_inputs():
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _gptneo_masked_softmax_kernel(
    arg8_ptr,
    bmm_ptr,
    cumsum_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < TOTAL_ROWS
    col_mask = cols < SEQ_LEN
    mask = row_mask[:, None] & col_mask[None, :]

    q = rows % SEQ_LEN
    batch_head = rows // SEQ_LEN
    batch = batch_head // N_HEADS

    q_segment = tl.load(cumsum_ptr + batch * SEQ_LEN + q, mask=row_mask, other=0)
    k_segment = tl.load(
        cumsum_ptr + batch[:, None] * SEQ_LEN + cols[None, :],
        mask=mask,
        other=q_segment[:, None] + 1,
    )
    structured_mask = (cols[None, :] <= q[:, None]) & (k_segment == q_segment[:, None])

    scores = tl.load(
        bmm_ptr + rows[:, None] * SEQ_LEN + cols[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    attn_mask = tl.load(
        arg8_ptr + q[:, None] * MASK_STRIDE + cols[None, :],
        mask=mask,
        other=0,
    ).to(tl.int1)

    min_f32 = tl.full((BLOCK_N,), -3.4028234663852886e38, tl.float32)
    masked_scores = tl.where(attn_mask, scores, min_f32)
    masked_scores += tl.where(structured_mask, 0.0, min_f32)

    row_max = tl.max(masked_scores, axis=1)
    numer = tl.exp(masked_scores - row_max[:, None])
    denom = tl.sum(numer, axis=1)
    out = numer / denom[:, None]

    tl.store(out_ptr + rows[:, None] * SEQ_LEN + cols[None, :], out, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([1, 1, 2048, 2048], b8), T([512, 128, 128], f32), T([32, 128], i64, gen=Index(32)), S([32, 16, 128, 128]), S([32, -1, 128, 128]), S([32, 16, 128, 128]), S([512, 128, 128]))")
def oracle_forward(inputs):
    arg8_1, bmm, cumsum, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    out = torch.empty_strided(
        (BATCH * HEADS, SEQ, SEQ),
        (SEQ * SEQ, SEQ, 1),
        device=bmm.device,
        dtype=torch.float32,
    )
    block_m = 8
    _gptneo_masked_softmax_kernel[(triton.cdiv(N_ROWS, block_m),)](
        arg8_1,
        bmm,
        cumsum,
        out,
        BLOCK_M=block_m,
        BLOCK_N=SEQ,
        TOTAL_ROWS=N_ROWS,
        SEQ_LEN=SEQ,
        N_HEADS=HEADS,
        MASK_STRIDE=ATTN_MASK_STRIDE,
        num_warps=4,
    )
    return out


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
