"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MT5 dual attention softmax/dropout return from Repro.forward by recomputing both relative-position bucket tables and the causal/noncausal structured masks inside Triton row kernels, adding the viewed BMM scores, performing each last-dimension softmax and dropout with exact Inductor random values, and returning both required [192, 128, 128] transposed views, whereas Inductor currently lowers the decomposed iota/bucket/embedding/where/view/add/amax/sub/exp/sum/div/random/dropout/permute graph as generic mask and relative-bias producers feeding separate softmax, RNG, and layout kernels; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize T5/MT5 relative-position attention with two sibling stochastic softmax epilogues and trailing transpose views into persistent row-softmax templates that recompute cheap structured bias/mask predicates at point of use; the fix is NEW_PATTERN: add an Inductor lowering for T5-style relative-position attention softmax/dropout that fuses bucket lookup, structured masks, dropout, and output-layout epilogues into the row kernels."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
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



REPRO_ID = "amax_sum_amax_9faf4a02126a"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
N_HEADS = 6
Q_LEN = 128
K_LEN = 128
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
ENCODER_SEED_INDEX = 1
DECODER_SEED_INDEX = 35

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
def _mt5_relative_softmax_dropout_kernel(
    bmm_ptr,
    rel_bias_ptr,
    random_ptr,
    out_base_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    IS_BIDIRECTIONAL_BUCKET: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    row = tl.program_id(0)
    bh = row // Q
    head = bh - (bh // H) * H
    q = row - bh * Q

    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    rel = cols.to(tl.int32) - q

    if IS_BIDIRECTIONAL_BUCKET:
        distance = tl.where(rel < 0, -rel, rel)
        bucket = distance
        bucket = tl.where(distance >= 8, 8, bucket)
        bucket = tl.where(distance >= 12, 9, bucket)
        bucket = tl.where(distance >= 16, 10, bucket)
        bucket = tl.where(distance >= 23, 11, bucket)
        bucket = tl.where(distance >= 32, 12, bucket)
        bucket = tl.where(distance >= 46, 13, bucket)
        bucket = tl.where(distance >= 64, 14, bucket)
        bucket = tl.where(distance >= 91, 15, bucket)
        bucket += tl.where(rel > 0, 16, 0)
    else:
        distance = tl.where(rel < 0, -rel, 0)
        bucket = distance
        bucket = tl.where(distance >= 16, 16, bucket)
        bucket = tl.where(distance >= 19, 17, bucket)
        bucket = tl.where(distance >= 21, 18, bucket)
        bucket = tl.where(distance >= 24, 19, bucket)
        bucket = tl.where(distance >= 27, 20, bucket)
        bucket = tl.where(distance >= 31, 21, bucket)
        bucket = tl.where(distance >= 35, 22, bucket)
        bucket = tl.where(distance >= 40, 23, bucket)
        bucket = tl.where(distance >= 46, 24, bucket)
        bucket = tl.where(distance >= 52, 25, bucket)
        bucket = tl.where(distance >= 59, 26, bucket)
        bucket = tl.where(distance >= 67, 27, bucket)
        bucket = tl.where(distance >= 77, 28, bucket)
        bucket = tl.where(distance >= 87, 29, bucket)
        bucket = tl.where(distance >= 99, 30, bucket)
        bucket = tl.where(distance >= 113, 31, bucket)

    row_offsets = row * K + cols
    bias = tl.load(
        rel_bias_ptr + bucket * H + head,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    bmm = tl.load(
        bmm_ptr + row_offsets,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    scores = bmm + bias
    if IS_CAUSAL:
        scores = tl.where(cols <= q, scores, -3.4028234663852886e38)
    scores = tl.where(col_mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    softmax = numer / denom

    random = tl.load(random_ptr + row_offsets, mask=col_mask, other=0.0).to(tl.float32)
    keep = random > DROPOUT_P_CONST
    out = tl.where(keep, softmax * DROPOUT_SCALE_CONST, 0.0)

    tl.store(out_base_ptr + row_offsets, out, mask=col_mask)


def _inductor_random_like_repro(
    inductor_seeds: torch.Tensor,
    seed_index: int,
) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, seed_index)
    return torch.ops.prims.inductor_random.default(
        [BATCH, N_HEADS, Q_LEN, K_LEN],
        seed,
        "rand",
    )


def _check_shape_params(
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
) -> None:
    assert list(_shape_param_0) == [BATCH, -1, Q_LEN, K_LEN]
    assert list(_shape_param_1) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_2) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_3) == [BH, Q_LEN, K_LEN]
    assert list(_shape_param_4) == [BATCH, -1, Q_LEN, K_LEN]
    assert list(_shape_param_5) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_6) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_7) == [BH, Q_LEN, K_LEN]


def _check_kernel_inputs(
    bmm: torch.Tensor,
    rel_bias: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
) -> None:
    assert bmm.is_cuda and rel_bias.is_cuda and random_values.is_cuda and out_base.is_cuda
    assert bmm.dtype == torch.float32 and bmm.shape == (BH, Q_LEN, K_LEN)
    assert rel_bias.dtype == torch.float32 and rel_bias.shape == (32, N_HEADS)
    assert random_values.dtype == torch.float32
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert out_base.dtype == torch.float32 and out_base.shape == (BH, Q_LEN, K_LEN)
    assert bmm.is_contiguous()
    assert rel_bias.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()


def _launch_attention_branch(
    bmm: torch.Tensor,
    rel_bias: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    *,
    is_causal: bool,
    is_bidirectional_bucket: bool,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    _check_kernel_inputs(bmm, rel_bias, random_values, out_base)
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    _mt5_relative_softmax_dropout_kernel[(N_ROWS,)](
        bmm,
        rel_bias,
        random_values,
        out_base,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        BLOCK_K=block_k,
        IS_CAUSAL=is_causal,
        IS_BIDIRECTIONAL_BUCKET=is_bidirectional_bucket,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_mt5_attention_softmax_dropout(
    bmm: torch.Tensor,
    arg6_1: torch.Tensor,
    inductor_seeds: torch.Tensor,
    bmm_16: torch.Tensor,
    arg81_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
    _shape_param_7,
    *,
    encoder_random: torch.Tensor | None = None,
    decoder_random: torch.Tensor | None = None,
    encoder_out_base: torch.Tensor | None = None,
    decoder_out_base: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_shape_params(
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
        _shape_param_7,
    )

    bmm_view = bmm.view(BH, Q_LEN, K_LEN)
    bmm_16_view = bmm_16.view(BH, Q_LEN, K_LEN)

    if encoder_random is None:
        encoder_random = _inductor_random_like_repro(inductor_seeds, ENCODER_SEED_INDEX)
    if decoder_random is None:
        decoder_random = _inductor_random_like_repro(inductor_seeds, DECODER_SEED_INDEX)
    if encoder_out_base is None:
        encoder_out_base = torch.empty_like(bmm_view)
    if decoder_out_base is None:
        decoder_out_base = torch.empty_like(bmm_16_view)

    encoder_out = _launch_attention_branch(
        bmm_view,
        arg6_1,
        encoder_random,
        encoder_out_base,
        is_causal=False,
        is_bidirectional_bucket=True,
        block_k=block_k,
        num_warps=num_warps,
    )
    decoder_out = _launch_attention_branch(
        bmm_16_view,
        arg81_1,
        decoder_random,
        decoder_out_base,
        is_causal=True,
        is_bidirectional_bucket=False,
        block_k=block_k,
        num_warps=num_warps,
    )
    return decoder_out, encoder_out


@oracle_impl(hardware="H100", shapes="(T([192, 128, 128], f32), T([32, 6], f32), T([84], i64), T([192, 128, 128], f32), T([32, 6], f32), S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]), S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]))")
def oracle_forward(inputs):
    return oracle_full_mt5_attention_softmax_dropout(*inputs)


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
