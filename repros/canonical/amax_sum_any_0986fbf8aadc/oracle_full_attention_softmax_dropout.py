"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DistilBERT additive-bias attention softmax/dropout return from Repro.forward, including the `[96,512,512]` BMM view, broadcast `[8,1,512,512]` bias add, stable last-dimension softmax, all-`-inf` row fallback to `full_2`, exact Inductor RNG dropout scale, and final transposed `[96,512,512]` output view, whereas Inductor currently lowers the decomposed view/add/amax/sub/exp/sum/div/eq/any/where/random/dropout/view/permute graph as generic reductions, pointwise kernels, RNG, and layout work over materialized softmax intermediates; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize additive-bias attention softmax with a row-all-masked guard, stochastic dropout, and a trailing layout-only transpose into one persistent online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering for additive-bias attention softmax/dropout that preserves the all-masked-row guard and fuses dropout plus the output-layout epilogue into the row softmax kernel."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "amax_sum_any_0986fbf8aadc"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
N_HEADS = 12
Q_LEN = 512
K_LEN = 512
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 11

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
def _bias_softmax_any_dropout_kernel(
    bmm_ptr,
    where_ptr,
    full2_ptr,
    random_ptr,
    out_base_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    row = tl.program_id(0)
    bh = row // Q
    batch = bh // H
    q = row - bh * Q

    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    row_offsets = row * K + cols
    where_offsets = (batch * Q + q) * K + cols

    bmm = tl.load(bmm_ptr + row_offsets, mask=col_mask, other=-float("inf")).to(tl.float32)
    bias = tl.load(where_ptr + where_offsets, mask=col_mask, other=0.0).to(tl.float32)
    scores = bmm + bias

    value_mask = col_mask & (scores != -float("inf"))
    row_has_value = tl.sum(value_mask.to(tl.int32), axis=0) != 0

    row_max = tl.max(scores, axis=0)
    safe_max = tl.where(row_has_value, row_max, 0.0)
    numer = tl.exp(scores - safe_max)
    denom = tl.sum(numer, axis=0)
    safe_denom = tl.where(row_has_value, denom, 1.0)
    softmax = numer / safe_denom
    softmax = tl.where(row_has_value, softmax, 0.0)

    fallback = tl.load(
        full2_ptr + row_offsets,
        mask=col_mask & (row_has_value == 0),
        other=0.0,
    ).to(tl.float32)
    value = tl.where(row_has_value, softmax, fallback)

    random = tl.load(random_ptr + row_offsets, mask=col_mask, other=0.0).to(tl.float32)
    keep = (random > DROPOUT_P_CONST).to(tl.float32)
    out = value * keep * DROPOUT_SCALE_CONST

    tl.store(out_base_ptr + row_offsets, out, mask=col_mask)


def _check_shape_params(
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> None:
    assert list(_shape_param_0) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_1) == [BATCH, N_HEADS, Q_LEN, K_LEN]
    assert list(_shape_param_2) == [BH, Q_LEN, K_LEN]


def _inductor_random_like_repro(inductor_seeds: torch.Tensor) -> torch.Tensor:
    seed = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, SEED_INDEX)
    return torch.ops.prims.inductor_random.default(
        [BATCH, N_HEADS, Q_LEN, K_LEN],
        seed,
        "rand",
    )


def _launch_oracle(
    bmm_10: torch.Tensor,
    where: torch.Tensor,
    full_2: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    assert bmm_10.is_cuda and where.is_cuda and full_2.is_cuda
    assert random_values.is_cuda and out_base.is_cuda
    assert bmm_10.dtype == torch.float32 and bmm_10.shape == (BH, Q_LEN, K_LEN)
    assert where.dtype == torch.float32 and where.shape == (BATCH, 1, Q_LEN, K_LEN)
    assert full_2.dtype == torch.float32 and full_2.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert random_values.dtype == torch.float32
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert out_base.dtype == torch.float32 and out_base.shape == bmm_10.shape
    assert bmm_10.is_contiguous()
    assert where.is_contiguous()
    assert full_2.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    _bias_softmax_any_dropout_kernel[(N_ROWS,)](
        bmm_10,
        where,
        full_2,
        random_values,
        out_base,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        BLOCK_K=block_k,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_attention_softmax_dropout(
    bmm_10: torch.Tensor,
    where: torch.Tensor,
    full_2: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    *,
    out_base: torch.Tensor | None = None,
    random_values: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 8,
) -> torch.Tensor:
    _check_shape_params(_shape_param_0, _shape_param_1, _shape_param_2)

    if random_values is None:
        random_values = _inductor_random_like_repro(inductor_seeds)
    if out_base is None:
        out_base = torch.empty_like(bmm_10)
    return _launch_oracle(
        bmm_10,
        where,
        full_2,
        random_values,
        out_base,
        block_k=block_k,
        num_warps=num_warps,
    )


def oracle_forward(inputs):
    return oracle_full_attention_softmax_dropout(*inputs)


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
