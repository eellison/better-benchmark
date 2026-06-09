"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete M2M100 additive-mask attention softmax/dropout return from Repro.forward, including the bool mask/scalar where bias, the [1024,128,128] to [64,16,128,128] view, stable last-dimension softmax, all-minus-inf row zero guard, exact Inductor RNG dropout and scale, expand/view, and final transposed [1024,128,128] output view, whereas Inductor currently lowers the decomposed where/view/add/amax/sub/exp/sum/div/eq/any/where/inductor_random/dropout/expand/view/permute graph as generic pointwise, reduction, RNG, and layout kernels over materialized intermediates; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize scalar-selected additive attention masks with row-all-masked guards, stochastic dropout, and trailing layout-only transposes into one persistent online-softmax template; the fix is NEW_PATTERN: add an Inductor attention softmax/dropout lowering that folds the mask scalar bias, preserves the all-masked-row guard, and fuses dropout plus the output-layout epilogue into the row softmax kernel."""
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



REPRO_ID = "amax_sum_any_d16c257aa03e"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 64
N_HEADS = 16
Q_LEN = 128
K_LEN = 128
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 1


@triton.jit
def _masked_bias_softmax_any_dropout_kernel(
    mask_ptr,
    true_bias_ptr,
    false_bias_ptr,
    bmm_ptr,
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
    mask_offsets = (batch * Q + q) * K + cols

    bmm = tl.load(bmm_ptr + row_offsets, mask=col_mask, other=-float("inf")).to(tl.float32)
    raw_mask = tl.load(mask_ptr + mask_offsets, mask=col_mask, other=0).to(tl.int1)
    true_bias = tl.load(true_bias_ptr).to(tl.float32)
    false_bias = tl.load(false_bias_ptr).to(tl.float32)
    bias = tl.where(raw_mask, true_bias, false_bias)
    scores = bmm + bias

    value_mask = col_mask & (scores != -float("inf"))
    row_has_value = tl.sum(value_mask.to(tl.int32), axis=0) != 0

    row_max = tl.max(scores, axis=0)
    safe_max = tl.where(row_has_value, row_max, 0.0)
    numer = tl.exp(scores - safe_max)
    denom = tl.sum(numer, axis=0)
    safe_denom = tl.where(row_has_value, denom, 1.0)
    softmax = numer / safe_denom
    value = tl.where(row_has_value, softmax, 0.0)

    random = tl.load(random_ptr + row_offsets, mask=col_mask, other=0.0).to(tl.float32)
    keep = (random > DROPOUT_P_CONST).to(tl.float32)
    out = value * keep * DROPOUT_SCALE_CONST

    tl.store(out_base_ptr + row_offsets, out, mask=col_mask)


def _check_shape_params(
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
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
    mask: torch.Tensor,
    true_bias: torch.Tensor,
    false_bias: torch.Tensor,
    bmm: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    assert mask.is_cuda and true_bias.is_cuda and false_bias.is_cuda
    assert bmm.is_cuda and random_values.is_cuda and out_base.is_cuda
    assert mask.dtype == torch.bool and mask.shape == (BATCH, 1, Q_LEN, K_LEN)
    assert true_bias.shape == () and true_bias.dtype == torch.float32
    assert false_bias.shape == () and false_bias.dtype == torch.float32
    assert bmm.dtype == torch.float32 and bmm.shape == (BH, Q_LEN, K_LEN)
    assert random_values.dtype == torch.float32
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert out_base.dtype == torch.float32 and out_base.shape == bmm.shape
    assert mask.is_contiguous()
    assert bmm.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()
    assert block_k >= K_LEN and (block_k & (block_k - 1)) == 0

    _masked_bias_softmax_any_dropout_kernel[(N_ROWS,)](
        mask,
        true_bias,
        false_bias,
        bmm,
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


def oracle_full_m2m100_attention_softmax_dropout(
    mask: torch.Tensor,
    true_bias: torch.Tensor,
    false_bias: torch.Tensor,
    bmm: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    out_base: torch.Tensor | None = None,
    random_values: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    _check_shape_params(_shape_param_0, _shape_param_1, _shape_param_2)

    if random_values is None:
        random_values = _inductor_random_like_repro(inductor_seeds)
    if out_base is None:
        out_base = torch.empty_like(bmm)
    return _launch_oracle(
        mask,
        true_bias,
        false_bias,
        bmm,
        random_values,
        out_base,
        block_k=block_k,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([64, 1, 128, 128], b8), T([], f32), T([], f32), T([1024, 128, 128], f32), T([4], i64), S([64, 16, 128, 128]), S([64, 16, 128, 128]), S([1024, 128, 128]))")
def oracle_forward(inputs):
    return oracle_full_m2m100_attention_softmax_dropout(*inputs)


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
