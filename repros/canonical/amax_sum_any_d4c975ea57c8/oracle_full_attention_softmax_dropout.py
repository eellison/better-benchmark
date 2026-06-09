"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT additive-bias attention softmax/dropout return from Repro.forward, including the [1024,128,128] BMM view to [256,4,128,128], broadcast [256,1,128,128] bias add, stable last-dimension softmax, all -inf row fallback to full_3, exact Inductor RNG dropout scale, and final [1024,128,128] transpose view, whereas Inductor currently lowers the decomposed view/add/amax/sub/exp/sum/div/eq/any/where/inductor_random/dropout/expand/view/permute graph as generic reductions, pointwise RNG/dropout, and layout work over materialized intermediates; Inductor cannot do this today because its pattern matcher and scheduler do not canonicalize additive-bias attention softmax with a row-all-masked guard, stochastic dropout, and trailing layout-only transpose into one persistent online-softmax template; the fix is NEW_PATTERN: add an Inductor lowering for additive-bias attention softmax/dropout that preserves the all-masked-row guard and fuses dropout plus the output-layout epilogue into the row softmax kernel."""
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



REPRO_ID = "amax_sum_any_d4c975ea57c8"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 256
N_HEADS = 4
Q_LEN = 128
K_LEN = 128
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 23


@triton.jit
def _bias_softmax_any_dropout_kernel(
    bmm_ptr,
    where_ptr,
    full3_ptr,
    random_or_seed_ptr,
    out_base_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    N_TOTAL_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
    SEED_IDX: tl.constexpr,
    INLINE_RNG: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)

    row_mask = rows < N_TOTAL_ROWS
    mask = row_mask[:, None]

    batch = rows // (H * Q)
    q = rows % Q
    offsets = rows[:, None] * K + cols[None, :]
    where_offsets = (batch[:, None] * Q + q[:, None]) * K + cols[None, :]

    bmm = tl.load(bmm_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    bias = tl.load(where_ptr + where_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = bmm + bias

    row_max = tl.max(scores, axis=1)
    row_has_value = row_max != -float("inf")
    safe_max = tl.where(row_has_value, row_max, 0.0)
    numer = tl.exp(scores - safe_max[:, None])
    denom = tl.sum(numer, axis=1)
    safe_denom = tl.where(row_has_value, denom, 1.0)
    softmax = numer / safe_denom[:, None]
    softmax = tl.where(row_has_value[:, None], softmax, 0.0)

    fallback = tl.load(
        full3_ptr + offsets,
        mask=mask & (row_has_value[:, None] == 0),
        other=0.0,
    ).to(tl.float32)
    value = tl.where(row_has_value[:, None], softmax, fallback)

    if INLINE_RNG:
        seed = tl.load(random_or_seed_ptr + SEED_IDX)
        random = tl.rand(seed, offsets.to(tl.uint32))
    else:
        random = tl.load(random_or_seed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.where(random > DROPOUT_P_CONST, value * DROPOUT_SCALE_CONST, 0.0)

    tl.store(out_base_ptr + offsets, out, mask=mask)


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
    bmm_46: torch.Tensor,
    where: torch.Tensor,
    full_3: torch.Tensor,
    random_values: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    assert bmm_46.is_cuda and where.is_cuda and full_3.is_cuda
    assert random_values.is_cuda and out_base.is_cuda
    assert bmm_46.dtype == torch.float32 and bmm_46.shape == (BH, Q_LEN, K_LEN)
    assert where.dtype == torch.float32 and where.shape == (BATCH, 1, Q_LEN, K_LEN)
    assert full_3.dtype == torch.float32 and full_3.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert random_values.dtype == torch.float32
    assert random_values.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert out_base.dtype == torch.float32 and out_base.shape == bmm_46.shape
    assert bmm_46.is_contiguous()
    assert where.is_contiguous()
    assert full_3.is_contiguous()
    assert random_values.is_contiguous()
    assert out_base.is_contiguous()
    assert block_m > 0 and (block_m & (block_m - 1)) == 0
    assert block_k == K_LEN

    grid = (triton.cdiv(N_ROWS, block_m),)
    _bias_softmax_any_dropout_kernel[grid](
        bmm_46,
        where,
        full_3,
        random_values,
        out_base,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        N_TOTAL_ROWS=N_ROWS,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        SEED_IDX=SEED_INDEX,
        INLINE_RNG=False,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def _launch_oracle_inline_rng(
    bmm_46: torch.Tensor,
    where: torch.Tensor,
    full_3: torch.Tensor,
    inductor_seeds: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    assert bmm_46.is_cuda and where.is_cuda and full_3.is_cuda
    assert inductor_seeds.is_cuda and out_base.is_cuda
    assert bmm_46.dtype == torch.float32 and bmm_46.shape == (BH, Q_LEN, K_LEN)
    assert where.dtype == torch.float32 and where.shape == (BATCH, 1, Q_LEN, K_LEN)
    assert full_3.dtype == torch.float32 and full_3.shape == (BATCH, N_HEADS, Q_LEN, K_LEN)
    assert inductor_seeds.dtype == torch.int64 and inductor_seeds.numel() > SEED_INDEX
    assert out_base.dtype == torch.float32 and out_base.shape == bmm_46.shape
    assert bmm_46.is_contiguous()
    assert where.is_contiguous()
    assert full_3.is_contiguous()
    assert out_base.is_contiguous()
    assert block_m > 0 and (block_m & (block_m - 1)) == 0
    assert block_k == K_LEN

    grid = (triton.cdiv(N_ROWS, block_m),)
    _bias_softmax_any_dropout_kernel[grid](
        bmm_46,
        where,
        full_3,
        inductor_seeds,
        out_base,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        N_TOTAL_ROWS=N_ROWS,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        SEED_IDX=SEED_INDEX,
        INLINE_RNG=True,
        num_warps=num_warps,
    )
    return out_base.permute(0, 2, 1)


def oracle_full_attention_softmax_dropout(
    bmm_46: torch.Tensor,
    where: torch.Tensor,
    full_3: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    out_base: torch.Tensor | None = None,
    random_values: torch.Tensor | None = None,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    _check_shape_params(_shape_param_0, _shape_param_1, _shape_param_2)

    if random_values is None:
        random_values = _inductor_random_like_repro(inductor_seeds)
    if out_base is None:
        out_base = torch.empty_like(bmm_46)
    return _launch_oracle(
        bmm_46,
        where,
        full_3,
        random_values,
        out_base,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([1024, 128, 128], f32), T([256, 1, 128, 128], f32), T([256, 4, 128, 128], f32), T([24], i64), S([256, 4, 128, 128]), S([256, 4, 128, 128]), S([1024, 128, 128]))")
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
