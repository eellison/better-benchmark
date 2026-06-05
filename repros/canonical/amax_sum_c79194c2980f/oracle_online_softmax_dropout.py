"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5 attention softmax/dropout return from Repro.forward, including the [96,1024,1024] to [8,12,1024,1024] score view, fp32 bias add, stable last-dimension softmax, Inductor stateless RNG dropout from inductor_seeds[119], dropout scaling, same-shape expand/view, and final [96,1024,1024] permuted output layout, whereas Inductor currently lowers the decomposed add/amax/sub/exp/sum/div/inductor_random/dropout/expand/view/permute graph through generic reduction, stochastic pointwise, and layout scheduling; Inductor cannot do this today because its pattern library does not have a full-scope attention-softmax/dropout template that sinks the seeded RNG epilogue and trailing layout view into the row reduction for this K=1024 T5 shape; the fix is NEW_PATTERN: add a guarded online attention softmax/dropout/layout lowering that emits the row kernel directly and benchmark-gates it against the generic schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
Q_LEN = 1024
K_LEN = 1024
BH = BATCH * HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
ADD_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
OUT_BASE_SHAPE = BMM_SHAPE
OUT_VIEW_STRIDE = (Q_LEN * K_LEN, 1, K_LEN)
SEED_INDEX = 119
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)


@triton.jit
def _softmax_dropout_layout_kernel(
    bmm_ptr,
    add_ptr,
    seeds_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_ROWS_CONST: tl.constexpr,
    K_LEN_CONST: tl.constexpr,
    SEED_INDEX_CONST: tl.constexpr,
    DROPOUT_P_CONST: tl.constexpr,
    DROPOUT_SCALE_CONST: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < N_ROWS_CONST
    col_mask = cols < K_LEN_CONST
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * K_LEN_CONST + cols[None, :]

    bmm = tl.load(bmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(add_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scores = tl.where(mask, bmm + bias, -float("inf"))

    row_max = tl.max(scores, axis=1)
    numer = tl.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    seed = tl.load(seeds_ptr + SEED_INDEX_CONST)
    random_values = tl.rand(seed, offsets.to(tl.uint32))
    keep = (random_values > DROPOUT_P_CONST).to(tl.float32)
    out = probs * keep * DROPOUT_SCALE_CONST

    tl.store(out_ptr + offsets, out, mask=mask)


def _validate_shape_param(name: str, actual: object, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm_70: torch.Tensor,
    add_74: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
) -> None:
    if not (bmm_70.is_cuda and add_74.is_cuda and inductor_seeds.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_70.dtype != torch.float32 or add_74.dtype != torch.float32:
        raise TypeError(f"expected fp32 score inputs, got {bmm_70.dtype} and {add_74.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 inductor_seeds, got {inductor_seeds.dtype}")
    if tuple(bmm_70.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_70 shape: {tuple(bmm_70.shape)}")
    if tuple(add_74.shape) != ADD_SHAPE:
        raise ValueError(f"unexpected add_74 shape: {tuple(add_74.shape)}")
    if tuple(inductor_seeds.shape) != (124,):
        raise ValueError(f"unexpected inductor_seeds shape: {tuple(inductor_seeds.shape)}")
    if tuple(bmm_70.stride()) != (Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected bmm_70 stride: {tuple(bmm_70.stride())}")
    if tuple(add_74.stride()) != (HEADS * Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected add_74 stride: {tuple(add_74.stride())}")
    _validate_shape_param("_shape_param_0", _shape_param_0, ADD_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, ADD_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_BASE_SHAPE)


def _make_output_base(device: torch.device) -> torch.Tensor:
    return torch.empty(OUT_BASE_SHAPE, device=device, dtype=torch.float32)


def _launch_oracle(
    bmm_70: torch.Tensor,
    add_74: torch.Tensor,
    inductor_seeds: torch.Tensor,
    out_base: torch.Tensor,
    *,
    block_m: int,
    num_warps: int,
) -> torch.Tensor:
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")
    if tuple(out_base.shape) != OUT_BASE_SHAPE:
        raise ValueError(f"output base must have shape {OUT_BASE_SHAPE}, got {tuple(out_base.shape)}")
    if tuple(out_base.stride()) != (Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"output base must be contiguous, got stride {tuple(out_base.stride())}")
    if out_base.dtype != torch.float32 or not out_base.is_cuda:
        raise ValueError("output base must be CUDA fp32")

    _softmax_dropout_layout_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm_70,
        add_74,
        inductor_seeds,
        out_base,
        BLOCK_M=block_m,
        BLOCK_K=K_LEN,
        N_ROWS_CONST=N_ROWS,
        K_LEN_CONST=K_LEN,
        SEED_INDEX_CONST=SEED_INDEX,
        DROPOUT_P_CONST=DROPOUT_P,
        DROPOUT_SCALE_CONST=DROPOUT_SCALE,
        num_warps=num_warps,
    )
    out = out_base.permute(0, 2, 1)
    if tuple(out.stride()) != OUT_VIEW_STRIDE:
        raise RuntimeError(f"unexpected output stride: {tuple(out.stride())}")
    return out


def oracle_online_softmax_dropout(
    bmm_70: torch.Tensor,
    add_74: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    block_m: int = 2,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(
        bmm_70,
        add_74,
        inductor_seeds,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    out_base = _make_output_base(bmm_70.device)
    return _launch_oracle(
        bmm_70,
        add_74,
        inductor_seeds,
        out_base,
        block_m=block_m,
        num_warps=num_warps,
    )


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    return oracle_online_softmax_dropout(*inputs)


def main() -> None:
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
