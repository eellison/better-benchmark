"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete MT5 causal relative-position attention softmax Repro.forward result by recomputing the logarithmic relative-position bucket lookup, embedding bias, causal fp32 mask, score add, any/all-masked guard, and row softmax inside one Triton row kernel, whereas Inductor lowers the decomposed iota/minimum/log/bucket/embedding/permute/mask/add/any/amax/exp/sum/div graph through generic producers and reductions that nevertheless time within noise of the fused oracle; Inductor cannot materially improve this case today because the captured shape is already at the small-row softmax/output-write floor rather than blocked by a profitable missing fusion; the fix is BANDWIDTH_BOUND: no gap-specific scheduler change is justified by this measurement, though a T5/MT5 relative-position softmax pattern could simplify codegen without moving runtime."""
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

BATCH = 32
N_HEADS = 6
Q_LEN = 128
K_LEN = 128
BH = BATCH * N_HEADS
N_ROWS = BH * Q_LEN
OUT_SHAPE = (BH, Q_LEN, K_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)


@triton.jit
def _mt5_relative_position_softmax_kernel(
    bmm_ptr,
    rel_bias_ptr,
    out_ptr,
    H: tl.constexpr,
    Q: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    bh = row // Q
    head = bh - (bh // H) * H
    q = row - bh * Q

    cols = tl.arange(0, BLOCK_K)
    col_mask = cols < K
    distance = tl.where(cols <= q, q - cols, 0)

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

    offsets = row * K + cols
    bmm = tl.load(bmm_ptr + offsets, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(
        rel_bias_ptr + bucket * H + head,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    scores = bmm + bias
    scores = tl.where(cols <= q, scores, -3.4028234663852886e38)
    scores = tl.where(col_mask, scores, -float("inf"))

    finite = scores != -float("inf")
    any_finite = tl.max(finite.to(tl.int32), axis=0) != 0
    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    probs = numer / denom
    probs = tl.where(any_finite, probs, 0.0)

    tl.store(out_ptr + offsets, probs, mask=col_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm: torch.Tensor,
    rel_bias: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> None:
    if not bmm.is_cuda or not rel_bias.is_cuda:
        raise RuntimeError("CUDA tensor inputs are required")
    if bmm.dtype != torch.float32:
        raise TypeError(f"expected bmm dtype torch.float32, got {bmm.dtype}")
    if rel_bias.dtype != torch.float32:
        raise TypeError(f"expected rel_bias dtype torch.float32, got {rel_bias.dtype}")
    if tuple(bmm.shape) != OUT_SHAPE:
        raise ValueError(f"expected bmm shape {OUT_SHAPE}, got {tuple(bmm.shape)}")
    if tuple(bmm.stride()) != OUT_STRIDE:
        raise ValueError(f"expected contiguous bmm stride {OUT_STRIDE}, got {tuple(bmm.stride())}")
    if tuple(rel_bias.shape) != (32, N_HEADS):
        raise ValueError(f"expected rel_bias shape {(32, N_HEADS)}, got {tuple(rel_bias.shape)}")
    if tuple(rel_bias.stride()) != (N_HEADS, 1):
        raise ValueError(f"expected contiguous rel_bias stride {(N_HEADS, 1)}, got {tuple(rel_bias.stride())}")

    _validate_shape_param("_shape_param_0", _shape_param_0, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, -1, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def oracle_full_mt5_relative_position_softmax(
    bmm: torch.Tensor,
    rel_bias: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
    *,
    out: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(
        bmm,
        rel_bias,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")

    if out is None:
        out = torch.empty_strided(
            OUT_SHAPE,
            OUT_STRIDE,
            device=bmm.device,
            dtype=torch.float32,
        )
    else:
        if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
            raise ValueError(f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}")
        if out.dtype != torch.float32 or out.device != bmm.device:
            raise ValueError(f"unexpected output dtype/device: {out.dtype} {out.device}")

    _mt5_relative_position_softmax_kernel[(N_ROWS,)](
        bmm,
        rel_bias,
        out,
        H=N_HEADS,
        Q=Q_LEN,
        K=K_LEN,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def oracle_forward(inputs):
    return oracle_full_mt5_relative_position_softmax(*inputs)


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
