"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full T5 inference attention softmax returned by Repro.forward, including the view/expand/view output contract, by folding the tautological arange(2048) >= 0 mask and zero fp16 bias into one Triton row softmax over the [12, 2048, 2048] scores, whereas Inductor currently lowers the decomposed view/full/iota/ge/expand/where/add/cast/amax/sub/exp/sum/div/cast/expand/view graph as generic pointwise mask/bias work plus softmax reductions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen simplifier does not prove the shape-derived arange(2048) >= 0 predicate remains always true through unsqueeze/expand/where before reduction scheduling; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate and zero-bias simplification that canonicalizes tautological structured masks before softmax scheduling."""
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



REPRO_ID = "amax_sum_a7ede60031ef"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
N_HEADS = 12
Q_LEN = 2048
K_LEN = 2048
N_ROWS = BATCH * N_HEADS * Q_LEN
OUT_SHAPE = (N_HEADS, Q_LEN, K_LEN)
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)


@triton.jit
def _attention_softmax_kernel(
    bmm_ptr,
    out_ptr,
    k_len: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, block_k)
    mask = cols < k_len
    offsets = row * k_len + cols

    # The captured iota/ge/where mask is true for every query position and
    # contributes a zero fp16 bias, so the full graph is just row softmax.
    scores = tl.load(bmm_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    out = numer / denom

    tl.store(out_ptr + offsets, out, mask=mask)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    actual_tuple = tuple(int(dim) for dim in actual)
    if actual_tuple != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual_tuple}")


def _validate_inputs(
    bmm_26: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> None:
    if not bmm_26.is_cuda:
        raise RuntimeError("CUDA input is required")
    if bmm_26.dtype != torch.float16:
        raise TypeError(f"expected bmm_26 dtype torch.float16, got {bmm_26.dtype}")
    if tuple(bmm_26.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected bmm_26 shape: {tuple(bmm_26.shape)}")
    if tuple(bmm_26.stride()) != OUT_STRIDE:
        raise ValueError(f"unexpected bmm_26 stride: {tuple(bmm_26.stride())}")

    _validate_shape_param("_shape_param_0", _shape_param_0, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, -1, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def _launch_oracle(
    bmm_26: torch.Tensor,
    out: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    _attention_softmax_kernel[(N_ROWS,)](
        bmm_26,
        out,
        k_len=K_LEN,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


def oracle_full_attention_softmax(
    bmm_26: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
    *,
    out: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 8,
) -> torch.Tensor:
    _validate_inputs(
        bmm_26,
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
            device=bmm_26.device,
            dtype=torch.float16,
        )
    else:
        if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
            raise ValueError(
                f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}"
            )
        if out.dtype != torch.float16 or out.device != bmm_26.device:
            raise ValueError(f"unexpected output dtype/device: {out.dtype} {out.device}")

    return _launch_oracle(bmm_26, out, block_k=block_k, num_warps=num_warps)


@oracle_impl(hardware="H100", shapes="(T([12, 2048, 2048], f16), S([1, 12, 2048, 2048]), S([1, -1, 2048, 2048]), S([1, 12, 2048, 2048]), S([12, 2048, 2048]))")
def oracle_forward(inputs):
    return oracle_full_attention_softmax(*inputs)


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
