"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bidirectional T5 relative-position attention softmax returned by Repro.forward, including logarithmic bucket selection, positive-direction bucket offset, embedding bias gather, tautological zero mask, score add, fp32 row softmax, fp16 cast, expand, and final [H, 2048, 2048] view in one Triton row kernel, whereas Inductor currently lowers the decomposed iota/abs/log/minimum/where/embedding/permute/mask/add/amax/sub/exp/sum/div/cast/expand/view graph as generic pointwise bias and mask producers feeding reduction scheduling; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize T5 bidirectional relative-position bucket computation and zero structured masks into a fused attention-softmax template that recomputes cheap indices at point of use; the fix is NEW_PATTERN: add an Inductor lowering for T5 bidirectional relative-position attention softmax that fuses bucket lookup, bias gather, zero-mask elimination, normalization, and output-layout epilogue into the row-kernel schedule."""
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "amax_sum_4bd27b112605"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
Q_LEN = 2048
K_LEN = 2048
N_BUCKETS = 32
DEFAULT_BLOCK_K = K_LEN


@triton.jit
def _t5_bidirectional_relative_position_softmax_kernel(
    bmm_ptr,
    rel_bias_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    rel_bias_s0: tl.constexpr,
    rel_bias_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    head = row // q_len
    query = row - head * q_len

    cols = tl.arange(0, block_k)
    col_mask = cols < k_len
    rel_pos = cols - query
    distance = tl.abs(rel_pos)

    # Exact integer thresholds for int(log(abs(distance / 8)) / log(16) * 8) + 8,
    # clamped to 15, over the captured 2048-wide sequence length.
    bucket = distance
    bucket = tl.where(distance >= 8, 8, bucket)
    bucket = tl.where(distance >= 12, 9, bucket)
    bucket = tl.where(distance >= 16, 10, bucket)
    bucket = tl.where(distance >= 23, 11, bucket)
    bucket = tl.where(distance >= 32, 12, bucket)
    bucket = tl.where(distance >= 46, 13, bucket)
    bucket = tl.where(distance >= 64, 14, bucket)
    bucket = tl.where(distance >= 91, 15, bucket)
    bucket = tl.where(rel_pos > 0, bucket + 16, bucket)

    bmm_offsets = head * bmm_s0 + query * bmm_s1 + cols * bmm_s2
    bias_offsets = bucket * rel_bias_s0 + head * rel_bias_s1

    scores = tl.load(bmm_ptr + bmm_offsets, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(rel_bias_ptr + bias_offsets, mask=col_mask, other=0.0).to(tl.float32)
    scores = scores + bias
    scores = tl.where(col_mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    probs = numer / denom

    out_offsets = head * out_s0 + query * out_s1 + cols * out_s2
    tl.store(out_ptr + out_offsets, probs, mask=col_mask)


def get_inputs() -> list[object]:
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_2: torch.Tensor,
    arg6_1: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> int:
    if not bmm_2.is_cuda or not arg6_1.is_cuda:
        raise RuntimeError("CUDA tensor inputs are required")
    if bmm_2.dtype != torch.float16:
        raise TypeError(f"expected bmm_2 dtype torch.float16, got {bmm_2.dtype}")
    if arg6_1.dtype != torch.float16:
        raise TypeError(f"expected arg6_1 dtype torch.float16, got {arg6_1.dtype}")
    if bmm_2.ndim != 3 or tuple(bmm_2.shape[1:]) != (Q_LEN, K_LEN):
        raise ValueError(f"expected bmm_2 shape [H, {Q_LEN}, {K_LEN}], got {tuple(bmm_2.shape)}")

    n_heads = int(bmm_2.shape[0])
    out_shape = (n_heads, Q_LEN, K_LEN)
    out_stride = (Q_LEN * K_LEN, K_LEN, 1)
    if tuple(bmm_2.stride()) != out_stride:
        raise ValueError(f"expected contiguous bmm_2 stride {out_stride}, got {tuple(bmm_2.stride())}")
    if tuple(arg6_1.shape) != (N_BUCKETS, n_heads):
        raise ValueError(f"expected arg6_1 shape {(N_BUCKETS, n_heads)}, got {tuple(arg6_1.shape)}")
    if tuple(arg6_1.stride()) != (n_heads, 1):
        raise ValueError(f"expected contiguous arg6_1 stride {(n_heads, 1)}, got {tuple(arg6_1.stride())}")

    _validate_shape_param("_shape_param_0", _shape_param_0, (BATCH, n_heads, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, -1, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BATCH, n_heads, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_3", _shape_param_3, out_shape)
    return n_heads


def oracle_full_t5_bidirectional_relative_position_softmax(
    bmm_2: torch.Tensor,
    arg6_1: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
    *,
    out: torch.Tensor | None = None,
    block_k: int = DEFAULT_BLOCK_K,
    num_warps: int = 8,
) -> torch.Tensor:
    n_heads = _validate_inputs(
        bmm_2,
        arg6_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")

    out_shape = (n_heads, Q_LEN, K_LEN)
    out_stride = (Q_LEN * K_LEN, K_LEN, 1)
    if out is None:
        out = torch.empty_strided(
            out_shape,
            out_stride,
            device=bmm_2.device,
            dtype=torch.float16,
        )
    else:
        if tuple(out.shape) != out_shape or tuple(out.stride()) != out_stride:
            raise ValueError(f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}")
        if out.dtype != torch.float16 or out.device != bmm_2.device:
            raise ValueError(f"unexpected output dtype/device: {out.dtype} {out.device}")

    _t5_bidirectional_relative_position_softmax_kernel[(n_heads * Q_LEN,)](
        bmm_2,
        arg6_1,
        out,
        bmm_s0=bmm_2.stride(0),
        bmm_s1=bmm_2.stride(1),
        bmm_s2=bmm_2.stride(2),
        rel_bias_s0=arg6_1.stride(0),
        rel_bias_s1=arg6_1.stride(1),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        q_len=Q_LEN,
        k_len=K_LEN,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([8, 2048, 2048], f16), T([32, 8], f16), S([1, 8, 2048, 2048]), S([1, -1, 2048, 2048]), S([1, 8, 2048, 2048]), S([8, 2048, 2048]))")
def oracle_forward(inputs: tuple[object, ...] | list[object]) -> torch.Tensor:
    return oracle_full_t5_bidirectional_relative_position_softmax(*inputs)


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
