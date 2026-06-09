"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full T5 inference attention softmax returned by Repro.forward, including the view/expand/view layout contract, by proving the generated iota/ge/where mask always contributes a zero fp16 bias and folding it before the fp32 row softmax writes the contiguous [8, 2048, 2048] fp16 output, whereas Inductor lowers the decomposed view/full/iota/ge/where/add/cast/amax/sub/exp/sum/div/cast/expand/view graph as generic pointwise mask/bias work adjacent to the softmax reductions; Inductor cannot do this today because its scheduler/codegen simplifier does not prove shape-derived predicates such as arange(2048) >= 0 are tautologically true through expand/where and delete the resulting zero additive bias before reduction scheduling; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate/value simplification that canonicalizes always-true structured masks and removes zero-bias attention additions before existing softmax codegen."""
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "amax_sum_7493058b895f"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 1
N_HEADS = 8
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

    # The captured iota/ge/where path is true for every key position and adds 0.
    scores = tl.load(bmm_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    row_max = tl.max(scores, axis=0)
    numer = tl.exp(scores - row_max)
    denom = tl.sum(numer, axis=0)
    out = numer / denom

    tl.store(out_ptr + offsets, out, mask=mask)


def _validate_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm_14: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
) -> None:
    if not bmm_14.is_cuda:
        raise RuntimeError("CUDA input is required")
    if bmm_14.dtype != torch.float16:
        raise TypeError(f"expected bmm_14 dtype torch.float16, got {bmm_14.dtype}")
    if tuple(bmm_14.shape) != OUT_SHAPE:
        raise ValueError(f"unexpected bmm_14 shape: {tuple(bmm_14.shape)}")
    if tuple(bmm_14.stride()) != OUT_STRIDE:
        raise ValueError(f"unexpected bmm_14 stride: {tuple(bmm_14.stride())}")

    _validate_shape_param("_shape_param_0", _shape_param_0, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_1", _shape_param_1, (BATCH, -1, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_2", _shape_param_2, (BATCH, N_HEADS, Q_LEN, K_LEN))
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def _launch_oracle(
    bmm_14: torch.Tensor,
    out: torch.Tensor,
    *,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    _attention_softmax_kernel[(N_ROWS,)](
        bmm_14,
        out,
        k_len=K_LEN,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


def oracle_full_attention_softmax(
    bmm_14: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    *,
    out: torch.Tensor | None = None,
    block_k: int = K_LEN,
    num_warps: int = 8,
) -> torch.Tensor:
    _validate_inputs(
        bmm_14,
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
            device=bmm_14.device,
            dtype=torch.float16,
        )
    else:
        if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
            raise ValueError(f"unexpected output layout: shape={tuple(out.shape)} stride={tuple(out.stride())}")
        if out.dtype != torch.float16 or out.device != bmm_14.device:
            raise ValueError(f"unexpected output dtype/device: {out.dtype} {out.device}")

    return _launch_oracle(bmm_14, out, block_k=block_k, num_warps=num_warps)


@oracle_impl(hardware="H100", shapes="(T([8, 2048, 2048], f16), S([1, 8, 2048, 2048]), S([1, -1, 2048, 2048]), S([1, 8, 2048, 2048]), S([8, 2048, 2048]))")
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
