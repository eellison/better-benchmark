"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete Blenderbot attention softmax returned by Repro.forward, including the [512,128,128] -> [16,32,128,128] view, the shape-derived iota/ge/where mask, the zero/-inf bias add, the any(eq(-inf)) all-masked-row guard, stable last-dimension softmax, zero fill for all--inf rows, expand, and final [512,128,128] view, by folding the tautological arange(128) >= 0 predicate and zero bias into one Triton row-softmax kernel; Inductor currently lowers the decomposed mask construction, add, boolean guard, reductions, division, where, expand, and view as generic pointwise/reduction work instead of proving the structured mask is always true before softmax scheduling; Inductor cannot do this today because its scheduler/codegen simplifier does not canonicalize this shape-derived predicate and zero-bias path through the all--inf-safe softmax idiom; the fix is ALGEBRAIC_ELIMINATION: add shape-aware predicate and zero-bias simplification that canonicalizes tautological masks before emitting the semantic row-softmax schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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


REPRO_ID = "any_amax_sum_0815d0a11243"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
HEADS = 32
Q_LEN = 128
K_LEN = 128
N_ROWS = BATCH * HEADS * Q_LEN
IN_SHAPE = (BATCH * HEADS, Q_LEN, K_LEN)
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
MASK_EXPAND_SHAPE = (BATCH, -1, Q_LEN, K_LEN)
OUT_SHAPE = IN_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _tautological_mask_softmax_kernel(
    scores_ptr,
    out_ptr,
    n_rows: tl.constexpr,
    k_len: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    cols = tl.arange(0, block_k)
    offsets = rows[:, None] * k_len + cols[None, :]

    # The captured iota/ge predicate is true for every query row, so the
    # full mask/bias producer contributes only zero while the all--inf guard
    # must still be preserved for rows whose input scores are all -inf.
    scores = tl.load(scores_ptr + offsets, eviction_policy="evict_first").to(tl.float32)
    row_max = tl.max(scores, axis=1)
    numer = tl.exp(scores - row_max[:, None])
    denom = tl.sum(numer, axis=1)
    probs = tl.where(row_max[:, None] != -float("inf"), numer / denom[:, None], 0.0)

    tl.store(out_ptr + offsets, probs)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    got = _shape_tuple(actual)
    if got != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {got}")


def _validate_inputs(
    bmm_4: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> None:
    if not bmm_4.is_cuda:
        raise RuntimeError("CUDA input is required for this Triton oracle")
    if bmm_4.dtype != torch.float32:
        raise TypeError(f"expected bmm_4 dtype torch.float32, got {bmm_4.dtype}")
    if tuple(bmm_4.shape) != IN_SHAPE:
        raise ValueError(f"expected bmm_4 shape {IN_SHAPE}, got {tuple(bmm_4.shape)}")
    if tuple(bmm_4.stride()) != OUT_STRIDE:
        raise ValueError(f"expected contiguous bmm_4 stride {OUT_STRIDE}, got {tuple(bmm_4.stride())}")

    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, MASK_EXPAND_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, VIEW_SHAPE)
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def oracle_online_softmax(
    bmm_4: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
    *,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 4,
) -> torch.Tensor:
    """Run the full all--inf-safe softmax and final view contract."""
    _validate_inputs(
        bmm_4,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    )
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")
    if block_m <= 0 or N_ROWS % block_m != 0:
        raise ValueError(f"block_m must evenly divide {N_ROWS}, got {block_m}")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_4.device,
        dtype=torch.float32,
    )
    _tautological_mask_softmax_kernel[(N_ROWS // block_m,)](
        bmm_4,
        out,
        n_rows=N_ROWS,
        k_len=K_LEN,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([512, 128, 128], f32), S([16, 32, 128, 128]), S([16, -1, 128, 128]), S([16, 32, 128, 128]), S([512, 128, 128]))")
def oracle_forward(inputs):
    """Run the oracle computation with the same inputs as Repro.forward."""
    return oracle_online_softmax(*inputs)


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
