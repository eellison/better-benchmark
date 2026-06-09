"""Full-scope Triton oracle for any_amax_sum_56d491110e60.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete
MT5 inference attention-softmax region from repro.py, including the view of
bmm_2 to [32, 6, 128, 128], the relative-position bucket math, the embedding
load from arg6_1, the tautological decoder-position mask, the any(eq(-inf))
all-masked-row guard, stable last-dimension softmax, zero fill for all--inf
rows, expand, and final contiguous [192, 128, 128] view. It differs from
Inductor by recomputing the cheap structured bucket/mask predicates inside a
head-tiled row-softmax kernel, reading each score once and never materializing
the bucket, embedding, boolean guard, or intermediate softmax tensors. Inductor
cannot do this today because the decomposed iota/log/embedding/permute/add/
eq/any/amax/exp/sum/div/where graph is scheduled as generic producer and
online-softmax fragments rather than recognized as a T5/MT5 relative-position
attention-softmax template. The fix class is NEW_PATTERN: add an Inductor
lowering for relative-position attention softmax that fuses bucket lookup,
structured mask handling, all--inf row handling, and the layout-only epilogue
into the row kernel.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
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


REPRO_ID = "any_amax_sum_56d491110e60"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
QUEUE_PATH = REPO_ROOT / "investigation_results" / "oracle_gap_closure_queue.csv"
HISTORICAL_BEST_COMPILE_US = 16.127999871969223
CLASSIFICATION = "NEW_PATTERN"

BATCH = 32
HEADS = 6
Q_LEN = 128
K_LEN = 128
BH = BATCH * HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
MASK_EXPAND_SHAPE_PARAM = (BATCH, -1, Q_LEN, K_LEN)
OUT_SHAPE = BMM_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
        "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
        "triton.multi_kernel=3",
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
def _mt5_relative_softmax_heads_kernel(
    bmm_ptr,
    rel_bias_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    rel_s0: tl.constexpr,
    rel_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
):
    batch_q = tl.program_id(0)
    batch = batch_q // q_len
    q = batch_q - batch * q_len

    h = tl.arange(0, block_h)
    cols = tl.arange(0, block_k)
    h_mask = h < heads
    col_mask = cols < k_len
    mask = h_mask[:, None] & col_mask[None, :]
    flat_bh = batch * heads + h

    rel = cols.to(tl.int32) - q
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

    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q * bmm_s1
        + cols[None, :] * bmm_s2
    )
    bias_offsets = bucket[None, :] * rel_s0 + h[:, None] * rel_s1
    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
    bias_vals = tl.load(rel_bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = bmm_vals + bias_vals
    scores = tl.where(mask, scores, -float("inf"))

    finite = (scores != -float("inf")) & mask
    has_any = tl.sum(tl.where(finite, 1, 0), axis=1) > 0

    row_max = tl.max(scores, axis=1)
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    out_vals = tl.where(has_any[:, None], numer / denom[:, None], 0.0)

    out_offsets = flat_bh[:, None] * out_s0 + q * out_s1 + cols[None, :] * out_s2
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


@triton.jit
def _mt5_relative_softmax_block_m_kernel(
    bmm_ptr,
    rel_bias_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    rel_s0: tl.constexpr,
    rel_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    n_rows: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    flat_bh = rows // q_len
    q = rows - flat_bh * q_len
    head = flat_bh - (flat_bh // heads) * heads

    cols = tl.arange(0, block_k)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    rel = cols[None, :].to(tl.int32) - q[:, None]
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

    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q[:, None] * bmm_s1
        + cols[None, :] * bmm_s2
    )
    bias_offsets = bucket * rel_s0 + head[:, None] * rel_s1
    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
    bias_vals = tl.load(rel_bias_ptr + bias_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = bmm_vals + bias_vals
    scores = tl.where(mask, scores, -float("inf"))

    finite = (scores != -float("inf")) & mask
    has_any = tl.sum(tl.where(finite, 1, 0), axis=1) > 0

    row_max = tl.max(scores, axis=1)
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    out_vals = tl.where(has_any[:, None], numer / denom[:, None], 0.0)

    out_offsets = (
        flat_bh[:, None] * out_s0
        + q[:, None] * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


def _load_repro_module() -> Any:
    sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_inputs() -> tuple[Any, ...]:
    module = _load_repro_module()
    return tuple(module.make_inputs())


def get_repro_instance() -> torch.nn.Module:
    module = _load_repro_module()
    return module.Repro()


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _make_inputs(module: Any, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return tuple(module.make_inputs())


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    actual_tuple = tuple(int(dim) for dim in actual)
    if actual_tuple != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual_tuple}")


def _validate_inputs(
    bmm_2: torch.Tensor,
    arg6_1: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
    _shape_param_3: Any,
) -> None:
    if not (bmm_2.is_cuda and arg6_1.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_2.dtype != torch.float32 or arg6_1.dtype != torch.float32:
        raise TypeError(f"expected fp32 inputs, got {bmm_2.dtype} and {arg6_1.dtype}")
    if tuple(bmm_2.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_2 shape: {tuple(bmm_2.shape)}")
    if tuple(arg6_1.shape) != (32, HEADS):
        raise ValueError(f"unexpected arg6_1 shape: {tuple(arg6_1.shape)}")
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, MASK_EXPAND_SHAPE_PARAM)
    _validate_shape_param("_shape_param_2", _shape_param_2, VIEW_SHAPE)
    _validate_shape_param("_shape_param_3", _shape_param_3, OUT_SHAPE)


def _validate_launch(out: torch.Tensor, block_m: int, block_k: int) -> None:
    if out.shape != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError(f"output must be CUDA fp32 with shape {OUT_SHAPE}")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"output stride must be {OUT_STRIDE}, got {tuple(out.stride())}")
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")


def _launch_oracle(
    bmm_2: torch.Tensor,
    arg6_1: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
) -> torch.Tensor:
    _validate_launch(out, block_m, block_k)
    if tile_heads:
        _mt5_relative_softmax_heads_kernel[(BATCH * Q_LEN,)](
            bmm_2,
            arg6_1,
            out,
            bmm_s0=bmm_2.stride(0),
            bmm_s1=bmm_2.stride(1),
            bmm_s2=bmm_2.stride(2),
            rel_s0=arg6_1.stride(0),
            rel_s1=arg6_1.stride(1),
            out_s0=out.stride(0),
            out_s1=out.stride(1),
            out_s2=out.stride(2),
            heads=HEADS,
            q_len=Q_LEN,
            k_len=K_LEN,
            block_h=triton.next_power_of_2(HEADS),
            block_k=block_k,
            num_warps=num_warps,
        )
        return out

    _mt5_relative_softmax_block_m_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm_2,
        arg6_1,
        out,
        bmm_s0=bmm_2.stride(0),
        bmm_s1=bmm_2.stride(1),
        bmm_s2=bmm_2.stride(2),
        rel_s0=arg6_1.stride(0),
        rel_s1=arg6_1.stride(1),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        heads=HEADS,
        q_len=Q_LEN,
        k_len=K_LEN,
        n_rows=N_ROWS,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([192, 128, 128], f32), T([32, 6], f32), S([32, 6, 128, 128]), S([32, -1, 128, 128]), S([32, 6, 128, 128]), S([192, 128, 128]))")
def oracle_forward(
    inputs: tuple[Any, ...],
    *,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 1,
    tile_heads: bool = True,
) -> torch.Tensor:
    """Compute exactly Repro()(*inputs): same inputs, one fp32 strided output."""
    bmm_2, arg6_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    _validate_inputs(bmm_2, arg6_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_2.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        bmm_2,
        arg6_1,
        out,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
    )


def _clone_with_edge_cases(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    bmm_2, arg6_1, *shape_params = inputs
    bmm_clone = bmm_2.clone()
    arg6_clone = arg6_1.clone()

    bmm_clone[0, 0, :] = -float("inf")
    bmm_clone[0, 1, 0:4] = -float("inf")
    arg6_clone[0, 0] = -float("inf")
    return (bmm_clone, arg6_clone, *shape_params)


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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
