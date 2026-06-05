"""Full-scope Triton oracle for any_amax_sum_c1a313554420.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete
MT5 masked softmax region from repro.py, including the view of bmm_30 to
[32, 6, 128, 128], the strided add_6 bias add, the any(eq(-inf)) all-masked-row
guard, stable last-dimension softmax, zero fill for all--inf rows, expand, and
final contiguous [192, 128, 128] view. It differs from Inductor by fusing the
mask predicate, row reductions, normalization, and final view into one row
kernel that reads each score and bias once and never materializes the
intermediate boolean/all-inf tensors. Inductor cannot do this today because the
decomposed eq/logical_not/any/logical_not/amax/sub/exp/sum/div/where graph is
scheduled as generic pointwise plus reduction fragments rather than recognized
as a masked-softmax template with a semantic all-masked-row epilogue. The fix
class is NEW_PATTERN: add an Inductor lowering for this all--inf-safe attention
masked softmax that fuses the predicate, reductions, zero-row handling, and
layout-only epilogue into the row kernel.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
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



REPRO_ID = "any_amax_sum_c1a313554420"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 6
Q_LEN = 128
K_LEN = 128
BH = BATCH * HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
ADD_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
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
def _masked_softmax_add_kernel(
    bmm_ptr,
    add_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    add_s0: tl.constexpr,
    add_s1: tl.constexpr,
    add_s2: tl.constexpr,
    add_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_k: tl.constexpr,
):
    row = tl.program_id(0)
    flat_bh = row // q_len
    q = row - flat_bh * q_len
    batch = flat_bh // heads
    head = flat_bh - batch * heads

    cols = tl.arange(0, block_k)
    col_mask = cols < k_len

    bmm_offsets = flat_bh * bmm_s0 + q * bmm_s1 + cols * bmm_s2
    add_offsets = batch * add_s0 + head * add_s1 + q * add_s2 + cols * add_s3

    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=col_mask, other=0.0).to(tl.float32)
    add_vals = tl.load(add_ptr + add_offsets, mask=col_mask, other=0.0).to(tl.float32)
    scores = bmm_vals + add_vals
    scores = tl.where(col_mask, scores, -float("inf"))

    not_neg_inf = scores != -float("inf")
    has_any = tl.sum(tl.where(not_neg_inf & col_mask, 1, 0), axis=0) > 0

    row_max = tl.max(scores, axis=0)
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max) * 1.4426950408889634)
    numer = tl.where(col_mask, numer, 0.0)
    denom = tl.sum(numer, axis=0)
    out_vals = tl.where(has_any, numer / denom, 0.0)

    out_offsets = flat_bh * out_s0 + q * out_s1 + cols * out_s2
    tl.store(out_ptr + out_offsets, out_vals, mask=col_mask)


@triton.jit
def _masked_softmax_add_block_m_kernel(
    bmm_ptr,
    add_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    add_s0: tl.constexpr,
    add_s1: tl.constexpr,
    add_s2: tl.constexpr,
    add_s3: tl.constexpr,
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
    row_offsets = tl.program_id(0) * block_m + tl.arange(0, block_m)
    flat_bh = row_offsets // q_len
    q = row_offsets - flat_bh * q_len
    batch = flat_bh // heads
    head = flat_bh - batch * heads

    cols = tl.arange(0, block_k)
    row_mask = row_offsets < n_rows
    col_mask = cols < k_len
    mask = row_mask[:, None] & col_mask[None, :]

    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q[:, None] * bmm_s1
        + cols[None, :] * bmm_s2
    )
    add_offsets = (
        batch[:, None] * add_s0
        + head[:, None] * add_s1
        + q[:, None] * add_s2
        + cols[None, :] * add_s3
    )

    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
    add_vals = tl.load(add_ptr + add_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = bmm_vals + add_vals
    scores = tl.where(mask, scores, -float("inf"))

    not_neg_inf = scores != -float("inf")
    has_any = tl.sum(tl.where(not_neg_inf & mask, 1, 0), axis=1) > 0

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


@triton.jit
def _masked_softmax_add_heads_kernel(
    bmm_ptr,
    add_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    add_s0: tl.constexpr,
    add_s1: tl.constexpr,
    add_s2: tl.constexpr,
    add_s3: tl.constexpr,
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

    bmm_offsets = flat_bh[:, None] * bmm_s0 + q * bmm_s1 + cols[None, :] * bmm_s2
    add_offsets = (
        batch * add_s0
        + h[:, None] * add_s1
        + q * add_s2
        + cols[None, :] * add_s3
    )

    bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=mask, other=0.0).to(tl.float32)
    add_vals = tl.load(add_ptr + add_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = bmm_vals + add_vals
    scores = tl.where(mask, scores, -float("inf"))

    not_neg_inf = scores != -float("inf")
    has_any = tl.sum(tl.where(not_neg_inf & mask, 1, 0), axis=1) > 0

    row_max = tl.max(scores, axis=1)
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    out_vals = tl.where(has_any[:, None], numer / denom[:, None], 0.0)

    out_offsets = flat_bh[:, None] * out_s0 + q * out_s1 + cols[None, :] * out_s2
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


def _validate_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm_30: torch.Tensor,
    add_6: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> None:
    if not (bmm_30.is_cuda and add_6.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_30.dtype != torch.float32 or add_6.dtype != torch.float32:
        raise TypeError(f"expected fp32 inputs, got {bmm_30.dtype} and {add_6.dtype}")
    if tuple(bmm_30.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_30 shape: {tuple(bmm_30.shape)}")
    if tuple(add_6.shape) != ADD_SHAPE:
        raise ValueError(f"unexpected add_6 shape: {tuple(add_6.shape)}")
    _validate_shape_param("_shape_param_0", _shape_param_0, ADD_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, ADD_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _launch_oracle(
    bmm_30: torch.Tensor,
    add_6: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
) -> torch.Tensor:
    if out.shape != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError(f"output must be CUDA fp32 with shape {OUT_SHAPE}")
    if out.stride() != OUT_STRIDE:
        raise ValueError(f"output stride must be {OUT_STRIDE}, got {out.stride()}")
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")

    if tile_heads:
        _masked_softmax_add_heads_kernel[(BATCH * Q_LEN,)](
            bmm_30,
            add_6,
            out,
            bmm_s0=bmm_30.stride(0),
            bmm_s1=bmm_30.stride(1),
            bmm_s2=bmm_30.stride(2),
            add_s0=add_6.stride(0),
            add_s1=add_6.stride(1),
            add_s2=add_6.stride(2),
            add_s3=add_6.stride(3),
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

    if block_m > 1:
        _masked_softmax_add_block_m_kernel[(triton.cdiv(N_ROWS, block_m),)](
            bmm_30,
            add_6,
            out,
            bmm_s0=bmm_30.stride(0),
            bmm_s1=bmm_30.stride(1),
            bmm_s2=bmm_30.stride(2),
            add_s0=add_6.stride(0),
            add_s1=add_6.stride(1),
            add_s2=add_6.stride(2),
            add_s3=add_6.stride(3),
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

    _masked_softmax_add_kernel[(N_ROWS,)](
        bmm_30,
        add_6,
        out,
        bmm_s0=bmm_30.stride(0),
        bmm_s1=bmm_30.stride(1),
        bmm_s2=bmm_30.stride(2),
        add_s0=add_6.stride(0),
        add_s1=add_6.stride(1),
        add_s2=add_6.stride(2),
        add_s3=add_6.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        heads=HEADS,
        q_len=Q_LEN,
        k_len=K_LEN,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_30: torch.Tensor,
    add_6: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    *,
    block_m: int = 1,
    block_k: int = K_LEN,
    num_warps: int = 1,
    tile_heads: bool = True,
) -> torch.Tensor:
    _validate_inputs(bmm_30, add_6, _shape_param_0, _shape_param_1, _shape_param_2)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_30.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        bmm_30,
        add_6,
        out,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
    )


def _clone_with_all_inf_probe(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    bmm_30, add_6, *shape_params = inputs
    bmm_clone = bmm_30.clone()
    add_clone = torch.empty_strided(
        tuple(add_6.shape),
        tuple(add_6.stride()),
        device=add_6.device,
        dtype=add_6.dtype,
    )
    add_clone.copy_(add_6)

    add_clone[0, 0, 0, :] = -float("inf")
    add_clone[0, 0, 1, 0:4] = -float("inf")
    return (bmm_clone, add_clone, *shape_params)


def _check_one(
    model,
    inputs: tuple[Any, ...],
    *,
    label: str,
    block_k: int,
    num_warps: int,
    block_m: int,
    tile_heads: bool,
    rtol: float,
    atol: float,
) -> bool:
    _validate_inputs(*inputs)
    with torch.no_grad():
        expected = _as_tuple(model(*inputs))
        actual = _as_tuple(
            oracle_online_softmax(
                *inputs,
                block_m=block_m,
                block_k=block_k,
                num_warps=num_warps,
                tile_heads=tile_heads,
            )
        )
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"{label}: output arity mismatch oracle={len(actual)} ref={len(expected)}")
        return False

    ok = True
    for idx, (got_item, ref_item) in enumerate(zip(actual, expected)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            print(f"{label} output[{idx}]: non-tensor equal={item_ok}")
            ok = ok and bool(item_ok)
            continue

        metadata_ok = (
            got_item.shape == ref_item.shape
            and got_item.dtype == ref_item.dtype
            and got_item.stride() == ref_item.stride()
        )
        value_ok = torch.allclose(
            got_item,
            ref_item,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        max_abs, max_rel = _max_diff(got_item, ref_item)
        item_ok = metadata_ok and value_ok
        ok = ok and item_ok
        print(
            f"{label} output[{idx}]: shape={list(got_item.shape)} "
            f"dtype={got_item.dtype} stride={list(got_item.stride())} "
            f"ref_stride={list(ref_item.stride())} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={value_ok} metadata={metadata_ok}"
        )
    return bool(ok)


def oracle_forward(inputs):
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
