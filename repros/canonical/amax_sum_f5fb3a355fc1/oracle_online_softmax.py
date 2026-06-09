"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo scalar-fill masked additive-bias softmax returned by Repro.forward, including the [512,128,128] to [32,16,128,128] view, [1,1,2048,2048] bool mask slice to [1,1,128,128], scalar `full_2` fill for masked-off positions, [32,1,128,128] broadcast bias add, stable last-dimension softmax, expand, and final contiguous [512,128,128] view, whereas Inductor lowers the decomposed view/slice/where/add/amax/sub/exp/sum/div/expand/view graph through a generic online-softmax schedule without a template that hoists the shared mask and per-batch/query bias across head rows; Inductor cannot do this today because its pattern library does not recognize this scalar-fill masked attention softmax with broadcast bias as one reusable row-blocked online-softmax lowering; the fix is NEW_PATTERN: add a guarded K=128 masked scalar-fill additive-bias attention softmax template that reuses the sliced mask and bias across heads while writing the final view layout directly."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 16
Q_LEN = 128
K_LEN = 128
MASK_SHAPE = (1, 1, 2048, 2048)
BMM_SHAPE = (BATCH * HEADS, Q_LEN, K_LEN)
BIAS_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
VIEW_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
OUT_SHAPE = BMM_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, K_LEN, 1)


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _masked_scalar_bias_softmax_heads_kernel(
    bmm_ptr,
    mask_ptr,
    fill_ptr,
    bias_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    bias_s0: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
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
    head_block = tl.program_id(1)

    batch = batch_q // q_len
    q = batch_q - batch * q_len
    head_offsets = head_block * block_h + tl.arange(0, block_h)
    cols = tl.arange(0, block_k)

    head_mask = head_offsets < heads
    col_mask = cols < k_len
    elem_mask = head_mask[:, None] & col_mask[None, :]

    keep = tl.load(
        mask_ptr + q * mask_s2 + cols * mask_s3,
        mask=col_mask,
        other=0,
    )
    bias_vals = tl.load(
        bias_ptr + batch * bias_s0 + q * bias_s2 + cols * bias_s3,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    flat_bh = batch * heads + head_offsets
    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q * bmm_s1
        + cols[None, :] * bmm_s2
    )
    bmm_vals = tl.load(
        bmm_ptr + bmm_offsets,
        mask=elem_mask & keep[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.where(keep[None, :], bmm_vals, fill) + bias_vals[None, :]
    scores = tl.where(elem_mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    numer = tl.exp(scores - row_max[:, None])
    denom = tl.sum(numer, axis=1)
    out_vals = numer / denom[:, None]

    out_offsets = (
        flat_bh[:, None] * out_s0
        + q * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=elem_mask)


@triton.jit
def _masked_scalar_bias_softmax_rows_kernel(
    bmm_ptr,
    mask_ptr,
    fill_ptr,
    bias_ptr,
    out_ptr,
    bmm_s0: tl.constexpr,
    bmm_s1: tl.constexpr,
    bmm_s2: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    bias_s0: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    n_rows: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    flat_bh = rows // q_len
    batch = flat_bh // heads
    q = rows - flat_bh * q_len
    cols = tl.arange(0, block_k)

    row_mask = rows < n_rows
    col_mask = cols < k_len
    elem_mask = row_mask[:, None] & col_mask[None, :]

    keep = tl.load(
        mask_ptr + q[:, None] * mask_s2 + cols[None, :] * mask_s3,
        mask=elem_mask,
        other=0,
    )
    bias_vals = tl.load(
        bias_ptr + batch[:, None] * bias_s0 + q[:, None] * bias_s2 + cols[None, :] * bias_s3,
        mask=elem_mask,
        other=0.0,
    ).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q[:, None] * bmm_s1
        + cols[None, :] * bmm_s2
    )
    bmm_vals = tl.load(
        bmm_ptr + bmm_offsets,
        mask=elem_mask & keep,
        other=0.0,
    ).to(tl.float32)
    scores = tl.where(keep, bmm_vals, fill) + bias_vals
    scores = tl.where(elem_mask, scores, -float("inf"))

    row_max = tl.max(scores, axis=1)
    numer = tl.exp(scores - row_max[:, None])
    denom = tl.sum(numer, axis=1)
    out_vals = numer / denom[:, None]

    out_offsets = (
        flat_bh[:, None] * out_s0
        + q[:, None] * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=elem_mask)


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    actual_tuple = tuple(int(dim) for dim in actual)
    if actual_tuple != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual_tuple}")


def _validate_inputs(
    bmm_46: torch.Tensor,
    arg330_1: torch.Tensor,
    full_2: torch.Tensor,
    where: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
) -> None:
    if not (bmm_46.is_cuda and arg330_1.is_cuda and full_2.is_cuda and where.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_46.dtype != torch.float32 or full_2.dtype != torch.float32 or where.dtype != torch.float32:
        raise TypeError(
            "expected fp32 scores, scalar fill, and bias, "
            f"got {bmm_46.dtype}, {full_2.dtype}, {where.dtype}"
        )
    if arg330_1.dtype != torch.bool:
        raise TypeError(f"expected arg330_1 bool mask, got {arg330_1.dtype}")
    if tuple(bmm_46.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_46 shape: {tuple(bmm_46.shape)}")
    if tuple(arg330_1.shape) != MASK_SHAPE:
        raise ValueError(f"unexpected arg330_1 shape: {tuple(arg330_1.shape)}")
    if full_2.ndim != 0:
        raise ValueError(f"expected scalar full_2, got shape {tuple(full_2.shape)}")
    if tuple(where.shape) != BIAS_SHAPE:
        raise ValueError(f"unexpected where shape: {tuple(where.shape)}")
    if tuple(bmm_46.stride()) != OUT_STRIDE:
        raise ValueError(f"unexpected bmm_46 stride: {tuple(bmm_46.stride())}")
    if tuple(where.stride()) != (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected where stride: {tuple(where.stride())}")
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _validate_launch(
    out: torch.Tensor,
    *,
    block_h: int,
    block_m: int,
    block_k: int,
    tile_heads: bool,
) -> None:
    if tuple(out.shape) != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError(f"output must be CUDA fp32 with shape {OUT_SHAPE}")
    if tuple(out.stride()) != OUT_STRIDE:
        raise ValueError(f"output stride must be {OUT_STRIDE}, got {tuple(out.stride())}")
    if block_k < K_LEN or block_k & (block_k - 1):
        raise ValueError(f"block_k must be a power of two >= {K_LEN}, got {block_k}")
    if tile_heads:
        if block_h <= 0 or block_h & (block_h - 1):
            raise ValueError(f"block_h must be a positive power of two, got {block_h}")
    elif block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")


def _launch_oracle(
    bmm_46: torch.Tensor,
    arg330_1: torch.Tensor,
    full_2: torch.Tensor,
    where: torch.Tensor,
    out: torch.Tensor,
    *,
    block_h: int,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
) -> torch.Tensor:
    _validate_launch(
        out,
        block_h=block_h,
        block_m=block_m,
        block_k=block_k,
        tile_heads=tile_heads,
    )
    if tile_heads:
        _masked_scalar_bias_softmax_heads_kernel[
            (BATCH * Q_LEN, triton.cdiv(HEADS, block_h))
        ](
            bmm_46,
            arg330_1,
            full_2,
            where,
            out,
            bmm_s0=bmm_46.stride(0),
            bmm_s1=bmm_46.stride(1),
            bmm_s2=bmm_46.stride(2),
            mask_s2=arg330_1.stride(2),
            mask_s3=arg330_1.stride(3),
            bias_s0=where.stride(0),
            bias_s2=where.stride(2),
            bias_s3=where.stride(3),
            out_s0=out.stride(0),
            out_s1=out.stride(1),
            out_s2=out.stride(2),
            heads=HEADS,
            q_len=Q_LEN,
            k_len=K_LEN,
            block_h=block_h,
            block_k=block_k,
            num_warps=num_warps,
        )
        return out

    _masked_scalar_bias_softmax_rows_kernel[
        (triton.cdiv(BATCH * HEADS * Q_LEN, block_m),)
    ](
        bmm_46,
        arg330_1,
        full_2,
        where,
        out,
        bmm_s0=bmm_46.stride(0),
        bmm_s1=bmm_46.stride(1),
        bmm_s2=bmm_46.stride(2),
        mask_s2=arg330_1.stride(2),
        mask_s3=arg330_1.stride(3),
        bias_s0=where.stride(0),
        bias_s2=where.stride(2),
        bias_s3=where.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        n_rows=BATCH * HEADS * Q_LEN,
        heads=HEADS,
        q_len=Q_LEN,
        k_len=K_LEN,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([512, 128, 128], f32), T([1, 1, 2048, 2048], b8), T([], f32), T([32, 1, 128, 128], f32), S([32, 16, 128, 128]), S([32, 16, 128, 128]), S([512, 128, 128]))")
def oracle_forward(
    inputs: tuple[Any, ...],
    *,
    block_h: int = HEADS,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 4,
    tile_heads: bool = True,
) -> torch.Tensor:
    """Compute exactly Repro()(*inputs): same inputs and same output contract."""
    (
        bmm_46,
        arg330_1,
        full_2,
        where,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    _validate_inputs(
        bmm_46,
        arg330_1,
        full_2,
        where,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_46.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        bmm_46,
        arg330_1,
        full_2,
        where,
        out,
        block_h=block_h,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
    )


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
