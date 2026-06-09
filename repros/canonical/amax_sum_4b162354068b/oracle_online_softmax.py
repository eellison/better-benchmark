"""Full-scope Triton oracle for amax_sum_4b162354068b.

Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete
GPT-Neo masked additive-bias attention softmax returned by Repro.forward,
including the [1,1,2048,2048] bool mask slice to [1,1,128,128], broadcast of
that mask to fp32-min fill across [32,16,128,128], the [32,1,128,128] fp32 bias add, stable
last-dimension softmax, expand, and final contiguous [512,128,128] view. It
differs from Inductor by using one shape-specialized Triton row-blocked
attention-softmax kernel that tiles heads for each batch/query row, reuses the
same sliced mask and batch/query bias vector across the head block, and writes
the final view layout directly. Inductor can fuse this graph today, but its
generic online-softmax schedule does not express this small-K masked
attention-specific reuse of mask and additive bias across heads. The Inductor
change that would fix it is NEW_PATTERN: add a K=128 masked additive-bias
attention softmax template, or extend the online-softmax scheduler so
broadcast mask/bias producers can be hoisted/reused across head tiles before
normalization and final view emission.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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


REPRO_ID = "amax_sum_4b162354068b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
HISTORICAL_BEST_COMPILE_US = 26.43200010061264
CLASSIFICATION = "NEW_PATTERN"

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
def _masked_bias_softmax_heads_kernel(
    mask_ptr,
    bmm_ptr,
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
    load_all_scores: tl.constexpr,
):
    batch_q = tl.program_id(0)
    head_block = tl.program_id(1)

    batch = batch_q // q_len
    q = batch_q - batch * q_len
    heads_offsets = head_block * block_h + tl.arange(0, block_h)
    cols = tl.arange(0, block_k)

    h_mask = heads_offsets < heads
    col_mask = cols < k_len
    elem_mask = h_mask[:, None] & col_mask[None, :]

    mask_offsets = q * mask_s2 + cols * mask_s3
    keep_cols = tl.load(mask_ptr + mask_offsets, mask=col_mask, other=0)

    bias_offsets = batch * bias_s0 + q * bias_s2 + cols * bias_s3
    bias_vals = tl.load(bias_ptr + bias_offsets, mask=col_mask, other=0.0).to(tl.float32)

    flat_bh = batch * heads + heads_offsets
    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q * bmm_s1
        + cols[None, :] * bmm_s2
    )
    active_mask = elem_mask & keep_cols[None, :]
    if load_all_scores:
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=elem_mask, other=0.0)
        scores = tl.where(
            active_mask,
            bmm_vals.to(tl.float32) + bias_vals[None, :],
            -3.4028234663852886e38,
        )
    else:
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=active_mask, other=0.0)
        scores = bmm_vals.to(tl.float32) + bias_vals[None, :]
        scores = tl.where(active_mask, scores, -3.4028234663852886e38)

    row_max = tl.max(scores, axis=1)
    numer = tl.exp2((scores - row_max[:, None]) * 1.4426950408889634)
    denom = tl.sum(numer, axis=1)
    out_vals = numer / denom[:, None]

    out_offsets = (
        flat_bh[:, None] * out_s0
        + q * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=elem_mask)


@triton.jit
def _masked_bias_softmax_rows_kernel(
    mask_ptr,
    bmm_ptr,
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
    load_all_scores: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    flat_bh = rows // q_len
    batch = flat_bh // heads
    q = rows - flat_bh * q_len

    cols = tl.arange(0, block_k)
    row_mask = rows < n_rows
    col_mask = cols < k_len
    elem_mask = row_mask[:, None] & col_mask[None, :]

    mask_offsets = q[:, None] * mask_s2 + cols[None, :] * mask_s3
    keep = tl.load(mask_ptr + mask_offsets, mask=elem_mask, other=0)

    bias_offsets = batch[:, None] * bias_s0 + q[:, None] * bias_s2 + cols[None, :] * bias_s3
    bias_vals = tl.load(bias_ptr + bias_offsets, mask=elem_mask, other=0.0).to(tl.float32)

    bmm_offsets = (
        flat_bh[:, None] * bmm_s0
        + q[:, None] * bmm_s1
        + cols[None, :] * bmm_s2
    )
    active_mask = elem_mask & keep
    if load_all_scores:
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=elem_mask, other=0.0)
        scores = tl.where(
            active_mask,
            bmm_vals.to(tl.float32) + bias_vals,
            -3.4028234663852886e38,
        )
    else:
        bmm_vals = tl.load(bmm_ptr + bmm_offsets, mask=active_mask, other=0.0)
        scores = bmm_vals.to(tl.float32) + bias_vals
        scores = tl.where(active_mask, scores, -3.4028234663852886e38)

    row_max = tl.max(scores, axis=1)
    numer = tl.exp2((scores - row_max[:, None]) * 1.4426950408889634)
    denom = tl.sum(numer, axis=1)
    out_vals = numer / denom[:, None]

    out_offsets = (
        flat_bh[:, None] * out_s0
        + q[:, None] * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, out_vals, mask=elem_mask)


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


def _validate_shape_param(name: str, actual: Any, expected: tuple[int, ...]) -> None:
    actual_tuple = tuple(int(dim) for dim in actual)
    if actual_tuple != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual_tuple}")


def _validate_inputs(
    arg330_1: torch.Tensor,
    bmm_46: torch.Tensor,
    where_1: torch.Tensor,
    _shape_param_0: Any,
    _shape_param_1: Any,
    _shape_param_2: Any,
) -> None:
    if not (arg330_1.is_cuda and bmm_46.is_cuda and where_1.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if arg330_1.dtype != torch.bool:
        raise TypeError(f"expected arg330_1 bool mask, got {arg330_1.dtype}")
    if bmm_46.dtype != torch.float32 or where_1.dtype != torch.float32:
        raise TypeError(f"expected fp32 scores and bias, got {bmm_46.dtype} and {where_1.dtype}")
    if tuple(arg330_1.shape) != MASK_SHAPE:
        raise ValueError(f"unexpected arg330_1 shape: {tuple(arg330_1.shape)}")
    if tuple(bmm_46.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_46 shape: {tuple(bmm_46.shape)}")
    if tuple(where_1.shape) != BIAS_SHAPE:
        raise ValueError(f"unexpected where_1 shape: {tuple(where_1.shape)}")
    if tuple(bmm_46.stride()) != OUT_STRIDE:
        raise ValueError(f"unexpected bmm_46 stride: {tuple(bmm_46.stride())}")
    if tuple(where_1.stride()) != (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected where_1 stride: {tuple(where_1.stride())}")
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
    if out.shape != OUT_SHAPE or out.dtype != torch.float32 or not out.is_cuda:
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
    arg330_1: torch.Tensor,
    bmm_46: torch.Tensor,
    where_1: torch.Tensor,
    out: torch.Tensor,
    *,
    block_h: int,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
    load_all_scores: bool,
) -> torch.Tensor:
    _validate_launch(
        out,
        block_h=block_h,
        block_m=block_m,
        block_k=block_k,
        tile_heads=tile_heads,
    )
    if tile_heads:
        _masked_bias_softmax_heads_kernel[
            (BATCH * Q_LEN, triton.cdiv(HEADS, block_h))
        ](
            arg330_1,
            bmm_46,
            where_1,
            out,
            bmm_s0=bmm_46.stride(0),
            bmm_s1=bmm_46.stride(1),
            bmm_s2=bmm_46.stride(2),
            mask_s2=arg330_1.stride(2),
            mask_s3=arg330_1.stride(3),
            bias_s0=where_1.stride(0),
            bias_s2=where_1.stride(2),
            bias_s3=where_1.stride(3),
            out_s0=out.stride(0),
            out_s1=out.stride(1),
            out_s2=out.stride(2),
            heads=HEADS,
            q_len=Q_LEN,
            k_len=K_LEN,
            block_h=block_h,
            block_k=block_k,
            load_all_scores=load_all_scores,
            num_warps=num_warps,
        )
        return out

    _masked_bias_softmax_rows_kernel[(triton.cdiv(BATCH * HEADS * Q_LEN, block_m),)](
        arg330_1,
        bmm_46,
        where_1,
        out,
        bmm_s0=bmm_46.stride(0),
        bmm_s1=bmm_46.stride(1),
        bmm_s2=bmm_46.stride(2),
        mask_s2=arg330_1.stride(2),
        mask_s3=arg330_1.stride(3),
        bias_s0=where_1.stride(0),
        bias_s2=where_1.stride(2),
        bias_s3=where_1.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        n_rows=BATCH * HEADS * Q_LEN,
        heads=HEADS,
        q_len=Q_LEN,
        k_len=K_LEN,
        block_m=block_m,
        block_k=block_k,
        load_all_scores=load_all_scores,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([1, 1, 2048, 2048], b8), T([512, 128, 128], f32), T([32, 1, 128, 128], f32), S([32, 16, 128, 128]), S([32, 16, 128, 128]), S([512, 128, 128]))")
def oracle_forward(
    inputs: tuple[Any, ...],
    *,
    block_h: int = 16,
    block_m: int = 4,
    block_k: int = K_LEN,
    num_warps: int = 4,
    tile_heads: bool = True,
    load_all_scores: bool = False,
) -> torch.Tensor:
    """Compute exactly Repro()(*inputs): same inputs, same fp32 contiguous output."""
    arg330_1, bmm_46, where_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    _validate_inputs(
        arg330_1,
        bmm_46,
        where_1,
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
        arg330_1,
        bmm_46,
        where_1,
        out,
        block_h=block_h,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
        load_all_scores=load_all_scores,
    )


def _clone_with_edge_cases(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    arg330_1, bmm_46, where_1, *shape_params = inputs
    mask_clone = arg330_1.clone()
    bmm_clone = bmm_46.clone()
    bias_clone = where_1.clone()

    mask_clone[:, :, :4, :] = True
    mask_clone[:, :, 0, :] = False
    mask_clone[:, :, 1, 0:4] = False
    mask_clone[:, :, 2, :] = True
    mask_clone[:, :, 3, :] = True
    bmm_clone[0, 2, :] = -float("inf")
    bmm_clone[0, 3, 0:4] = -float("inf")
    bias_clone[0, 0, 3, 4:8] = -3.0
    return (mask_clone, bmm_clone, bias_clone, *shape_params)


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
