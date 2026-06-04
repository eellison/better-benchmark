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
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


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


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected_f32.abs().clamp_min(1e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


def _check_one(
    model: torch.nn.Module,
    inputs: tuple[Any, ...],
    *,
    label: str,
    block_h: int,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
    load_all_scores: bool,
    rtol: float,
    atol: float,
) -> bool:
    _validate_inputs(*inputs)
    with torch.no_grad():
        expected = _as_tuple(model(*inputs))
        actual = _as_tuple(
            oracle_forward(
                inputs,
                block_h=block_h,
                block_m=block_m,
                block_k=block_k,
                num_warps=num_warps,
                tile_heads=tile_heads,
                load_all_scores=load_all_scores,
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


def run_check(
    *,
    block_h: int,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
    load_all_scores: bool,
    rtol: float,
    atol: float,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    inputs = get_inputs()
    model = get_repro_instance().cuda()

    ok_default = _check_one(
        model,
        inputs,
        label="default inputs",
        block_h=block_h,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
        load_all_scores=load_all_scores,
        rtol=rtol,
        atol=atol,
    )
    ok_edges = _check_one(
        model,
        _clone_with_edge_cases(inputs),
        label="forced all-masked/partial-masked/all-inf rows",
        block_h=block_h,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
        load_all_scores=load_all_scores,
        rtol=rtol,
        atol=atol,
    )

    ok = ok_default and ok_edges
    print(
        "check compared against full Repro.forward return value, including "
        "mask slice/broadcast, fp32-min mask fill, additive position bias, "
        "stable softmax, expand, and final contiguous view"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda_graph(fn: Any, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(rep):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def _compile_with_config(
    module: Any,
    inputs: tuple[Any, ...],
    config: dict[str, object],
    warmup: int,
) -> Any:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = module.Repro().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(
    *,
    block_h: int,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
    load_all_scores: bool,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(4321)
    torch.cuda.manual_seed_all(4321)
    module = _load_repro_module()
    inputs = tuple(module.make_inputs())
    _validate_inputs(*inputs)
    arg330_1, bmm_46, where_1, *_ = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_46.device,
        dtype=torch.float32,
    )

    true_mask_values = int(arg330_1[:, :, :Q_LEN, :K_LEN].sum().item())
    full_score_bytes = BATCH * HEADS * Q_LEN * K_LEN * 4
    output_bytes = full_score_bytes
    if tile_heads:
        bias_read_bytes = BATCH * Q_LEN * triton.cdiv(HEADS, block_h) * K_LEN * 4
        mask_read_bytes = BATCH * Q_LEN * triton.cdiv(HEADS, block_h) * K_LEN
    else:
        bias_read_bytes = BATCH * HEADS * Q_LEN * K_LEN * 4
        mask_read_bytes = BATCH * HEADS * Q_LEN * K_LEN
    if load_all_scores:
        estimated_score_read_bytes = full_score_bytes
    else:
        estimated_score_read_bytes = true_mask_values * BATCH * HEADS * 4
    estimated_effective_bytes = (
        estimated_score_read_bytes + output_bytes + bias_read_bytes + mask_read_bytes
    )
    logical_bytes = full_score_bytes * 3 + BATCH * HEADS * Q_LEN * K_LEN

    print(
        "oracle shape: "
        f"arg330_1=bool{MASK_SHAPE} stride={tuple(arg330_1.stride())} "
        f"bmm_46=f32{BMM_SHAPE} stride={tuple(bmm_46.stride())} "
        f"where_1=f32{BIAS_SHAPE} stride={tuple(where_1.stride())} "
        f"out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"tile_heads={tile_heads} block_h={block_h} block_m={block_m} "
        f"block_k={block_k} num_warps={num_warps} "
        f"load_all_scores={load_all_scores} "
        f"mask_true_values={true_mask_values}/{Q_LEN * K_LEN} "
        f"logical_bytes={logical_bytes / 1e6:.1f} MB "
        f"estimated_effective_bytes={estimated_effective_bytes / 1e6:.1f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
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
            ),
            warmup=warmup,
            rep=rep,
        )

    compile_results: dict[str, float] = {}
    if not no_compile:
        holder: list[Any] = [None]
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(module, inputs, config, warmup)
                compile_results[label] = _bench_cuda_graph(
                    lambda compiled=compiled: holder.__setitem__(0, compiled(*inputs)),
                    warmup=warmup,
                    rep=rep,
                )
                torch.cuda.synchronize()
            except Exception as exc:
                print(f"torch.compile {label}: FAILED ({exc})")

    measured_best = min(compile_results.values()) if compile_results else math.nan
    comparison_best = HISTORICAL_BEST_COMPILE_US
    if compile_results:
        comparison_best = min(HISTORICAL_BEST_COMPILE_US, measured_best)
    speedup_vs_best = comparison_best / oracle_us if oracle_us > 0 else math.nan
    valid_floor = bool(
        len(compile_results) == len(COMPILE_CONFIGS)
        and all(oracle_us < value for value in compile_results.values())
        and oracle_us < HISTORICAL_BEST_COMPILE_US
    )

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(measured_best, 3) if compile_results else None,
        "compile_configs_us": {k: round(v, 3) for k, v in compile_results.items()},
        "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
        "comparison_best_compile_us": round(comparison_best, 3),
        "ratio": round(speedup_vs_best, 3),
        "valid_floor": valid_floor,
        "true_floor": valid_floor,
        "status": "GOOD" if valid_floor else "DIAGNOSIS_ONLY",
        "classification": CLASSIFICATION,
        "logical_bytes": logical_bytes,
        "estimated_effective_bytes": estimated_effective_bytes,
    }
    print(json.dumps(result))
    print(f"oracle full-scope masked-bias softmax: {oracle_us:.3f} us")
    for label, value in compile_results.items():
        print(f"torch.compile {label}: {value:.3f} us")
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"true floor: {'yes' if valid_floor else 'no'}")
    if not valid_floor:
        print(
            "diagnosis_only: oracle is not a true floor because it does not beat "
            "every required local compile config and the historical best_compile_us"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-h", type=int, default=16, help="heads per Triton program")
    parser.add_argument("--block-m", type=int, default=4, help="rows per Triton program")
    parser.add_argument("--block-k", type=int, default=K_LEN, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=4, help="Triton warps per program")
    parser.add_argument(
        "--tile-heads",
        dest="tile_heads",
        action="store_true",
        help="tile a block of heads for each batch/query",
    )
    parser.add_argument(
        "--no-tile-heads",
        dest="tile_heads",
        action="store_false",
        help="use block-m consecutive flattened rows instead of head tiling",
    )
    parser.set_defaults(tile_heads=True)
    parser.add_argument("--rtol", type=float, default=1e-4, help="correctness relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5, help="correctness absolute tolerance")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile baselines")
    parser.add_argument(
        "--load-all-scores",
        action="store_true",
        help="load all score columns before applying the bool mask",
    )
    parser.add_argument(
        "--predicated-score-loads",
        dest="load_all_scores",
        action="store_false",
        help="predicate score loads with the bool mask",
    )
    parser.set_defaults(load_all_scores=False)
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    block_k = max(args.block_k, triton.next_power_of_2(K_LEN))
    if args.check and not run_check(
        block_h=args.block_h,
        block_m=args.block_m,
        block_k=block_k,
        num_warps=args.num_warps,
        tile_heads=args.tile_heads,
        load_all_scores=args.load_all_scores,
        rtol=args.rtol,
        atol=args.atol,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            block_h=args.block_h,
            block_m=args.block_m,
            block_k=block_k,
            num_warps=args.num_warps,
            tile_heads=args.tile_heads,
            load_all_scores=args.load_all_scores,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
