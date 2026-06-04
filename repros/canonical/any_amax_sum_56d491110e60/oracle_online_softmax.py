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
import math
import sys
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


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
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
    rtol: float,
    atol: float,
) -> bool:
    _validate_inputs(*inputs)
    with torch.no_grad():
        expected = _as_tuple(model(*inputs))
        actual = _as_tuple(
            oracle_forward(
                inputs,
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


def run_check(
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
    rtol: float,
    atol: float,
) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
    model = module.Repro().cuda()

    ok_default = _check_one(
        model,
        inputs,
        label="default inputs",
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
        rtol=rtol,
        atol=atol,
    )
    ok_edges = _check_one(
        model,
        _clone_with_edge_cases(inputs),
        label="forced all-inf/partial-inf/bias-inf rows",
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
        rtol=rtol,
        atol=atol,
    )

    ok = ok_default and ok_edges
    print(
        "check compared against full Repro.forward return value, including "
        "bmm view, MT5 relative-position bucket embedding, tautological mask, "
        "any/all-inf handling, stable softmax, expand, and final view"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda_graph(fn, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
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
):
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


def _historical_best_compile_us() -> float:
    if not QUEUE_PATH.exists():
        return HISTORICAL_BEST_COMPILE_US
    try:
        with QUEUE_PATH.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("repro_id") == REPRO_ID:
                    value = row.get("best_compile_us") or ""
                    return float(value) if value else HISTORICAL_BEST_COMPILE_US
    except Exception:
        return HISTORICAL_BEST_COMPILE_US
    return HISTORICAL_BEST_COMPILE_US


def run_bench(
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
    tile_heads: bool,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    _validate_inputs(*inputs)
    bmm_2, arg6_1, *_ = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_2.device,
        dtype=torch.float32,
    )

    score_bytes = N_ROWS * K_LEN * 4
    output_bytes = N_ROWS * K_LEN * 4
    bias_logical_bytes = N_ROWS * K_LEN * 4
    logical_bytes = score_bytes + output_bytes + bias_logical_bytes

    print(
        "oracle shape: "
        f"bmm_2=f32{BMM_SHAPE} stride={tuple(bmm_2.stride())} "
        f"arg6_1=f32{tuple(arg6_1.shape)} stride={tuple(arg6_1.stride())} "
        f"out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"rows={N_ROWS} block_m={block_m} block_k={block_k} "
        f"num_warps={num_warps} tile_heads={tile_heads} "
        f"logical_read_write_bytes={logical_bytes / 1e6:.1f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
                bmm_2,
                arg6_1,
                out,
                block_m=block_m,
                block_k=block_k,
                num_warps=num_warps,
                tile_heads=tile_heads,
            ),
            warmup=warmup,
            rep=rep,
        )

    historical_best = _historical_best_compile_us()
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
    comparison_best = historical_best
    if compile_results:
        comparison_best = min(historical_best, measured_best)
    speedup_vs_best = comparison_best / oracle_us if oracle_us > 0 else math.nan
    valid_floor = bool(
        len(compile_results) == len(COMPILE_CONFIGS)
        and all(oracle_us < value for value in compile_results.values())
        and oracle_us < historical_best
    )

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(measured_best, 3) if compile_results else None,
        "compile_configs_us": {k: round(v, 3) for k, v in compile_results.items()},
        "historical_best_compile_us": round(historical_best, 3),
        "comparison_best_compile_us": round(comparison_best, 3),
        "ratio": round(speedup_vs_best, 3),
        "valid_floor": valid_floor,
        "status": "GOOD" if valid_floor else "DIAGNOSIS_ONLY",
        "classification": CLASSIFICATION,
        "logical_bytes": logical_bytes,
    }
    print(json.dumps(result))
    print(f"oracle full-scope MT5 relative-position softmax: {oracle_us:.3f} us")
    for label, value in compile_results.items():
        print(f"torch.compile {label}: {value:.3f} us")
    print(f"historical best_compile_us: {historical_best:.3f} us")
    print(f"valid floor: {'yes' if valid_floor else 'no'}")
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
    parser.add_argument("--block-m", type=int, default=4, help="Triton rows per program")
    parser.add_argument("--block-k", type=int, default=K_LEN, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=1, help="Triton warps per program")
    parser.add_argument(
        "--tile-heads",
        dest="tile_heads",
        action="store_true",
        help="tile all heads for each batch/query",
    )
    parser.add_argument(
        "--no-tile-heads",
        dest="tile_heads",
        action="store_false",
        help="use block-m rows instead of head tiling",
    )
    parser.set_defaults(tile_heads=True)
    parser.add_argument("--rtol", type=float, default=1e-4, help="correctness relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-5, help="correctness absolute tolerance")
    parser.add_argument("--no-compile", action="store_true", help="skip torch.compile baselines")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    block_k = max(args.block_k, triton.next_power_of_2(K_LEN))
    if args.check and not run_check(
        block_m=args.block_m,
        block_k=block_k,
        num_warps=args.num_warps,
        tile_heads=args.tile_heads,
        rtol=args.rtol,
        atol=args.atol,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            block_m=args.block_m,
            block_k=block_k,
            num_warps=args.num_warps,
            tile_heads=args.tile_heads,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
