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


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _make_inputs(module, seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)


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


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected_f32.abs().clamp_min(1e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


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
    module,
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


def run_check(
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
    ok_all_inf = _check_one(
        model,
        _clone_with_all_inf_probe(inputs),
        label="forced all-inf/partial-inf rows",
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        tile_heads=tile_heads,
        rtol=rtol,
        atol=atol,
    )

    ok = ok_default and ok_all_inf
    print(
        "check compared against full Repro.forward return value, including "
        "bmm view, strided add, any/all-inf handling, stable softmax, expand, "
        "and final view"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(
    block_k: int,
    num_warps: int,
    block_m: int,
    tile_heads: bool,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    _validate_inputs(*inputs)
    bmm_30, add_6, *_ = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_30.device,
        dtype=torch.float32,
    )

    score_bytes = N_ROWS * K_LEN * 4
    logical_bytes = score_bytes * 3
    print(
        "oracle shape: "
        f"bmm_30=f32{BMM_SHAPE} stride={tuple(bmm_30.stride())} "
        f"add_6=f32{ADD_SHAPE} stride={tuple(add_6.stride())} "
        f"out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"rows={N_ROWS} block_m={block_m} block_k={block_k} num_warps={num_warps} "
        f"tile_heads={tile_heads} "
        f"logical_read_write_bytes={logical_bytes / 1e6:.1f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
                bmm_30,
                add_6,
                out,
                block_m=block_m,
                block_k=block_k,
                num_warps=num_warps,
                tile_heads=tile_heads,
            ),
            warmup=warmup,
            rep=rep,
        )
    oracle_bw = logical_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope masked softmax: "
        f"{oracle_us:.3f} us ({oracle_bw:.3f} TB/s logical bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    if no_compile:
        return

    print(
        "torch.compile timings cover the same repro.py view, add, any/all-inf "
        "guard, stable softmax, where, expand, and final view"
    )
    compile_times: list[tuple[str, float]] = []
    holder: list[Any] = [None]
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module, inputs, config, warmup)
            us = _bench_cuda_graph(
                lambda: holder.__setitem__(0, compiled(*inputs)),
                warmup=warmup,
                rep=rep,
            )
            compile_times.append((label, us))
            print(f"torch.compile {label}: {us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")

    if compile_times:
        best_compile = min(us for _, us in compile_times)
        valid_floor = (
            len(compile_times) == len(COMPILE_CONFIGS)
            and oracle_us < min(us for _, us in compile_times)
        )
        print(f"best_required_compile_us={best_compile:.3f}")
        print(f"valid_floor={valid_floor}")
        if not valid_floor:
            print("diagnosis_only: oracle is not a true floor because a required compile config is faster")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=50, help="benchmark repetitions")
    parser.add_argument("--block-m", type=int, default=1, help="Triton rows per program")
    parser.add_argument("--block-k", type=int, default=K_LEN, help="Triton reduction tile size")
    parser.add_argument("--num-warps", type=int, default=1, help="Triton warps per row program")
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
        help="use row-order tiling instead of head tiling",
    )
    parser.set_defaults(tile_heads=True)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile baselines for the requested configs",
    )
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
