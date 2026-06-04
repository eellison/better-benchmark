"""Full-scope Triton diagnostic oracle for any_amax_sum_727a4ae37fb6.

Gap diagnosis (classification: BANDWIDTH_BOUND): the oracle computes the
complete DistilBERT masked softmax region from repro.py, including the logical
view of bmm_10 to [256, 12, 128, 128], the broadcast bool mask to 0/-inf, the
any(eq(-inf)) all-masked-row guard, stable last-dimension softmax, zero fill
for all--inf rows, expand, and final contiguous [3072, 128, 128] view. It
differs from Inductor by using the original stride-zero mask semantics directly
inside one row-softmax Triton kernel and masking score loads for rows that the
broadcast mask turns into all-zero output. Inductor already recognizes this
decomposed graph as a prepare_softmax_online persistent reduction and emits one
full-scope fused kernel, so the standalone masked-load specialization does not
expose a true lower floor. There is no actionable Inductor fix for this repro
beyond preserving the existing online-softmax/persistent-reduction lowering and
tuning; the row is diagnosis-only when the historical-best gate is applied.
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


REPRO_ID = "any_amax_sum_727a4ae37fb6"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
QUEUE_PATH = REPO_ROOT / "investigation_results" / "oracle_gap_closure_queue.csv"
HISTORICAL_BEST_COMPILE_US = 73.37599992752075

BATCH = 256
HEADS = 12
Q_LEN = 128
K_LEN = 128
BH = BATCH * HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
MASK_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
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
def _broadcast_masked_softmax_kernel(
    bmm_ptr,
    mask_ptr,
    out_ptr,
    mask_s0: tl.constexpr,
    mask_s1: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
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

    mask_offsets = batch * mask_s0 + q * mask_s2
    keep_row = tl.load(mask_ptr + mask_offsets, mask=row_mask, other=0)
    load_mask = row_mask[:, None] & keep_row[:, None] & col_mask[None, :]

    offsets = rows[:, None] * k_len + cols[None, :]
    scores = tl.load(bmm_ptr + offsets, mask=load_mask, other=-float("inf")).to(tl.float32)
    scores = tl.where(load_mask, scores, -float("inf"))

    finite = (scores != -float("inf")) & load_mask
    has_any = tl.sum(tl.where(finite, 1, 0), axis=1) > 0

    row_max = tl.max(scores, axis=1)
    stable_max = tl.where(has_any, row_max, 0.0)
    numer = tl.exp2((scores - stable_max[:, None]) * 1.4426950408889634)
    numer = tl.where(load_mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    out_vals = tl.where(has_any[:, None], numer / denom[:, None], 0.0)

    store_mask = row_mask[:, None] & col_mask[None, :]
    tl.store(out_ptr + offsets, out_vals, mask=store_mask)


def _load_repro_module():
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


def _validate_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if tuple(int(dim) for dim in actual) != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {tuple(actual)}")


def _validate_inputs(
    bmm_10: torch.Tensor,
    expand_2: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> None:
    if not (bmm_10.is_cuda and expand_2.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_10.dtype != torch.float32 or expand_2.dtype != torch.bool:
        raise TypeError(f"expected f32 scores and bool mask, got {bmm_10.dtype} and {expand_2.dtype}")
    if tuple(bmm_10.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_10 shape: {tuple(bmm_10.shape)}")
    if tuple(expand_2.shape) != MASK_SHAPE:
        raise ValueError(f"unexpected expand_2 shape: {tuple(expand_2.shape)}")
    _validate_shape_param("_shape_param_0", _shape_param_0, VIEW_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, VIEW_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


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
    bmm_10: torch.Tensor,
    expand_2: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    _validate_launch(out, block_m, block_k)
    _broadcast_masked_softmax_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm_10,
        expand_2,
        out,
        mask_s0=expand_2.stride(0),
        mask_s1=expand_2.stride(1),
        mask_s2=expand_2.stride(2),
        mask_s3=expand_2.stride(3),
        n_rows=N_ROWS,
        heads=HEADS,
        q_len=Q_LEN,
        k_len=K_LEN,
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
    num_warps: int = 4,
) -> torch.Tensor:
    """Compute exactly Repro()(*inputs): same inputs, one fp32 strided output."""
    bmm_10, expand_2, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    _validate_inputs(bmm_10, expand_2, _shape_param_0, _shape_param_1, _shape_param_2)
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_10.device,
        dtype=torch.float32,
    )
    return _launch_oracle(
        bmm_10,
        expand_2,
        out,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )


def _mask_storage_view(mask: torch.Tensor) -> torch.Tensor:
    base = torch.empty((Q_LEN,), dtype=torch.bool, device=mask.device)
    base.copy_(mask[0, 0, :, 0])
    return base.as_strided(MASK_SHAPE, tuple(mask.stride()))


def _clone_with_edge_cases(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    bmm_10, expand_2, *shape_params = inputs
    bmm_clone = bmm_10.clone()
    mask_clone = _mask_storage_view(expand_2)

    mask_clone[0, 0, 0, 0] = False
    mask_clone[0, 0, 1, 0] = True
    mask_clone[0, 0, 2, 0] = True
    bmm_clone[0, 1, :] = -float("inf")
    bmm_clone[0, 2, 0:4] = -float("inf")
    return (bmm_clone, mask_clone, *shape_params)


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
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        rtol=rtol,
        atol=atol,
    )
    ok_edges = _check_one(
        model,
        _clone_with_edge_cases(inputs),
        label="forced masked/all-inf/partial-inf rows",
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        rtol=rtol,
        atol=atol,
    )

    ok = ok_default and ok_edges
    print(
        "check compared against full Repro.forward return value, including "
        "view, broadcast bool mask to 0/-inf, any/all-inf handling, stable "
        "softmax, zero fill, expand, and final view"
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
    bmm_10, expand_2, *_ = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=bmm_10.device,
        dtype=torch.float32,
    )

    mask_true = int(expand_2[0, 0, :, 0].sum().item())
    full_score_bytes = N_ROWS * K_LEN * 4
    output_bytes = N_ROWS * K_LEN * 4
    logical_bytes = full_score_bytes + output_bytes + BATCH * Q_LEN * K_LEN
    effective_score_read_bytes = BATCH * HEADS * mask_true * K_LEN * 4
    effective_bytes = effective_score_read_bytes + output_bytes + Q_LEN

    print(
        "oracle shape: "
        f"bmm_10=f32{BMM_SHAPE} stride={tuple(bmm_10.stride())} "
        f"expand_2=bool{MASK_SHAPE} stride={tuple(expand_2.stride())} "
        f"out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"rows={N_ROWS} block_m={block_m} block_k={block_k} num_warps={num_warps} "
        f"mask_true_queries={mask_true}/{Q_LEN} "
        f"logical_bytes={logical_bytes / 1e6:.1f} MB "
        f"estimated_effective_bytes={effective_bytes / 1e6:.1f} MB"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(
                bmm_10,
                expand_2,
                out,
                block_m=block_m,
                block_k=block_k,
                num_warps=num_warps,
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
        "classification": "BANDWIDTH_BOUND",
        "logical_bytes": logical_bytes,
        "estimated_effective_bytes": effective_bytes,
    }
    print(json.dumps(result))
    print(f"oracle full-scope broadcast-mask softmax: {oracle_us:.3f} us")
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
    parser.add_argument("--num-warps", type=int, default=4, help="Triton warps per program")
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
        rtol=args.rtol,
        atol=args.atol,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(
            block_m=args.block_m,
            block_k=block_k,
            num_warps=args.num_warps,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    with torch.no_grad():
        main()
