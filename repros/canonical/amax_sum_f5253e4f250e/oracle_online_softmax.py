"""Full-scope Triton oracle for amax_sum_f5253e4f250e.

Gap diagnosis:
  Classification: NEW_PATTERN
  What oracle does differently: Computes the complete T5 attention
    bias-add, last-dimension softmax, Inductor RNG dropout, dropout scale, and
    final permuted output stride in one shape-specialized Triton row-softmax
    kernel.
  What Inductor change would fix: Add or extend an attention softmax/dropout
    lowering that recognizes broadcast additive bias and fuses RNG dropout plus
    the layout-only permute into the row-softmax schedule.
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 12
Q_LEN = 1024
K_LEN = 1024
BH = BATCH * HEADS
N_ROWS = BH * Q_LEN
BMM_SHAPE = (BH, Q_LEN, K_LEN)
WHERE_SHAPE = (BATCH, 1, Q_LEN, K_LEN)
ADD_SHAPE = (BATCH, HEADS, Q_LEN, K_LEN)
OUT_SHAPE = BMM_SHAPE
OUT_STRIDE = (Q_LEN * K_LEN, 1, K_LEN)
DROPOUT_P = 0.1
DROPOUT_SCALE = 1.0 / (1.0 - DROPOUT_P)
SEED_INDEX = 53
CLASSIFICATION = "NEW_PATTERN"
HISTORICAL_BEST_COMPILE_US = 484.44798588752747

CD_CONFIG_LABEL = "coordinate_descent_tuning=True"
COMBO_CONFIG_LABEL = (
    "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
    "coordinate_descent_tuning=True,benchmark_combo_kernel=True,"
    "triton.multi_kernel=3"
)
COMPILE_CONFIGS = [
    (CD_CONFIG_LABEL, {"coordinate_descent_tuning": True}),
    (
        COMBO_CONFIG_LABEL,
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
def _broadcast_bias_softmax_dropout_kernel(
    bmm_ptr,
    where_ptr,
    seeds_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < 98304
    col_mask = cols < 1024
    mask = row_mask[:, None] & col_mask[None, :]

    bh = rows // 1024
    batch = bh // 12
    q = rows - bh * 1024

    offsets = rows[:, None] * 1024 + cols[None, :]
    where_offsets = batch[:, None] * (1024 * 1024) + q[:, None] * 1024 + cols[None, :]

    bmm_vals = tl.load(bmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    where_vals = tl.load(where_ptr + where_offsets, mask=mask, other=0.0).to(tl.float32)
    scores = tl.where(mask, bmm_vals + where_vals, -float("inf"))

    row_max = tl.max(scores, axis=1)
    numer = tl.exp(scores - row_max[:, None])
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)

    seed = tl.load(seeds_ptr + 53)
    random_vals = tl.rand(seed, offsets.to(tl.uint32))
    keep = random_vals > 0.1
    out_vals = tl.where(keep, numer / denom[:, None] * 1.1111111111111112, 0.0)

    # Store into the physical storage underlying the final permuted output.
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _make_inputs(seed: int) -> tuple[Any, ...]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return get_inputs()


def _validate_shape_param(name: str, actual: object, expected: tuple[int, ...]) -> None:
    actual_tuple = tuple(int(dim) for dim in actual)
    if actual_tuple != expected:
        raise ValueError(f"{name} mismatch: expected {expected}, got {actual_tuple}")


def _validate_inputs(
    bmm_26: torch.Tensor,
    where: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
) -> None:
    if not (bmm_26.is_cuda and where.is_cuda and inductor_seeds.is_cuda):
        raise RuntimeError("CUDA tensors are required")
    if bmm_26.dtype != torch.float32 or where.dtype != torch.float32:
        raise TypeError(f"expected fp32 score inputs, got {bmm_26.dtype} and {where.dtype}")
    if inductor_seeds.dtype != torch.int64:
        raise TypeError(f"expected int64 inductor_seeds, got {inductor_seeds.dtype}")
    if tuple(bmm_26.shape) != BMM_SHAPE:
        raise ValueError(f"unexpected bmm_26 shape: {tuple(bmm_26.shape)}")
    if tuple(where.shape) != WHERE_SHAPE:
        raise ValueError(f"unexpected where shape: {tuple(where.shape)}")
    if tuple(inductor_seeds.shape) != (124,):
        raise ValueError(f"unexpected inductor_seeds shape: {tuple(inductor_seeds.shape)}")
    if tuple(bmm_26.stride()) != (Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected bmm_26 stride: {tuple(bmm_26.stride())}")
    if tuple(where.stride()) != (Q_LEN * K_LEN, Q_LEN * K_LEN, K_LEN, 1):
        raise ValueError(f"unexpected where stride: {tuple(where.stride())}")
    _validate_shape_param("_shape_param_0", _shape_param_0, ADD_SHAPE)
    _validate_shape_param("_shape_param_1", _shape_param_1, ADD_SHAPE)
    _validate_shape_param("_shape_param_2", _shape_param_2, OUT_SHAPE)


def _make_output(device: torch.device) -> torch.Tensor:
    return torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.float32)


def _launch_oracle(
    bmm_26: torch.Tensor,
    where: torch.Tensor,
    inductor_seeds: torch.Tensor,
    out: torch.Tensor,
    *,
    block_m: int,
    num_warps: int,
) -> torch.Tensor:
    if block_m <= 0 or block_m & (block_m - 1):
        raise ValueError(f"block_m must be a positive power of two, got {block_m}")
    if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
        raise ValueError(f"output must have shape {OUT_SHAPE} and stride {OUT_STRIDE}")
    if out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("output must be CUDA fp32")

    _broadcast_bias_softmax_dropout_kernel[(triton.cdiv(N_ROWS, block_m),)](
        bmm_26,
        where,
        inductor_seeds,
        out,
        BLOCK_M=block_m,
        BLOCK_K=K_LEN,
        num_warps=num_warps,
    )
    return out


def oracle_online_softmax(
    bmm_26: torch.Tensor,
    where: torch.Tensor,
    inductor_seeds: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    *,
    block_m: int = 2,
    num_warps: int = 4,
) -> torch.Tensor:
    _validate_inputs(
        bmm_26,
        where,
        inductor_seeds,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    )
    out = _make_output(bmm_26.device)
    return _launch_oracle(
        bmm_26,
        where,
        inductor_seeds,
        out,
        block_m=block_m,
        num_warps=num_warps,
    )


@oracle_impl(hardware="H100", shapes="(T([96, 1024, 1024], f32), T([8, 1, 1024, 1024], f32), T([124], i64), S([8, 12, 1024, 1024]), S([8, 12, 1024, 1024]), S([96, 1024, 1024]))")
def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    """Run the full compiled Repro.forward scope on the exact input tuple."""
    return oracle_online_softmax(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = float(torch.nan_to_num(diff, nan=0.0, posinf=math.inf).max().item())
    rel = diff / expected_f32.abs().clamp_min(1e-8)
    max_rel = float(torch.nan_to_num(rel, nan=0.0, posinf=math.inf).max().item())
    return max_abs, max_rel


def _bench_cuda_graph(fn: object, warmup: int, rep: int) -> tuple[float, float, list[float]]:
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
    return times[0], times[len(times) // 2], times


def _compile_with_config(
    module: Any,
    inputs: tuple[Any, ...],
    config: dict[str, object],
    warmup: int,
) -> object:
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


def run_check(block_m: int, num_warps: int, rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(seed=1234)
    _validate_inputs(*inputs)
    # Eager prims.inductor_random uses a separate fallback RNG path. Compare
    # against the compiled full repro so the dropout mask uses the same
    # tl.rand(seed, flat_offset) semantics as the timed oracle.
    compiled = _compile_with_config(module, inputs, {}, warmup=1)

    with torch.no_grad():
        expected = _as_tuple(compiled(*inputs))
        actual = _as_tuple(
            oracle_online_softmax(*inputs, block_m=block_m, num_warps=num_warps)
        )
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"  SCOPE_MISMATCH: oracle produces {len(actual)} outputs, expected {len(expected)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (got_item, ref_item) in enumerate(zip(actual, expected)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            print(f"  output {idx}: {'PASS' if item_ok else 'FAIL'} (non-tensor)")
            ok = ok and bool(item_ok)
            continue

        metadata_ok = (
            got_item.shape == ref_item.shape
            and got_item.dtype == ref_item.dtype
            and got_item.stride() == ref_item.stride()
        )
        value_ok = torch.allclose(
            got_item.float(),
            ref_item.float(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        max_abs, max_rel = _max_diff(got_item, ref_item)
        item_ok = metadata_ok and value_ok
        ok = ok and bool(item_ok)
        print(
            f"  output {idx}: {'PASS' if item_ok else 'FAIL'} "
            f"(shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} ref_stride={list(ref_item.stride())} "
            f"max_diff={max_abs:.2e} max_rel={max_rel:.2e} "
            f"allclose={value_ok} metadata={metadata_ok})"
        )

    print(
        "check compared against compiled full Repro.forward, including broadcast "
        "bias add, softmax, tl.rand dropout, scale, expand/view, and final permute"
    )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_oracle(
    inputs: tuple[Any, ...],
    *,
    block_m: int,
    num_warps: int,
    warmup: int,
    rep: int,
) -> tuple[float, float, list[float]]:
    bmm_26, where, inductor_seeds, *_ = inputs
    out = _make_output(bmm_26.device)
    return _bench_cuda_graph(
        lambda: _launch_oracle(
            bmm_26,
            where,
            inductor_seeds,
            out,
            block_m=block_m,
            num_warps=num_warps,
        ),
        warmup=warmup,
        rep=rep,
    )


def run_bench(
    block_m: int,
    num_warps: int,
    warmup: int,
    rep: int,
    no_compile: bool,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(seed=4321)
    _validate_inputs(*inputs)

    score_elems = N_ROWS * K_LEN
    logical_unique_bytes = (
        score_elems * 4  # bmm read
        + (BATCH * Q_LEN * K_LEN) * 4  # broadcast where read, counted once
        + score_elems * 4  # output write
    )
    logical_kernel_bytes = score_elems * 4 * 3
    print(
        "oracle shape: "
        f"bmm_26=f32{BMM_SHAPE} where=f32{WHERE_SHAPE} "
        f"out=f32{OUT_SHAPE} stride={OUT_STRIDE}"
    )
    print(
        "oracle tiling: "
        f"rows={N_ROWS} k={K_LEN} block_m={block_m} num_warps={num_warps} "
        f"logical_unique_bytes={logical_unique_bytes / 1e6:.1f} MB "
        f"kernel_read_write_bytes={logical_kernel_bytes / 1e6:.1f} MB"
    )

    with torch.no_grad():
        oracle_min_us, oracle_us, _ = _bench_oracle(
            inputs,
            block_m=block_m,
            num_warps=num_warps,
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_unique_bytes / (oracle_us * 1e-6) / 1e12
    print(
        "oracle full-scope online softmax/dropout/layout: "
        f"{oracle_us:.3f} us median, {oracle_min_us:.3f} us min "
        f"({oracle_bw:.3f} TB/s unique logical bytes)"
    )
    print(f"oracle_us={oracle_us:.3f}")

    compile_results: list[dict[str, object]] = []
    if not no_compile:
        print(
            "torch.compile timings cover the same repro.py view, broadcast add, "
            "softmax, Inductor RNG dropout, scale, expand/view, and final permute"
        )
        holder: list[Any] = [None]
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(module, inputs, config, warmup)
                min_us, median_us, _ = _bench_cuda_graph(
                    lambda: holder.__setitem__(0, compiled(*inputs)),
                    warmup=warmup,
                    rep=rep,
                )
                compile_results.append(
                    {"label": label, "us": median_us, "min_us": min_us}
                )
                print(f"torch.compile {label}: {median_us:.3f} us median, {min_us:.3f} us min")
            except Exception as exc:
                compile_results.append({"label": label, "error": str(exc)})
                print(f"torch.compile {label}: FAILED ({exc})")

    successful_compile_us = [
        float(result["us"]) for result in compile_results if "us" in result
    ]
    best_required_compile_us = min(successful_compile_us) if successful_compile_us else None
    required_compile_gate = (
        len(successful_compile_us) == len(COMPILE_CONFIGS)
        and best_required_compile_us is not None
        and oracle_us < best_required_compile_us
    )
    historical_gate = oracle_us < HISTORICAL_BEST_COMPILE_US
    true_floor = bool(required_compile_gate and historical_gate)

    if best_required_compile_us is not None:
        print(f"best_required_compile_us={best_required_compile_us:.3f}")
    print(f"historical_best_compile_us={HISTORICAL_BEST_COMPILE_US:.3f}")
    print(f"beats_required_compile={required_compile_gate}")
    print(f"beats_historical_best={historical_gate}")
    print(f"true_floor={true_floor}")
    if not true_floor:
        print("diagnosis_only: oracle did not beat both required configs and historical best")

    result = {
        "repro_id": REPRO_ID,
        "classification": CLASSIFICATION,
        "oracle_us": round(oracle_us, 3),
        "oracle_min_us": round(oracle_min_us, 3),
        "compile_results": compile_results,
        "best_required_compile_us": best_required_compile_us,
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "beats_required_compile": required_compile_gate,
        "beats_historical_best": historical_gate,
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "warmup": warmup,
        "rep": rep,
    }
    print(json.dumps(result, sort_keys=True))
    return result


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
