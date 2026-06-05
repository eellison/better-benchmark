"""Full-scope Triton oracle for pointwise_f5d9e703009c.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete attention Q layout conversion returned by Repro.forward as one contiguous-storage Triton cast/scale pass while preserving the final non-contiguous f32[12,512,64] output layout, whereas tuned Inductor is expected to emit the same single materialization kernel; remaining differences are dominated by launch and memory traffic for reading fp16 and writing fp32, so the fix is BANDWIDTH_BOUND: treat this as an at-floor layout materialization unless tuned compile timing falls outside noise.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None

from oracle_harness import (  # noqa: E402
    _gpu_exclusive_lock,
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

HEAD_DIM = 64
SCALE = 0.3535533905932738
CLASSIFICATION = "BANDWIDTH_BOUND"

COMPILE_CONFIGS: tuple[tuple[str, dict[str, Any]], ...] = (
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
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _validate_and_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int], int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    addmm_67, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(addmm_67, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not addmm_67.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if addmm_67.dtype is not torch.float16:
        raise ValueError(f"{REPRO_ID} expects torch.float16 input, got {addmm_67.dtype}")
    if addmm_67.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(addmm_67.shape)}")
    if not addmm_67.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={addmm_67.stride()}")

    seq, hidden = (int(addmm_67.shape[0]), int(addmm_67.shape[1]))
    if hidden % HEAD_DIM != 0:
        raise ValueError(f"hidden size {hidden} is not divisible by {HEAD_DIM}")
    heads = hidden // HEAD_DIM

    shape0_t = _shape_tuple(shape0)
    shape1_t = _shape_tuple(shape1)
    shape2_t = _shape_tuple(shape2)
    out_shape = _shape_tuple(shape3)

    if shape0_t != (1, seq, hidden):
        raise ValueError(f"unexpected first view shape {shape0_t}")
    if shape1_t != (1, seq, -1, HEAD_DIM) and shape1_t != (1, seq, heads, HEAD_DIM):
        raise ValueError(f"unexpected head view shape {shape1_t}")
    if shape2_t != (1, heads, seq, HEAD_DIM):
        raise ValueError(f"unexpected expand shape {shape2_t}")
    if out_shape != (heads, seq, HEAD_DIM):
        raise ValueError(f"unexpected output shape {out_shape}")

    out_stride = (HEAD_DIM, hidden, 1)
    return addmm_67, out_shape, out_stride, seq * hidden


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _cast_scale_storage_kernel(
        in_ptr,
        out_ptr,
        N: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        values = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, values * scale, mask=mask)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward scope with one Triton pointwise kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    addmm_67, out_shape, out_stride, numel = _validate_and_layout(inputs)
    out = torch.empty_strided(
        out_shape,
        out_stride,
        device=addmm_67.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(numel, meta["BLOCK_SIZE"]),)
    _cast_scale_storage_kernel[grid](
        addmm_67,
        out,
        N=numel,
        scale=SCALE,
    )
    return out


def _check_layout(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == tuple(eager_out.shape)
        and oracle_out.dtype == eager_out.dtype == torch.float32
        and tuple(oracle_out.stride()) == tuple(eager_out.stride())
        and oracle_out.storage_offset() == eager_out.storage_offset() == 0
    )
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} "
        f"dtype={oracle_out.dtype} storage_offset={oracle_out.storage_offset()})"
    )
    return bool(layout_ok)


def _save_config_state(config: dict[str, Any]) -> dict[str, Any]:
    import torch._inductor.config as cfg

    saved: dict[str, Any] = {}
    for key in config:
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        saved[key] = getattr(obj, parts[-1])
    return saved


def _restore_config_state(saved: dict[str, Any]) -> None:
    import torch._inductor.config as cfg

    for key, value in saved.items():
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def _apply_config(config: dict[str, Any]) -> None:
    import torch._inductor.config as cfg

    for key, value in config.items():
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


def _compile_with_config(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    config: dict[str, Any],
) -> Any:
    import torch._dynamo

    saved = _save_config_state(config)
    try:
        _apply_config(config)
        torch._dynamo.reset()
        compiled = torch.compile(instance)
        with torch.no_grad():
            for _ in range(5):
                compiled(*inputs)
            torch.cuda.synchronize()
        return compiled
    finally:
        _restore_config_state(saved)


def _try_capture(fn: Any) -> torch.cuda.CUDAGraph | None:
    try:
        with torch.no_grad():
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
        return graph
    except Exception:
        torch.cuda.synchronize()
        return None


def _time_graph_or_fn(
    graph: torch.cuda.CUDAGraph | None,
    fn: Any,
    *,
    warmup: int,
    rep: int,
) -> float:
    from triton.testing import do_bench

    if graph is not None:
        return do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="min") * 1000.0
    return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def run_bench(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    warmup: int,
    rep: int,
    rounds: int = 5,
) -> dict[str, Any]:
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    _validate_and_layout(inputs)
    timed_items: list[tuple[str, torch.cuda.CUDAGraph | None, Any]] = []

    with torch.no_grad():
        for _ in range(3):
            oracle_forward(inputs)
        torch.cuda.synchronize()
    oracle_graph = _try_capture(lambda: oracle_forward(inputs))
    timed_items.append(("oracle", oracle_graph, lambda: oracle_forward(inputs)))

    for label, config in COMPILE_CONFIGS:
        compiled = _compile_with_config(instance, inputs, config)
        graph = _try_capture(lambda compiled=compiled: compiled(*inputs))
        timed_items.append((label, graph, lambda compiled=compiled: compiled(*inputs)))

    quick_times = [
        _time_graph_or_fn(graph, fn, warmup=5, rep=10)
        for _, graph, fn in timed_items
    ]
    est_us = min(quick_times)
    if est_us < 50:
        rep = max(rep, 500)
    elif est_us < 200:
        rep = max(rep, 300)

    best_times = {label: math.inf for label, _, _ in timed_items}
    with _gpu_exclusive_lock(REPRO_ID):
        for _ in range(warmup):
            for _, graph, fn in timed_items:
                if graph is not None:
                    graph.replay()
                else:
                    with torch.no_grad():
                        fn()
        torch.cuda.synchronize()

        for _ in range(rounds):
            for label, graph, fn in timed_items:
                us = _time_graph_or_fn(graph, fn, warmup=warmup, rep=rep)
                best_times[label] = min(best_times[label], us)

    oracle_us = best_times["oracle"]
    compile_results = [
        {"label": label, "us": round(best_times[label], 3)}
        for label, _ in COMPILE_CONFIGS
    ]
    compile_us = min(best_times[label] for label, _ in COMPILE_CONFIGS)
    ratio = compile_us / oracle_us if oracle_us > 0 else 0.0
    if ratio > 1.05:
        status = "GOOD"
    elif ratio < 0.95:
        status = "BAD_ORACLE"
    else:
        status = "AT_FLOOR"

    result: dict[str, Any] = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_us, 3),
        "ratio": round(ratio, 3),
        "status": status,
        "classification": CLASSIFICATION,
        "compile_results": compile_results,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
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
        ok = bool(ok and _check_layout(instance, inputs))
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = run_bench(instance, inputs, warmup=args.warmup, rep=args.rep)
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
