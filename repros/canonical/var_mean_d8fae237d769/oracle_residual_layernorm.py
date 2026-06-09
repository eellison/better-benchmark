"""
Oracle for var_mean_d8fae237d769

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured residual-add plus fp32 hidden-size-768 LayerNorm in one shape-specialized Triton row kernel, including the `[512, 768] -> [1, 512, 768]` view, fp16 residual add, fp32 `var_mean(correction=0, keepdim=True)` with `eps=1e-12`, affine scale/bias, fp16 cast, and final contiguous `[512, 768]` view, whereas tuned Inductor already emits the same fused normalization shape without avoidable intermediate materialization; Inductor cannot materially improve this repro today because the remaining cost is dominated by required memory traffic, row reduction, rsqrt, affine loads, and the output store; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless a broader normalization-template or launch-overhead improvement moves the baseline.
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
except ImportError:  # pragma: no cover - keeps import usable without Triton.
    triton = None
    tl = None

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

EPS = 1.0e-12
CLASSIFICATION = "BANDWIDTH_BOUND"

COMPILE_CONFIGS: list[tuple[str, dict[str, object]]] = [
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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        x = (addmm + residual).to(tl.float16).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _resolve_view_shape(shape: Any, numel: int) -> tuple[int, ...]:
    dims = list(shape)
    unknown_index = None
    known_product = 1
    for index, dim in enumerate(dims):
        dim = int(dim)
        dims[index] = dim
        if dim == -1:
            if unknown_index is not None:
                raise ValueError(f"view shape {shape!r} has more than one inferred dim")
            unknown_index = index
        else:
            known_product *= dim

    if unknown_index is not None:
        if known_product == 0 or numel % known_product != 0:
            raise ValueError(f"view shape {shape!r} is incompatible with {numel} elements")
        dims[unknown_index] = numel // known_product
    elif known_product != numel:
        raise ValueError(f"view shape {shape!r} is incompatible with {numel} elements")

    return tuple(dims)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], int, int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    addmm_70, residual, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (addmm_70, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first four repro inputs must be tensors")

    if addmm_70.ndim != 2:
        raise ValueError(f"addmm_70 must be rank-2, got shape {tuple(addmm_70.shape)}")
    if residual.ndim != 3:
        raise ValueError(f"residual must be rank-3, got shape {tuple(residual.shape)}")
    if weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("affine inputs must be rank-1 tensors")

    rows, hidden = (int(addmm_70.shape[0]), int(addmm_70.shape[1]))
    if tuple(residual.shape) != _resolve_view_shape(shape0, addmm_70.numel()):
        raise ValueError(f"residual shape {tuple(residual.shape)} does not match view shape {shape0!r}")
    if residual.numel() != addmm_70.numel():
        raise ValueError("addmm_70 and residual must have the same number of elements")
    if tuple(weight.shape) != (hidden,) or tuple(bias.shape) != (hidden,):
        raise ValueError(f"affine inputs must both have shape ({hidden},)")

    out_shape = _resolve_view_shape(shape1, addmm_70.numel())
    if math.prod(out_shape) != rows * hidden:
        raise ValueError(f"output view shape {shape1!r} is incompatible with addmm_70")

    for index, value in enumerate(tensor_inputs):
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    return addmm_70, residual, weight, bias, out_shape, rows, hidden


def _num_warps(hidden: int) -> int:
    return 8 if hidden >= 768 else 4


def _launch_oracle(
    addmm_70: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    out: torch.Tensor,
    rows: int,
    hidden: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_layernorm.py")
    if not out.is_cuda or out.dtype != torch.float16 or not out.is_contiguous():
        raise ValueError("output must be contiguous CUDA fp16")
    if out.numel() != rows * hidden:
        raise ValueError(f"output has {out.numel()} elements, expected {rows * hidden}")

    _residual_layernorm_kernel[(rows,)](
        addmm_70,
        residual,
        weight,
        bias,
        out,
        hidden=hidden,
        eps=EPS,
        BLOCK_N=triton.next_power_of_2(hidden),
        num_warps=_num_warps(hidden),
    )
    return out


@oracle_impl(hardware="H100", shapes="(T([512, 768], f16), T([1, 512, 768], f16), T([768], f16), T([768], f16), S([1, 512, 768]), S([512, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation with a Triton row kernel."""
    addmm_70, residual, weight, bias, out_shape, rows, hidden = _validate_inputs(inputs)
    out = torch.empty(out_shape, device=addmm_70.device, dtype=torch.float16)
    return _launch_oracle(addmm_70, residual, weight, bias, out, rows, hidden)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected_f32.abs().clamp_min(1.0e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


def run_check(rtol: float, atol: float) -> bool:
    if triton is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA and Triton are required for the oracle check")

    inputs = get_inputs()
    instance = get_repro_instance()

    with torch.no_grad():
        expected = _as_tuple(instance(*inputs))
        actual = _as_tuple(oracle_forward(inputs))
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"  SCOPE_MISMATCH: expected {len(expected)} outputs, got {len(actual)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for index, (got_item, ref_item) in enumerate(zip(actual, expected)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            ok = ok and bool(item_ok)
            print(f"  output {index}: {'PASS' if item_ok else 'FAIL'} non-tensor")
            continue

        shape_ok = got_item.shape == ref_item.shape
        dtype_ok = got_item.dtype == ref_item.dtype
        stride_ok = got_item.stride() == ref_item.stride()
        value_ok = torch.allclose(
            got_item.float(),
            ref_item.float(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        max_abs, max_rel = _max_diff(got_item, ref_item)
        item_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and bool(item_ok)
        print(
            f"  output {index}: {'PASS' if item_ok else 'FAIL'} "
            f"shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _bench_cuda_graph(fn: Any, warmup: int, rep: int) -> float:
    with torch.no_grad():
        for _ in range(max(1, warmup)):
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
    inputs: list[Any] | tuple[Any, ...],
    config: dict[str, object],
    warmup: int,
) -> Any:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = get_repro_instance().cuda()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        with torch.no_grad():
            for _ in range(max(1, warmup)):
                compiled(*inputs)
            torch.cuda.synchronize()
    return compiled


def run_bench(warmup: int, rep: int, no_compile: bool) -> dict[str, Any]:
    if triton is None or not torch.cuda.is_available():
        raise RuntimeError("CUDA and Triton are required for the oracle benchmark")

    inputs = get_inputs()
    addmm_70, residual, weight, bias, out_shape, rows, hidden = _validate_inputs(inputs)
    out = torch.empty(out_shape, device=addmm_70.device, dtype=torch.float16)

    logical_bytes = rows * hidden * 10
    print(
        "oracle shape: "
        f"addmm_70=f16{list(addmm_70.shape)} residual=f16{list(residual.shape)} "
        f"weight=f16[{hidden}] bias=f16[{hidden}] out=f16{list(out_shape)}"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(addmm_70, residual, weight, bias, out, rows, hidden),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_bytes / (oracle_us * 1.0e-6) / 1.0e9
    print(f"oracle full-scope residual_layernorm: {oracle_us:.3f} us ({oracle_bw:.3f} GB/s logical bytes)")
    print(f"oracle_us={oracle_us:.3f}")

    compile_times: list[tuple[str, float]] = []
    if not no_compile:
        print("CUDA graph replay timings cover the same repro.py residual add, var_mean, affine, cast, and final view")
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(inputs, config, warmup)
                us = _bench_cuda_graph(lambda: compiled(*inputs), warmup=warmup, rep=rep)
                compile_times.append((label, us))
                print(f"torch.compile {label}: {us:.3f} us")
            except Exception as exc:
                print(f"torch.compile {label}: FAILED ({exc})")

    best_compile = min((us for _, us in compile_times), default=float("nan"))
    ratio = best_compile / oracle_us if compile_times and oracle_us > 0 else float("nan")
    status = (
        "GOOD"
        if compile_times and ratio > 1.05
        else "AT_FLOOR"
        if compile_times and ratio >= 0.95
        else "BAD_ORACLE"
        if compile_times
        else "NO_COMPILE"
    )
    csv_status = "unknown" if not compile_times else ("implemented_unmeasured" if ratio >= 1.10 else "at_floor")
    result: dict[str, Any] = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(best_compile, 3) if compile_times else None,
        "ratio": round(ratio, 3) if compile_times else None,
        "status": status,
        "classification": CLASSIFICATION,
        "recommended_csv_status": csv_status,
        "compile_results": [
            {"label": label, "us": round(us, 3)} for label, us in compile_times
        ],
    }
    print(json.dumps(result, sort_keys=True))
    if compile_times:
        print(f"recommended CSV status: {csv_status}")
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
