"""
Oracle for var_mean_bf21b62ff8f8

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full captured BART residual-layernorm scope in one shape-specialized Triton row kernel, including the `[512,768] -> [1,512,768]` view, fp16-rounded residual add, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 normalization, affine scale/bias, final fp16 cast, and contiguous `[1,512,768]` output, whereas tuned Inductor already lands in the same launch and memory-traffic envelope; Inductor cannot materially improve this repro through local fusion or algebraic rewrites because the remaining work is the required residual/input/affine reads, fixed-width row reduction, rsqrt, output write, and launch overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor oracle unless broader launch-overhead or normalization-template improvements move the baseline.
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
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
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

ROWS = 512
HIDDEN = 768
OUTPUT_SHAPE = (1, ROWS, HIDDEN)
OUTPUT_STRIDE = (ROWS * HIDDEN, HIDDEN, 1)
EPS = 1.0e-5
CLASSIFICATION = "BANDWIDTH_BOUND"

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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=4, num_stages=3),
            triton.Config({}, num_warps=8, num_stages=3),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        x = (addmm + residual).to(tl.float16).to(tl.float32)
        x = tl.where(mask, x, 0.0)

        mean = tl.sum(x, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_59, convert_element_type_69, weight, bias, shape_param = inputs
    tensor_inputs = (addmm_59, convert_element_type_69, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = ((ROWS, HIDDEN), OUTPUT_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if tuple(int(dim) for dim in shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    return addmm_59, convert_element_type_69, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward residual-layernorm computation.

    SCOPE INVARIANT: accepts the same five inputs as Repro.forward() and returns
    the same single fp16 `[1,512,768]` contiguous output. The first view is
    metadata-only for the captured layout, so the kernel reads both activation
    tensors as flattened row-major `[512,768]` rows.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_layernorm.py")

    addmm_59, residual, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm_59.device,
        dtype=torch.float16,
    )
    _residual_layernorm_kernel[(ROWS,)](
        addmm_59,
        residual,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=triton.next_power_of_2(HIDDEN),
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()

    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    return ok


def _do_bench(fn: Any, *, warmup: int, rep: int) -> float:
    if triton is None:
        raise RuntimeError("Triton is required for benchmarking")
    from triton.testing import do_bench

    return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0


def _bench_cuda_graph(fn: Any, *, warmup: int, rep: int) -> float:
    with torch.no_grad():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            return _do_bench(lambda: graph.replay(), warmup=warmup, rep=rep)
        except Exception as exc:
            print(f"WARNING: CUDA graph capture failed; falling back to direct timing ({exc})")
            torch.cuda.synchronize()
            return _do_bench(fn, warmup=warmup, rep=rep)


def _bench_compile_config(
    label: str,
    config: dict[str, object],
    inputs: list[Any],
    *,
    warmup: int,
    rep: int,
) -> float:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    instance = get_repro_instance()
    holder: list[Any] = [None]
    with inductor_config.patch(config):
        compiled = torch.compile(instance)
        with torch.no_grad():
            for _ in range(5):
                holder[0] = compiled(*inputs)
            torch.cuda.synchronize()
            compile_us = _bench_cuda_graph(
                lambda: holder.__setitem__(0, compiled(*inputs)),
                warmup=warmup,
                rep=rep,
            )
    torch.cuda.synchronize()
    print(f"torch.compile {label}: {compile_us:.3f} us")
    return compile_us


def run_bench(*, warmup: int, rep: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    inputs = get_inputs()
    holder: list[Any] = [None]
    with torch.no_grad():
        holder[0] = oracle_forward(inputs)
        torch.cuda.synchronize()
        oracle_us = _bench_cuda_graph(
            lambda: holder.__setitem__(0, oracle_forward(inputs)),
            warmup=warmup,
            rep=rep,
        )

    compile_results: dict[str, float] = {}
    compile_errors: dict[str, str] = {}
    for label, config in COMPILE_CONFIGS:
        try:
            compile_results[label] = _bench_compile_config(
                label,
                config,
                inputs,
                warmup=warmup,
                rep=rep,
            )
        except Exception as exc:
            message = str(exc).splitlines()[0]
            compile_errors[label] = message
            print(f"torch.compile {label}: FAILED ({message})")

    cd_compile_us = compile_results.get("coordinate_descent_tuning=True", math.nan)
    best_compile_us = min(compile_results.values()) if compile_results else math.nan
    ratio = best_compile_us / oracle_us if oracle_us > 0 and compile_results else math.nan
    if math.isnan(ratio):
        status = "COMPILE_FAILED"
    elif ratio > 1.05:
        status = "GOOD"
    elif ratio < 0.95:
        status = "BAD_ORACLE"
    else:
        status = "AT_FLOOR"
    if math.isnan(ratio):
        csv_status = "unknown"
    else:
        csv_status = "implemented_unmeasured" if ratio >= 1.10 else "at_floor"

    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(cd_compile_us, 3) if not math.isnan(cd_compile_us) else None,
        "best_compile_us": round(best_compile_us, 3) if not math.isnan(best_compile_us) else None,
        "compile_configs_us": {key: round(value, 3) for key, value in compile_results.items()},
        "compile_errors": compile_errors,
        "ratio": round(ratio, 3) if not math.isnan(ratio) else None,
        "status": status,
        "classification": CLASSIFICATION,
        "recommended_csv_status": csv_status,
    }
    print(json.dumps(result))
    print(f"oracle full-scope residual_layernorm: {oracle_us:.3f} us")
    if compile_results:
        print(f"best required compile config: {best_compile_us:.3f} us")
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
