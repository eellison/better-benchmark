"""
Oracle for var_mean_mean_626fb0807512

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full captured DenseNet batchnorm-training, ReLU, and spatial-mean scope by reducing stats directly from the two cat inputs, updating the two running-stat copy_ outputs in place, and producing the returned pooled [128,184] tensor without materializing either the concatenated [128,184,4,4] tensor or the full BN/ReLU activation, whereas Inductor currently treats the cat, var_mean training-BN template, running-stat copy_ side outputs, ReLU, and downstream spatial mean as separately scheduled producer/consumer regions with an intermediate full activation; Inductor cannot do this today because its scheduler cannot fuse a normalization template with a following reduction consumer while also preserving mutable running-stat side effects; the fix is SCHEDULER_FUSION: add a BN-training producer schedule that can expose mean/var side outputs, honor copy_ mutation, and sink cheap normalization/ReLU work into the immediately following spatial reduction.
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
except ImportError:  # pragma: no cover - keeps static checks importable.
    triton = None
    tl = None

from oracle_harness import (
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "SCHEDULER_FUSION"

N = 128
C0 = 16
C1 = 168
C = 184
H = 4
W = 4
HW = H * W
K = N * HW
TOTAL_OUT = N * C
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0004885197850513
STATS_BLOCK = 2048
OUT_BLOCK = 128

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


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _fused_stats_update_pool_kernel(
        x0_ptr,
        x1_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        c0: tl.constexpr,
        c1: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elems_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N)
        hw_offsets = tl.arange(0, BLOCK_HW)
        n_matrix = n_offsets[:, None]
        hw_matrix = hw_offsets[None, :]

        from_x0 = channel < c0
        from_x1 = channel >= c0
        x0_flat = n_matrix * c0 * hw_size + channel * hw_size + hw_matrix
        x1_channel = channel - c0
        x1_flat = n_matrix * c1 * hw_size + x1_channel * hw_size + hw_matrix
        vals0 = tl.load(x0_ptr + x0_flat, mask=from_x0, other=0.0).to(tl.float32)
        vals1 = tl.load(x1_ptr + x1_flat, mask=from_x1, other=0.0).to(tl.float32)
        vals = vals0 + vals1

        row_sum = tl.sum(vals, axis=1)
        row_sum2 = tl.sum(vals * vals, axis=1)
        sum_x = tl.sum(row_sum, axis=0)
        sum_x2 = tl.sum(row_sum2, axis=0)
        mean = sum_x / elems_per_channel
        var = sum_x2 / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        y = (vals - mean) * invstd
        y = tl.maximum(y * weight + bias, 0.0)
        pooled = tl.sum(y, axis=1) * 0.0625
        tl.store(out_ptr + n_offsets * channels + channel, pooled, mask=n_offsets < BLOCK_N)

    @triton.jit
    def _stats_update_kernel(
        x0_ptr,
        x1_ptr,
        running_mean_ptr,
        running_var_ptr,
        mean_ptr,
        var_ptr,
        c0: tl.constexpr,
        c1: tl.constexpr,
        hw_size: tl.constexpr,
        elems_per_channel: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size

        from_x0 = channel < c0
        x0_flat = n_idx * c0 * hw_size + channel * hw_size + hw_idx
        x1_channel = channel - c0
        x1_flat = n_idx * c1 * hw_size + x1_channel * hw_size + hw_idx
        vals = tl.load(
            x0_ptr + x0_flat,
            mask=from_x0,
            other=0.0,
        ).to(tl.float32)
        vals = tl.where(
            from_x0,
            vals,
            tl.load(x1_ptr + x1_flat).to(tl.float32),
        )

        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
        mean = sum_x / elems_per_channel
        var = sum_x2 / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )
        tl.store(mean_ptr + channel, mean)
        tl.store(var_ptr + channel, var)

    @triton.jit
    def _pooled_bn_relu_kernel(
        x0_ptr,
        x1_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        var_ptr,
        out_ptr,
        c0: tl.constexpr,
        c1: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        total_out: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_out
        channel = offsets % channels
        n_idx = offsets // channels

        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.rsqrt(var + eps)

        acc = tl.zeros((BLOCK,), tl.float32)
        for hw in tl.static_range(0, hw_size):
            from_x0 = channel < c0
            x0_flat = n_idx * c0 * hw_size + channel * hw_size + hw
            x1_channel = channel - c0
            x1_flat = n_idx * c1 * hw_size + x1_channel * hw_size + hw
            x = tl.load(
                x0_ptr + x0_flat,
                mask=mask & from_x0,
                other=0.0,
            ).to(tl.float32)
            x = tl.where(
                from_x0,
                x,
                tl.load(x1_ptr + x1_flat, mask=mask, other=0.0).to(tl.float32),
            )
            y = (x - mean) * invstd
            y = y * weight + bias
            acc += tl.maximum(y, 0.0)

        tl.store(out_ptr + offsets, acc * 0.0625, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")
    x0, x1, running_mean, running_var, weight, bias, shape_param = inputs
    tensors = (x0, x1, running_mean, running_var, weight, bias)
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        raise TypeError("first six repro inputs must be tensors")
    expected = [(N, C0, H, W), (N, C1, H, W), (C,), (C,), (C,), (C,)]
    for index, (tensor, shape) in enumerate(zip(tensors, expected)):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"input {index} shape {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {tensor.dtype} != torch.float32")
        if not tensor.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not tensor.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")
    if tuple(shape_param) != (N, C):
        raise ValueError(f"shape parameter {shape_param!r} != {(N, C)}")
    return tensors  # type: ignore[return-value]


def _make_buffers(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "mean": torch.empty((C,), device=device, dtype=torch.float32),
        "var": torch.empty((C,), device=device, dtype=torch.float32),
        "out": torch.empty((N, C), device=device, dtype=torch.float32),
    }


def _launch_oracle(
    inputs: tuple[Any, ...] | list[Any],
    buffers: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    x0, x1, running_mean, running_var, weight, bias = _validate_inputs(inputs)

    _fused_stats_update_pool_kernel[(C,)](
        x0,
        x1,
        running_mean,
        running_var,
        weight,
        bias,
        buffers["out"],
        c0=C0,
        c1=C1,
        channels=C,
        hw_size=HW,
        elems_per_channel=K,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N=N,
        BLOCK_HW=HW,
        num_warps=8,
    )
    return buffers["out"], running_mean, running_var


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward scope and return all three outputs.

    The running mean and variance inputs are intentionally mutated in place to
    match the two aten.copy_ return values in eager.
    """
    tensors = _validate_inputs(inputs)
    buffers = _make_buffers(tensors[0].device)
    return _launch_oracle(inputs, buffers)


def _clone_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return tuple(cloned)


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                result.extend(_normalize_outputs(item))
        return result
    return []


def run_check(*, rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    base_inputs = get_inputs()
    eager_inputs = _clone_inputs(base_inputs)
    oracle_inputs = _clone_inputs(base_inputs)
    instance = get_repro_instance()

    with torch.no_grad():
        eager_out = instance(*eager_inputs)
        oracle_out = oracle_forward(oracle_inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager_out)
    oracle_list = _normalize_outputs(oracle_out)
    if len(eager_list) != len(oracle_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    ok_all = True
    for index, (eager, oracle) in enumerate(zip(eager_list, oracle_list)):
        if eager.shape != oracle.shape:
            print(
                f"  output {index}: SCOPE_MISMATCH shape oracle={list(oracle.shape)} "
                f"eager={list(eager.shape)}"
            )
            ok_all = False
            continue
        if eager.dtype != oracle.dtype:
            print(f"  output {index}: SCOPE_MISMATCH dtype oracle={oracle.dtype} eager={eager.dtype}")
            ok_all = False
            continue
        if eager.stride() != oracle.stride():
            print(
                f"  output {index}: SCOPE_MISMATCH stride oracle={oracle.stride()} "
                f"eager={eager.stride()}"
            )
            ok_all = False
            continue

        if eager.is_floating_point():
            max_diff = (eager.float() - oracle.float()).abs().max().item()
            values_ok = torch.allclose(eager.float(), oracle.float(), rtol=rtol, atol=atol)
            print(
                f"  output {index}: {'PASS' if values_ok else 'FAIL'} "
                f"(shape={list(eager.shape)} dtype={eager.dtype} stride={eager.stride()} "
                f"max_diff={max_diff:.2e})"
            )
        else:
            values_ok = torch.equal(eager, oracle)
            print(
                f"  output {index}: {'PASS' if values_ok else 'FAIL'} "
                f"(exact, shape={list(eager.shape)} dtype={eager.dtype})"
            )
        ok_all = ok_all and bool(values_ok)

    print("check compared full Repro.forward scope, including cat inputs, var_mean, affine, ReLU, spatial mean, and both in-place running-stat copy_ outputs")
    return bool(ok_all)


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
    best_us = math.inf
    for _ in range(rep):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best_us = min(best_us, start.elapsed_time(end) * 1000.0)
    return best_us


def _compile_with_config(inputs: tuple[Any, ...], config: dict[str, object], warmup: int) -> Any:
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    model = get_repro_instance()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(max(1, warmup)):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(*, warmup: int, rep: int, no_compile: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    oracle_inputs = _clone_inputs(get_inputs())
    buffers = _make_buffers(oracle_inputs[0].device)
    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(oracle_inputs, buffers),
            warmup=warmup,
            rep=rep,
        )

    compile_results: dict[str, float] = {}
    if not no_compile:
        for label, config in COMPILE_CONFIGS:
            try:
                compile_inputs = _clone_inputs(get_inputs())
                compiled = _compile_with_config(compile_inputs, config, warmup)
                holder: list[Any] = [None]
                with torch.no_grad():
                    compile_results[label] = _bench_cuda_graph(
                        lambda compiled=compiled, inputs=compile_inputs: holder.__setitem__(
                            0, compiled(*inputs)
                        ),
                        warmup=warmup,
                        rep=rep,
                    )
            except Exception as exc:  # pragma: no cover - diagnostic path.
                print(f"torch.compile {label}: FAILED ({exc})")
            finally:
                torch.cuda.synchronize()

    compile_us = min(compile_results.values()) if compile_results else math.nan
    ratio = compile_us / oracle_us if compile_results and oracle_us > 0 else math.nan
    true_floor = bool(compile_results and all(oracle_us < value for value in compile_results.values()))
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_us, 3) if compile_results else None,
        "compile_configs_us": {key: round(value, 3) for key, value in compile_results.items()},
        "ratio": round(ratio, 3) if ratio == ratio else None,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "true_floor": true_floor,
        "classification": CLASSIFICATION,
    }
    print(json.dumps(result))
    print(f"oracle full-scope BN/ReLU/spatial-mean: {oracle_us:.3f} us")
    for label, value in compile_results.items():
        print(f"torch.compile {label}: {value:.3f} us")
    print(f"true floor: {'yes' if true_floor else 'no'}")
    if not true_floor:
        print("diagnosis_only: oracle did not beat every required compile config")
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
    parser.add_argument("--no-compile", action="store_true", help="Skip torch.compile baselines")
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Accepted for template CLI compatibility")
    parser.add_argument("--all-shapes", action="store_true", help="Accepted for template CLI compatibility")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if args.all_shapes:
        print("NOTE: shapes.txt has one shape for this repro; running the default shape")

    if not args.check and not args.bench:
        args.check = args.bench = True

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; no outputs are skipped by this oracle check")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = run_check(rtol=args.rtol, atol=args.atol)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(warmup=args.warmup, rep=args.rep, no_compile=args.no_compile)


if __name__ == "__main__":
    main()
