"""
Oracle for var_mean_var_mean_6d7a29cb97f1

Gap diagnosis:
  Classification: RECOMPUTE_FUSION
  What oracle does differently: Computes the full two-BN DenseNet stem region with shape-specialized Triton reductions, a fused first BN-affine-ReLU plus low-memory max-pool kernel, and a fused second BN epilogue.
  What Inductor change would fix: Add a recompute-fusion path that sinks cheap BN-affine-ReLU pointwise work into overlapping pooling stencils instead of materializing the producer tensor.
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
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (  # noqa: E402
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


CLASSIFICATION = "RECOMPUTE_FUSION"
HISTORICAL_BEST_COMPILE_US = 442.4000084400177

N = 64
C = 64
H = 112
W = 112
POOL_H = 56
POOL_W = 56
HW = H * W
POOL_HW = POOL_H * POOL_W
TOTAL_X = N * C * H * W
TOTAL_POOL = N * C * POOL_H * POOL_W
EPS = 1.0e-5
MOMENTUM = 0.1
CORRECTION_1 = 1.0000012456169853
CORRECTION_2 = 1.0000049824865598
STAT_BLOCK_1 = 4096
STAT_BLOCK_2 = 4096
STAT_BLOCKS_1 = 196
STAT_BLOCKS_2 = 49
FINAL_BLOCK_1 = 256
FINAL_BLOCK_2 = 64
POOL_BLOCK = 256
POINTWISE_BLOCK = 256

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sum_sq_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elems_per_channel: tl.constexpr,
        n_blocks: tl.constexpr,
        block_size: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        channel = tl.program_id(1)
        offsets = block_id * block_size + tl.arange(0, block_size)
        mask = offsets < elems_per_channel

        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        flat_idx = n_idx * channels * hw_size + channel * hw_size + hw_idx
        vals = tl.load(x_ptr + flat_idx, mask=mask, other=0.0).to(tl.float32)

        local_sum = tl.sum(vals, axis=0)
        local_sum_sq = tl.sum(vals * vals, axis=0)
        out_idx = channel * n_blocks + block_id
        tl.store(partial_sum_ptr + out_idx, local_sum)
        tl.store(partial_sum_sq_ptr + out_idx, local_sum_sq)

    @triton.jit
    def _finalize_stats_update_kernel(
        partial_sum_ptr,
        partial_sum_sq_ptr,
        mean_ptr,
        invstd_ptr,
        running_mean_ptr,
        running_var_ptr,
        elems_per_channel: tl.constexpr,
        n_blocks: tl.constexpr,
        correction: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        block_size: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < n_blocks
        base = channel * n_blocks + offsets

        sums = tl.load(partial_sum_ptr + base, mask=mask, other=0.0)
        sums_sq = tl.load(partial_sum_sq_ptr + base, mask=mask, other=0.0)
        total_sum = tl.sum(sums, axis=0)
        total_sum_sq = tl.sum(sums_sq, axis=0)

        mean = total_sum / elems_per_channel
        var = total_sum_sq / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel)
        old_var = tl.load(running_var_ptr + channel)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * correction * momentum,
        )
        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

    @triton.jit
    def _bn_relu_pool_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        pool_ptr,
        pool_idx_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        in_h: tl.constexpr,
        in_w: tl.constexpr,
        out_h: tl.constexpr,
        out_w: tl.constexpr,
        hw_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total

        ow = offsets % out_w
        tmp = offsets // out_w
        oh = tmp % out_h
        tmp = tmp // out_h
        channel = tmp % channels
        n_idx = tmp // channels

        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)
        base = n_idx * channels * hw_size + channel * hw_size

        max_val = tl.full((block_size,), -float("inf"), tl.float32)
        best_idx = tl.full((block_size,), 0, tl.int32)

        for kh in tl.static_range(0, 3):
            ih = oh * 2 + kh - 1
            valid_h = (ih >= 0) & (ih < in_h)
            for kw in tl.static_range(0, 3):
                iw = ow * 2 + kw - 1
                valid = mask & valid_h & (iw >= 0) & (iw < in_w)
                x_val = tl.load(x_ptr + base + ih * in_w + iw, mask=valid, other=0.0).to(tl.float32)
                y = (x_val - mean) * invstd
                y = y * weight + bias
                y = tl.maximum(y, 0.0)
                better = valid & (y > max_val)
                max_val = tl.where(better, y, max_val)
                best_idx = tl.where(better, kh * 3 + kw, best_idx)

        tl.store(pool_ptr + offsets, max_val, mask=mask)
        tl.store(pool_idx_ptr + offsets, best_idx.to(tl.int8), mask=mask)

    @triton.jit
    def _bn2_relu_center_kernel(
        pool_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        relu_out_ptr,
        centered_out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total
        channel = (offsets // hw_size) % channels

        x = tl.load(pool_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)

        centered = x - mean
        y = centered * invstd
        y = y * weight + bias
        y = tl.maximum(y, 0.0)

        tl.store(centered_out_ptr + offsets, centered, mask=mask)
        tl.store(relu_out_ptr + offsets, y, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 9:
        raise ValueError(f"expected 9 inputs, got {len(inputs)}")
    tensors = tuple(inputs)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("all repro inputs must be tensors")
    x = tensors[0]
    expected_shapes = [
        (N, C, H, W),
        (C,),
        (C,),
        (C,),
        (C,),
        (C,),
        (C,),
        (C,),
        (C,),
    ]
    for i, (tensor, shape) in enumerate(zip(tensors, expected_shapes)):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"input {i} shape {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != torch.float32:
            raise TypeError(f"input {i} dtype {tensor.dtype} != torch.float32")
        if not tensor.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not tensor.is_contiguous():
            raise ValueError(f"input {i} must be contiguous")
    return tensors  # type: ignore[return-value]


def _make_buffers(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "partial_sum1": torch.empty(C * STAT_BLOCKS_1, device=device, dtype=torch.float32),
        "partial_sq1": torch.empty(C * STAT_BLOCKS_1, device=device, dtype=torch.float32),
        "mean1": torch.empty(C, device=device, dtype=torch.float32),
        "invstd1": torch.empty(C, device=device, dtype=torch.float32),
        "pool": torch.empty((N, C, POOL_H, POOL_W), device=device, dtype=torch.float32),
        "pool_idx": torch.empty((N, C, POOL_H, POOL_W), device=device, dtype=torch.int8),
        "partial_sum2": torch.empty(C * STAT_BLOCKS_2, device=device, dtype=torch.float32),
        "partial_sq2": torch.empty(C * STAT_BLOCKS_2, device=device, dtype=torch.float32),
        "mean2": torch.empty(C, device=device, dtype=torch.float32),
        "invstd2": torch.empty(C, device=device, dtype=torch.float32),
        "relu2": torch.empty((N, C, POOL_H, POOL_W), device=device, dtype=torch.float32),
        "centered2": torch.empty((N, C, POOL_H, POOL_W), device=device, dtype=torch.float32),
    }


def _launch_oracle(
    inputs: tuple[Any, ...] | list[Any],
    buffers: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for the oracle")

    (
        convolution,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg8_1,
        arg9_1,
        arg10_1,
        arg11_1,
    ) = _validate_inputs(inputs)

    _partial_stats_kernel[(STAT_BLOCKS_1, C)](
        convolution,
        buffers["partial_sum1"],
        buffers["partial_sq1"],
        channels=C,
        hw_size=HW,
        elems_per_channel=N * HW,
        n_blocks=STAT_BLOCKS_1,
        block_size=STAT_BLOCK_1,
        num_warps=8,
    )
    _finalize_stats_update_kernel[(C,)](
        buffers["partial_sum1"],
        buffers["partial_sq1"],
        buffers["mean1"],
        buffers["invstd1"],
        arg3_1,
        arg4_1,
        elems_per_channel=N * HW,
        n_blocks=STAT_BLOCKS_1,
        correction=CORRECTION_1,
        eps=EPS,
        momentum=MOMENTUM,
        block_size=FINAL_BLOCK_1,
        num_warps=8,
    )
    _bn_relu_pool_kernel[(triton.cdiv(TOTAL_POOL, POOL_BLOCK),)](
        convolution,
        buffers["mean1"],
        buffers["invstd1"],
        arg5_1,
        arg6_1,
        buffers["pool"],
        buffers["pool_idx"],
        total=TOTAL_POOL,
        channels=C,
        in_h=H,
        in_w=W,
        out_h=POOL_H,
        out_w=POOL_W,
        hw_size=HW,
        block_size=POOL_BLOCK,
        num_warps=4,
    )
    _partial_stats_kernel[(STAT_BLOCKS_2, C)](
        buffers["pool"],
        buffers["partial_sum2"],
        buffers["partial_sq2"],
        channels=C,
        hw_size=POOL_HW,
        elems_per_channel=N * POOL_HW,
        n_blocks=STAT_BLOCKS_2,
        block_size=STAT_BLOCK_2,
        num_warps=8,
    )
    _finalize_stats_update_kernel[(C,)](
        buffers["partial_sum2"],
        buffers["partial_sq2"],
        buffers["mean2"],
        buffers["invstd2"],
        arg8_1,
        arg9_1,
        elems_per_channel=N * POOL_HW,
        n_blocks=STAT_BLOCKS_2,
        correction=CORRECTION_2,
        eps=EPS,
        momentum=MOMENTUM,
        block_size=FINAL_BLOCK_2,
        num_warps=4,
    )
    _bn2_relu_center_kernel[(triton.cdiv(TOTAL_POOL, POINTWISE_BLOCK),)](
        buffers["pool"],
        buffers["mean2"],
        buffers["invstd2"],
        arg10_1,
        arg11_1,
        buffers["relu2"],
        buffers["centered2"],
        total=TOTAL_POOL,
        channels=C,
        hw_size=POOL_HW,
        block_size=POINTWISE_BLOCK,
        num_warps=4,
    )

    return (
        buffers["pool_idx"],
        buffers["invstd2"],
        buffers["relu2"],
        buffers["centered2"],
        arg3_1,
        arg4_1,
        arg8_1,
        arg9_1,
    )


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns
    the same 8 outputs with matching shapes, dtypes, and strides. The running
    mean/variance inputs are updated in-place to mirror aten.copy_ outputs.
    """
    tensors = _validate_inputs(inputs)
    buffers = _make_buffers(tensors[0].device)
    return _launch_oracle(tensors, buffers)


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

    base_inputs = tuple(get_inputs())
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
        print("Correctness: FAIL")
        return False

    ok_all = True
    for i, (eager, oracle) in enumerate(zip(eager_list, oracle_list)):
        same_shape = eager.shape == oracle.shape
        same_dtype = eager.dtype == oracle.dtype
        same_stride = eager.stride() == oracle.stride()
        if not same_shape:
            print(f"  output {i}: SCOPE_MISMATCH shape oracle={list(oracle.shape)} eager={list(eager.shape)}")
            ok_all = False
            continue
        if not same_dtype:
            print(f"  output {i}: SCOPE_MISMATCH dtype oracle={oracle.dtype} eager={eager.dtype}")
            ok_all = False
            continue
        if not same_stride:
            print(f"  output {i}: SCOPE_MISMATCH stride oracle={oracle.stride()} eager={eager.stride()}")
            ok_all = False
            continue

        if eager.is_floating_point():
            max_diff = (eager.float() - oracle.float()).abs().max().item()
            values_ok = torch.allclose(eager.float(), oracle.float(), rtol=rtol, atol=atol)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(shape={list(eager.shape)} dtype={eager.dtype} stride={eager.stride()} "
                f"max_diff={max_diff:.2e})"
            )
        else:
            values_ok = torch.equal(eager, oracle)
            mismatches = int((eager != oracle).sum().item()) if not values_ok else 0
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(exact, shape={list(eager.shape)} dtype={eager.dtype} "
                f"stride={eager.stride()} mismatches={mismatches})"
            )
        ok_all = ok_all and bool(values_ok)

    print("check compared against full Repro.forward, including both var_mean reductions, "
          "low-memory max-pool offsets, second normalization, centered side output, "
          "and four in-place running-stat copy_ outputs")
    print(f"Correctness: {'PASS' if ok_all else 'FAIL'}")
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

    oracle_inputs = _clone_inputs(tuple(get_inputs()))
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
                compile_inputs = _clone_inputs(tuple(get_inputs()))
                compiled = _compile_with_config(compile_inputs, config, warmup)
                holder: list[Any] = [None]
                compile_results[label] = _bench_cuda_graph(
                    lambda compiled=compiled, inputs=compile_inputs: holder.__setitem__(
                        0, compiled(*inputs)
                    ),
                    warmup=warmup,
                    rep=rep,
                )
            except Exception as exc:  # pragma: no cover - diagnostic path
                print(f"torch.compile {label}: FAILED ({exc})")
            finally:
                torch.cuda.synchronize()

    measured_best = min(compile_results.values()) if compile_results else math.nan
    comparison_best = HISTORICAL_BEST_COMPILE_US
    if compile_results:
        comparison_best = min(comparison_best, measured_best)
    ratio = comparison_best / oracle_us if oracle_us > 0 else math.nan
    true_floor = bool(
        len(compile_results) == len(COMPILE_CONFIGS)
        and all(oracle_us < value for value in compile_results.values())
        and oracle_us < HISTORICAL_BEST_COMPILE_US
    )

    logical_read_bytes = (
        TOTAL_X * 4
        + TOTAL_POOL * 9 * 4
        + TOTAL_POOL * 4
        + TOTAL_POOL * 4
    )
    logical_write_bytes = TOTAL_POOL * (4 + 1 + 4 + 4) + C * 5 * 4
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(measured_best, 3) if compile_results else None,
        "compile_configs_us": {k: round(v, 3) for k, v in compile_results.items()},
        "historical_best_compile_us": round(HISTORICAL_BEST_COMPILE_US, 3),
        "comparison_best_compile_us": round(comparison_best, 3),
        "ratio": round(ratio, 3),
        "valid_floor": true_floor,
        "true_floor": true_floor,
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "classification": CLASSIFICATION,
        "logical_bytes": logical_read_bytes + logical_write_bytes,
    }
    print(json.dumps(result))
    print(f"oracle full-scope two-BN/pool region: {oracle_us:.3f} us")
    for label, value in compile_results.items():
        print(f"torch.compile {label}: {value:.3f} us")
    print(f"historical best_compile_us: {HISTORICAL_BEST_COMPILE_US:.3f} us")
    print(f"true floor: {'yes' if true_floor else 'no'}")
    if not true_floor:
        print(
            "diagnosis_only: oracle is not a true floor because it does not beat "
            "every required local compile config and the historical best_compile_us"
        )
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
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(warmup=args.warmup, rep=args.rep, no_compile=args.no_compile)


if __name__ == "__main__":
    main()
