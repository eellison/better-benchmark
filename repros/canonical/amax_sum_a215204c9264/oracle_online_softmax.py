"""Full-scope Triton oracle for amax_sum_a215204c9264.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the
complete f32 `[128,2]` log_softmax region from `repro.py` in one
shape-specialized Triton kernel, covering the row amax, subtract, exp/sum,
log, and final subtract before writing the `[128,2]` output, while Inductor's
required configs already emit a faster compact compiled kernel for the same
full scope. Inductor cannot be materially improved by scheduler fusion here
because the tensor has only 256 elements and latency is dominated by one GPU
launch plus scalar exp/log work, not by extra memory traffic or missing fusion.
The required Inductor change is BANDWIDTH_BOUND: do not add a new lowering for
this repro unless a broader tiny-K log_softmax template beats the current
compiled kernel under interleaved timing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_ID = "amax_sum_a215204c9264"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 128
COLS = 2
OUT_SHAPE = (ROWS, COLS)
OUT_STRIDE = (COLS, 1)
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


@triton.jit
def _log_softmax_k2_kernel(
    x_ptr,
    out_ptr,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    block_m: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    mask = rows < 128

    x0 = tl.load(x_ptr + rows * x_s0, mask=mask, other=-float("inf")).to(tl.float32)
    x1 = tl.load(x_ptr + rows * x_s0 + x_s1, mask=mask, other=-float("inf")).to(tl.float32)

    row_max = tl.maximum(x0, x1)
    shifted0 = x0 - row_max
    shifted1 = x1 - row_max
    exp0 = tl.exp(shifted0)
    exp1 = tl.exp(shifted1)
    exp_sum = exp0 + exp1
    log_sum = tl.log(exp_sum)
    out0 = shifted0 - log_sum
    out1 = shifted1 - log_sum

    tl.store(out_ptr + rows * out_s0, out0, mask=mask)
    tl.store(out_ptr + rows * out_s0 + out_s1, out1, mask=mask)


def _load_repro_module() -> Any:
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
    return tuple(
        item.cuda() if isinstance(item, torch.Tensor) and not item.is_cuda else item
        for item in module.make_inputs()
    )


def _validate_input(addmm_72: torch.Tensor) -> None:
    if not addmm_72.is_cuda:
        raise RuntimeError("CUDA input is required")
    if addmm_72.dtype != torch.float32:
        raise TypeError(f"expected fp32 input, got {addmm_72.dtype}")
    if tuple(addmm_72.shape) != OUT_SHAPE:
        raise ValueError(f"expected input shape {OUT_SHAPE}, got {tuple(addmm_72.shape)}")


def _make_output(device: torch.device) -> torch.Tensor:
    return torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.float32)


def _launch_oracle(addmm_72: torch.Tensor, out: torch.Tensor, *, block_m: int) -> torch.Tensor:
    _validate_input(addmm_72)
    if tuple(out.shape) != OUT_SHAPE or tuple(out.stride()) != OUT_STRIDE:
        raise ValueError(f"output must have shape {OUT_SHAPE} and stride {OUT_STRIDE}")
    if out.dtype != torch.float32 or not out.is_cuda:
        raise ValueError("output must be CUDA fp32")

    _log_softmax_k2_kernel[(triton.cdiv(ROWS, block_m),)](
        addmm_72,
        out,
        x_s0=addmm_72.stride(0),
        x_s1=addmm_72.stride(1),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        block_m=block_m,
        num_warps=4,
    )
    return out


def oracle_online_softmax(addmm_72: torch.Tensor, *, block_m: int = 32) -> torch.Tensor:
    out = _make_output(addmm_72.device)
    return _launch_oracle(addmm_72, out, block_m=block_m)


def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    (addmm_72,) = inputs
    return oracle_online_softmax(addmm_72)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    diff = (actual_f32 - expected_f32).abs()
    max_abs = torch.nan_to_num(diff, nan=0.0).max().item()
    rel = diff / expected_f32.abs().clamp_min(1e-8)
    max_rel = torch.nan_to_num(rel, nan=0.0).max().item()
    return max_abs, max_rel


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


def run_check(block_m: int, rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=1234)
    model = module.Repro().cuda()

    with torch.no_grad():
        expected = _as_tuple(model(*inputs))
        actual = _as_tuple(oracle_forward(inputs))
        torch.cuda.synchronize()

    if len(actual) != len(expected):
        print(f"  SCOPE_MISMATCH: expected {len(expected)} outputs, got {len(actual)}")
        print("Correctness: FAIL")
        return False

    ok = True
    for idx, (got_item, ref_item) in enumerate(zip(actual, expected)):
        if not isinstance(got_item, torch.Tensor) or not isinstance(ref_item, torch.Tensor):
            item_ok = got_item == ref_item
            print(f"  output {idx}: {'PASS' if item_ok else 'FAIL'} non-tensor")
            ok = ok and bool(item_ok)
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
            f"  output {idx}: {'PASS' if item_ok else 'FAIL'} "
            f"shape={list(got_item.shape)} dtype={got_item.dtype} "
            f"stride={list(got_item.stride())} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def run_bench(block_m: int, warmup: int, rep: int, no_compile: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = _make_inputs(module, seed=4321)
    (addmm_72,) = inputs
    _validate_input(addmm_72)
    out = _make_output(addmm_72.device)
    logical_bytes = ROWS * COLS * 4 * 2

    print(
        "oracle shape: "
        f"addmm_72=f32[{ROWS},{COLS}] stride={tuple(addmm_72.stride())} "
        f"out=f32[{ROWS},{COLS}] stride={OUT_STRIDE}"
    )

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(addmm_72, out, block_m=block_m),
            warmup=warmup,
            rep=rep,
        )

    oracle_bw = logical_bytes / (oracle_us * 1e-6) / 1e9
    print(f"oracle full-scope log_softmax: {oracle_us:.3f} us ({oracle_bw:.3f} GB/s logical bytes)")
    print(f"oracle_us={oracle_us:.3f}")

    compile_times: list[tuple[str, float]] = []
    if not no_compile:
        print("CUDA graph replay timings cover the same repro.py amax, subtract, exp, sum, log, and subtract")
        for label, config in COMPILE_CONFIGS:
            try:
                compiled = _compile_with_config(module, inputs, config, warmup)
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
    result: dict[str, Any] = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(best_compile, 3) if compile_times else None,
        "ratio": round(ratio, 3) if compile_times else None,
        "status": status,
        "classification": CLASSIFICATION,
        "compile_results": [
            {"label": label, "us": round(us, 3)} for label, us in compile_times
        ],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Oracle for {REPRO_ID}")
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=200)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = args.bench = True

    if args.check:
        print(f"Checking {REPRO_ID}...")
        if not run_check(args.block_m, args.rtol, args.atol):
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(args.block_m, args.warmup, args.rep, args.no_compile)


if __name__ == "__main__":
    main()
