"""
Oracle for pointwise_5b4d8d4d06aa

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full captured view/add/view/permute graph with one Triton pointwise kernel that writes the output backing storage for the final `[128, 12, 1, 64]` tensor directly with eager's permuted stride `(768, 64, 768, 1)`, whereas Inductor lowers the same small layout-stencil region to an equivalent fused pointwise/layout kernel under the required tuned configurations; Inductor cannot materially improve this today because the repro has no remaining producer to fuse, reduction to reorganize, scatter to rewrite, or algebra to eliminate, so one launch plus the required `mm` read, broadcast-bias read, and output write dominate; the required Inductor change is BANDWIDTH_BOUND: treat this as a closed/diagnosis-only floor unless a broader launch-overhead reduction applies.
"""
from __future__ import annotations

import argparse
import contextlib
import json
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "BANDWIDTH_BOUND"

M = 128
K = 768
N_HEADS = 12
HEAD_DIM = 64
OUTPUT_SHAPE = (M, N_HEADS, 1, HEAD_DIM)
OUTPUT_STRIDE = (K, HEAD_DIM, K, 1)
NUMEL = M * K

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


from oracle_harness import (  # noqa: E402
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def get_hardware_info():
    """Get GPU properties with compatibility across PyTorch property names."""
    props = torch.cuda.get_device_properties(0)
    shared_mem = getattr(
        props,
        "max_shared_memory_per_multiprocessor",
        getattr(props, "shared_memory_per_multiprocessor", None),
    )
    return {
        "name": props.name,
        "sm_major": props.major,
        "sm_minor": props.minor,
        "num_sms": props.multi_processor_count,
        "shared_mem_per_sm": shared_mem,
        "total_mem_gb": props.total_memory / 1e9,
    }


if triton is not None:

    @triton.jit
    def _add_bias_layout_kernel(
        mm,
        bias,
        out,
        total: tl.constexpr,
        row_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        values = tl.load(mm + offsets, mask=mask, other=0.0)
        bias_values = tl.load(bias + (offsets % row_size), mask=mask, other=0.0)
        tl.store(out + offsets, values + bias_values, mask=mask)


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> list[Any]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return cloned


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 4:
        raise ValueError(f"expected 4 repro inputs, got {len(inputs)}")
    mm, bias, shape0, shape1 = inputs
    if not isinstance(mm, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("expected tensor inputs for mm and bias")
    if tuple(mm.shape) != (M, K):
        raise ValueError(f"unexpected mm shape: {tuple(mm.shape)}")
    if tuple(bias.shape) != (K,):
        raise ValueError(f"unexpected bias shape: {tuple(bias.shape)}")
    if mm.dtype is not torch.float32 or bias.dtype is not torch.float32:
        raise ValueError(f"unexpected dtypes: mm={mm.dtype}, bias={bias.dtype}")
    if not mm.is_cuda or not bias.is_cuda:
        raise ValueError("oracle_layout_stencil.py expects CUDA tensor inputs")
    if not mm.is_contiguous() or not bias.is_contiguous():
        raise ValueError("oracle expects contiguous mm and bias inputs")
    if tuple(shape0) != (M, 1, K):
        raise ValueError(f"unexpected first view shape: {shape0}")
    if tuple(shape1) != (M, 1, N_HEADS, HEAD_DIM):
        raise ValueError(f"unexpected second view shape: {shape1}")
    return mm, bias


def oracle_forward(inputs):
    """Run the full-scope Triton oracle for Repro.forward()."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")

    mm, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=mm.device,
        dtype=mm.dtype,
    )
    block = 256
    grid = (triton.cdiv(NUMEL, block),)
    _add_bias_layout_kernel[grid](mm, bias, out, NUMEL, K, BLOCK=block)
    return out


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def run_check(atol: float, rtol: float) -> bool:
    inputs = get_inputs()
    instance = get_repro_instance()

    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
    if inputs[0].is_cuda:
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(eager_list) != len(oracle_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape "
                f"oracle={list(actual.shape)} eager={list(expected.shape)}"
            )
            all_pass = False
            continue
        stride_ok = expected.stride() == actual.stride()
        dtype_ok = expected.dtype == actual.dtype
        max_diff = (expected.float() - actual.float()).abs().max().item()
        value_ok = torch.allclose(expected.float(), actual.float(), atol=atol, rtol=rtol)
        ok = stride_ok and dtype_ok and value_ok
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} "
            f"stride={actual.stride()} max_diff={max_diff:.2e})"
        )
        if not stride_ok:
            print(
                f"    stride mismatch: oracle={actual.stride()} "
                f"eager={expected.stride()}"
            )
        if not dtype_ok:
            print(f"    dtype mismatch: oracle={actual.dtype} eager={expected.dtype}")
        all_pass = all_pass and ok

    return all_pass


def _get_config_value(config, key: str):
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    return getattr(target, parts[-1])


def _set_config_value(config, key: str, value):
    target = config
    parts = key.split(".")
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


@contextlib.contextmanager
def _temporary_inductor_config(settings: dict[str, object]):
    import torch._inductor.config as cfg

    saved = {key: _get_config_value(cfg, key) for key in settings}
    try:
        for key, value in settings.items():
            _set_config_value(cfg, key, value)
        yield
    finally:
        for key, value in saved.items():
            _set_config_value(cfg, key, value)


def _do_bench(fn, device: torch.device, warmup: int, rep: int) -> float:
    if triton is not None and device.type == "cuda":
        from triton.testing import do_bench

        return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    import math
    import time

    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    best_us = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        best_us = min(best_us, (time.perf_counter() - start) * 1_000_000.0)
    return best_us


def run_bench(warmup: int, rep: int) -> dict[str, object]:
    inputs = get_inputs()
    device = inputs[0].device

    with torch.no_grad():
        oracle_forward(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        oracle_us = _do_bench(lambda: oracle_forward(inputs), device, warmup, rep)

    compile_results: dict[str, float] = {}
    for label, settings in COMPILE_CONFIGS:
        torch._dynamo.reset()
        instance = get_repro_instance()
        compiled_inputs = _clone_inputs(get_inputs())
        with _temporary_inductor_config(settings):
            compiled = torch.compile(instance)
            with torch.no_grad():
                for _ in range(5):
                    compiled(*compiled_inputs)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                compile_results[label] = _do_bench(
                    lambda: compiled(*compiled_inputs),
                    device,
                    warmup,
                    rep,
                )

    best_required_compile_us = min(compile_results.values())
    true_floor = oracle_us < best_required_compile_us
    status = "GOOD" if true_floor else "AT_FLOOR"
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_results[COMPILE_CONFIGS[0][0]], 3),
        "combo_compile_us": round(compile_results[COMPILE_CONFIGS[1][0]], 3),
        "best_required_compile_us": round(best_required_compile_us, 3),
        "ratio": round(best_required_compile_us / oracle_us, 3) if oracle_us > 0 else 0.0,
        "status": status,
        "classification": CLASSIFICATION,
        "true_floor": true_floor,
    }
    print(json.dumps(result))
    if not true_floor:
        print("diagnosis_only: full-scope Triton kernel does not beat the best required Inductor config")
    return result


def main():
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
                        help="Warmup milliseconds for Triton do_bench")
    parser.add_argument("--rep", type=int, default=200,
                        help="Benchmark milliseconds for Triton do_bench")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Accepted for template compatibility; all listed shapes share the same signature")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = run_check(atol=args.atol, rtol=args.rtol)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        if args.all_shapes:
            print("NOTE: --all-shapes requested; all listed shapes share the same signature")
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
