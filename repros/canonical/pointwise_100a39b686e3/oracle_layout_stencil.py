"""
Oracle for pointwise_100a39b686e3

Gap diagnosis:
  Classification: BANDWIDTH_BOUND
  What oracle does differently: uses one full-scope Triton pointwise kernel to increment the scalar int64 input in place and return the mutated tensor.
  What Inductor change would fix: no scheduler/layout change is indicated; this repro is already at the single-kernel launch floor for a 16-byte in-place scalar update.
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
HISTORICAL_BEST_COMPILE_US = 5.024000070989132
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


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> list[Any]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return cloned


if triton is not None:

    @triton.jit
    def _increment_scalar_i64_kernel(arg_ptr):
        value = tl.load(arg_ptr)
        tl.store(arg_ptr, value + 1)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the same input tuple as Repro.forward() and returns
    the same single scalar int64 tensor after mutating that input in place.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")

    (arg560_1,) = inputs
    if not isinstance(arg560_1, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(arg560_1)!r}")
    if tuple(arg560_1.shape) != ():
        raise ValueError(f"unexpected input shape: {tuple(arg560_1.shape)}")
    if arg560_1.dtype is not torch.int64:
        raise ValueError(f"unexpected input dtype: {arg560_1.dtype}")
    if not arg560_1.is_cuda:
        raise ValueError("oracle_layout_stencil.py expects CUDA inputs")

    _increment_scalar_i64_kernel[(1,)](arg560_1)
    return arg560_1


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
    """Correctness check that isolates the in-place input mutation."""
    inputs = get_inputs()
    eager_inputs = _clone_inputs(inputs)
    oracle_inputs = _clone_inputs(inputs)
    instance = get_repro_instance()

    with torch.no_grad():
        eager = instance(*eager_inputs)
        oracle_out = oracle_forward(oracle_inputs)
    if oracle_inputs[0].is_cuda:
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
        if expected.dtype != actual.dtype:
            print(
                f"  output {i}: FAIL dtype oracle={actual.dtype} "
                f"eager={expected.dtype}"
            )
            all_pass = False
            continue
        if expected.is_floating_point():
            max_diff = (expected.float() - actual.float()).abs().max().item()
            ok = torch.allclose(expected.float(), actual.float(), atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} "
                f"max_diff={max_diff:.2e})"
            )
        else:
            ok = torch.equal(expected, actual)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} exact)"
            )
        all_pass = all_pass and ok

    mutation_ok = torch.equal(eager_inputs[0], oracle_inputs[0])
    print(f"  input mutation: {'PASS' if mutation_ok else 'FAIL'} (arg0 in-place copy_ semantics)")
    all_pass = all_pass and mutation_ok

    alias_ok = (
        isinstance(oracle_out, torch.Tensor)
        and oracle_out.data_ptr() == oracle_inputs[0].data_ptr()
    )
    print(f"  output alias: {'PASS' if alias_ok else 'FAIL'} (returned tensor aliases arg0)")
    return all_pass and alias_ok


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
    true_floor = oracle_us < best_required_compile_us and oracle_us < HISTORICAL_BEST_COMPILE_US
    status = "GOOD" if true_floor else "AT_FLOOR"
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(compile_results[COMPILE_CONFIGS[0][0]], 3),
        "combo_compile_us": round(compile_results[COMPILE_CONFIGS[1][0]], 3),
        "best_required_compile_us": round(best_required_compile_us, 3),
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "ratio": round(best_required_compile_us / oracle_us, 3) if oracle_us > 0 else 0.0,
        "status": status,
        "classification": CLASSIFICATION,
        "true_floor": true_floor,
    }
    print(json.dumps(result))
    if not true_floor:
        print("diagnosis_only: launch-floor scalar update; not_true_floor")
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
                        help="Accepted for template compatibility; all shapes share the same scalar signature")
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
            print("NOTE: --all-shapes requested; all listed shapes use the same scalar signature")
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
