"""
Oracle for pointwise_36d7a6209c54

Gap diagnosis:
  Classification: BANDWIDTH_BOUND
  What oracle does differently: uses one minimal Triton scalar pointwise/index kernel for the full two-index lookup and bool result.
  What Inductor change would fix: no scheduler change is indicated unless a future full-scope kernel beats the launch-floor historical best.
"""
from __future__ import annotations

import argparse
import importlib.util
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "BANDWIDTH_BOUND"
HISTORICAL_BEST_COMPILE_US = 5.0

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

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def oracle_kernel(arg2_ptr, arg3_ptr, table_ptr, output_ptr):
        idx2 = tl.load(arg2_ptr)
        idx3 = tl.load(arg3_ptr)
        val2 = tl.load(table_ptr + idx2.to(tl.int64))
        val3 = tl.load(table_ptr + idx3.to(tl.int64))
        result = (idx2 >= idx3) & (val2 == val3)
        tl.store(output_ptr, result)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"expected three inputs, got {len(inputs)}")
    arg2, arg3, table = inputs
    if not all(isinstance(value, torch.Tensor) for value in (arg2, arg3, table)):
        raise TypeError("expected tensor inputs")
    if arg2.shape != torch.Size([]) or arg3.shape != torch.Size([]):
        raise ValueError(f"expected scalar index tensors, got {tuple(arg2.shape)} and {tuple(arg3.shape)}")
    if arg2.dtype != torch.int32 or arg3.dtype != torch.int32:
        raise TypeError(f"expected int32 scalar indices, got {arg2.dtype} and {arg3.dtype}")
    if tuple(table.shape) != (6144,) or table.dtype != torch.int64:
        raise TypeError(f"expected int64[6144] table, got {table.dtype}{tuple(table.shape)}")
    if not (arg2.is_cuda and arg3.is_cuda and table.is_cuda):
        raise RuntimeError("oracle_layout_stencil.py expects CUDA inputs")
    if arg2.device != arg3.device or arg2.device != table.device:
        raise ValueError("all inputs must be on the same CUDA device")
    if not table.is_contiguous():
        raise ValueError("expected contiguous int64 lookup table")
    return arg2, arg3, table


def _launch_oracle(
    arg2: torch.Tensor,
    arg3: torch.Tensor,
    table: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")
    oracle_kernel[(1,)](arg2, arg3, table, output)
    return output


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    arg2, arg3, table = _validate_inputs(inputs)
    output = torch.empty((), device=table.device, dtype=torch.bool)
    return _launch_oracle(arg2, arg3, table, output)


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bench_cuda_graph(fn: Any, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()

    from triton.testing import do_bench

    return do_bench(lambda: graph.replay(), warmup=warmup, rep=rep, return_mode="min") * 1000.0


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


def run_bench(warmup: int, rep: int, no_compile: bool) -> dict[str, object]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    module = _load_repro_module()
    inputs = tuple(module.make_inputs())
    arg2, arg3, table = _validate_inputs(inputs)
    output = torch.empty((), device=table.device, dtype=torch.bool)

    with torch.no_grad():
        oracle_us = _bench_cuda_graph(
            lambda: _launch_oracle(arg2, arg3, table, output),
            warmup=warmup,
            rep=rep,
        )

    print(
        "oracle shape: "
        f"arg2={arg2.dtype}{tuple(arg2.shape)} arg3={arg3.dtype}{tuple(arg3.shape)} "
        f"table={table.dtype}{tuple(table.shape)} output=torch.bool[]"
    )
    print(f"oracle full-scope scalar index/compare: {oracle_us:.3f} us")
    print(f"oracle_us={oracle_us:.3f}")

    compile_results: dict[str, float] = {}
    if not no_compile:
        holder: list[Any] = [None]
        print("torch.compile timings cover the same full Repro.forward return value")
        for label, config in COMPILE_CONFIGS:
            compiled = _compile_with_config(module, inputs, config, warmup)
            compile_us = _bench_cuda_graph(
                lambda: holder.__setitem__(0, compiled(*inputs)),
                warmup=warmup,
                rep=rep,
            )
            compile_results[label] = compile_us
            print(f"torch.compile {label}: {compile_us:.3f} us")

    best_required_compile_us = min(compile_results.values()) if compile_results else None
    beats_required_compile = (
        len(compile_results) == len(COMPILE_CONFIGS)
        and best_required_compile_us is not None
        and oracle_us < best_required_compile_us
    )
    beats_historical_best = oracle_us < HISTORICAL_BEST_COMPILE_US
    true_floor = bool(beats_required_compile and beats_historical_best)

    if best_required_compile_us is not None:
        print(f"best_required_compile_us={best_required_compile_us:.3f}")
    print(f"historical_best_compile_us={HISTORICAL_BEST_COMPILE_US:.3f}")
    print(f"beats_required_compile={beats_required_compile}")
    print(f"beats_historical_best={beats_historical_best}")
    print(f"true_floor={true_floor}")
    if not true_floor:
        print("diagnosis_only: scalar full-scope oracle is at the launch/bandwidth floor")

    ratio = (best_required_compile_us / oracle_us) if best_required_compile_us else 0.0
    result = {
        "repro_id": REPRO_ID,
        "oracle_us": round(oracle_us, 3),
        "compile_us": round(best_required_compile_us, 3) if best_required_compile_us else None,
        "ratio": round(ratio, 3),
        "status": "GOOD" if true_floor else "DIAGNOSIS_ONLY",
        "classification": CLASSIFICATION,
        "compile_configs_us": {label: round(value, 3) for label, value in compile_results.items()},
        "historical_best_compile_us": HISTORICAL_BEST_COMPILE_US,
        "beats_required_compile": beats_required_compile,
        "beats_historical_best": beats_historical_best,
        "true_floor": true_floor,
    }
    print(json.dumps(result, sort_keys=True))
    return result


# --- CLI entry point ---
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
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--no-compile", action="store_true",
                        help="Skip torch.compile baselines for the requested configs")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    # Handle --show-hw early
    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        run_bench(args.warmup, args.rep, args.no_compile)


if __name__ == "__main__":
    main()
