"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full `arg0_1 > 0`, metadata-only flatten view, and scalar `aten.any` reduction as one single-program Triton reduction, whereas Inductor currently emits a generic fused scalar-reduction template with an artificial size-1 X dimension, a reduction loop, a boolean accumulator tensor, and the helper `any` epilogue; Inductor cannot do this today because its scheduler/codegen lacks a zero-dimensional any/all specialization for small flattened reductions whose only producer is an inlineable predicate; the fix is SCHEDULER_FUSION: teach Inductor's reduction scheduler to lower such scalar any/all patterns to a compact one-block predicate reduction without the generic X/R loop scaffold."""
from __future__ import annotations

import argparse
import sys
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None

from repro_harness import load_shape_configs, make_inputs_from_config


REPRO_DIR = Path(__file__).resolve().parent

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)
REPRO_PATH = REPRO_DIR / "repro.py"
REPRO_ID = REPRO_DIR.name
HISTORICAL_BEST_COMPILE_US = 5.7


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shape_product(shape: Any) -> int:
    if isinstance(shape, int):
        return int(shape)
    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _default_inputs() -> list[Any]:
    module = _load_repro_module()
    return module.make_inputs()


def _named_inputs(shape_name: str | None) -> tuple[str, list[Any]]:
    if shape_name is None:
        return "default", _default_inputs()

    configs = load_shape_configs(str(REPRO_PATH))
    if shape_name not in configs:
        available = ", ".join(configs)
        raise ValueError(f"unknown shape {shape_name!r}; available shapes: {available}")
    return shape_name, make_inputs_from_config(configs[shape_name])


if triton is not None:

    @triton.jit
    def _gt_zero_any_kernel(
        x_ptr,
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        flags = (values > 0.0).to(tl.int32)
        any_positive = tl.max(flags, axis=0)
        tl.store(out_ptr, any_positive != 0)


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> torch.Tensor:
    """Compute exactly Repro()(*make_inputs()) for this captured graph."""
    if triton is None:
        raise RuntimeError("triton is required for oracle_any_reduce.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg0_1, shape_param = inputs
    if not isinstance(arg0_1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if arg0_1.dtype not in (torch.float16, torch.float32):
        raise ValueError(f"{REPRO_ID} expects f16 or f32 input, got {arg0_1.dtype}")
    if not arg0_1.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA inputs")
    if not arg0_1.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects the captured contiguous input layout")
    if _shape_product(shape_param) != arg0_1.numel():
        raise ValueError(
            f"{REPRO_ID} shape parameter {shape_param!r} does not match "
            f"input numel {arg0_1.numel()}"
        )

    out = torch.empty((), device=arg0_1.device, dtype=torch.bool)
    block = triton.next_power_of_2(arg0_1.numel())
    _gt_zero_any_kernel[(1,)](
        arg0_1,
        out,
        n_elements=arg0_1.numel(),
        BLOCK=block,
        num_warps=16,
        num_stages=1,
    )
    return out


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
