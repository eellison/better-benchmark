"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle covers the complete scalar int64 in-place add/copy_ scope by loading arg560_1, adding one, storing back to the same zero-dimensional CUDA tensor, and returning that alias from one Triton program, whereas Inductor currently emits the same one-launch scalar mutation and measures at the graph-captured launch floor; Inductor cannot do this today because its scheduler has no lower-cost representation for a user-visible in-place scalar write than one generated pointwise launch; the fix is SCHEDULER_FUSION: add a broader scalar in-place update lowering that bypasses generic pointwise scheduling only if it can beat the existing one-launch floor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def oracle_kernel(arg_ptr):
        value = tl.load(arg_ptr)
        tl.store(arg_ptr, value + 1)


@oracle_impl(hardware="H100", shapes="(T([], i64))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scalar in-place update.

    SCOPE INVARIANT: accepts the same single scalar int64 tensor as
    Repro.forward(), mutates it in place, and returns that same tensor alias.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")
    if len(inputs) != 1:
        raise ValueError(f"expected one input tensor, got {len(inputs)}")

    arg560_1 = inputs[0]
    if not isinstance(arg560_1, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(arg560_1)!r}")
    if tuple(arg560_1.shape) != ():
        raise ValueError(f"unexpected input shape: {tuple(arg560_1.shape)}")
    if arg560_1.dtype != torch.int64:
        raise ValueError(f"unexpected input dtype: {arg560_1.dtype}")
    if arg560_1.device.type != "cuda":
        raise RuntimeError("oracle_layout_stencil.py expects a CUDA input tensor")

    oracle_kernel[(1,)](arg560_1, num_warps=1)
    return arg560_1


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> list[Any]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return cloned


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _has_cuda_tensor(inputs: list[Any] | tuple[Any, ...]) -> bool:
    return any(isinstance(item, torch.Tensor) and item.is_cuda for item in inputs)


def _check_inplace_layout_and_alias(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
) -> bool:
    eager_inputs = _clone_inputs(inputs)
    oracle_inputs = _clone_inputs(inputs)

    with torch.no_grad():
        expected = instance(*eager_inputs)
        actual = oracle_forward(oracle_inputs)
    if _has_cuda_tensor(oracle_inputs):
        torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.dtype == expected_tensor.dtype
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} "
            f"dtype={actual_tensor.dtype})"
        )
        ok = ok and layout_ok

        if expected_tensor.is_floating_point():
            max_diff = (expected_tensor.float() - actual_tensor.float()).abs().max().item()
            values_ok = torch.allclose(expected_tensor.float(), actual_tensor.float())
        else:
            max_diff = (
                expected_tensor.to(torch.float64) - actual_tensor.to(torch.float64)
            ).abs().max().item()
            values_ok = torch.equal(expected_tensor, actual_tensor)
        print(
            f"  output {index} values: {'PASS' if values_ok else 'FAIL'} "
            f"(max_diff={max_diff:.2e})"
        )
        ok = ok and values_ok

    mutation_ok = torch.equal(eager_inputs[0], oracle_inputs[0])
    print(
        f"  input mutation: {'PASS' if mutation_ok else 'FAIL'} "
        "(arg0 in-place copy_ semantics)"
    )
    ok = ok and mutation_ok

    alias_ok = (
        isinstance(actual, torch.Tensor)
        and actual.data_ptr() == oracle_inputs[0].data_ptr()
        and actual.storage_offset() == oracle_inputs[0].storage_offset()
    )
    print(f"  output alias: {'PASS' if alias_ok else 'FAIL'} (returned tensor aliases arg0)")
    return ok and alias_ok


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
        import json

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
        ok = _check_inplace_layout_and_alias(instance, inputs) and ok
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
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
