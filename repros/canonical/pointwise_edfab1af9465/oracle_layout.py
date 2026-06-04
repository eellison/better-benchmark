"""
Full-scope oracle for pointwise_edfab1af9465.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full
`aten.tanh.default` over the default contiguous `float32[1000, 16]` input and
returns the required fresh contiguous `float32[1000, 16]` tensor with stride
`(16, 1)`. Inductor already lowers this single pointwise op to one fused Triton
kernel, so the remaining work is the required input read, tanh evaluation,
output materialization, allocation, and standalone launch rather than a missing
scheduler-fusion, scatter-reduce, cooperative split-K, algebraic-elimination,
or recompute-fusion transformation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # pragma: no cover - keeps py_compile useful.
    triton = None
    tl = None
    libdevice = None

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

INPUT_SHAPE = (1000, 16)
OUTPUT_SHAPE = INPUT_SHAPE
OUTPUT_STRIDE = (16, 1)
NUMEL = 1000 * 16
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL"],
    )
    @triton.jit
    def _tanh_f32_kernel(
        in_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL
        values = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        result = libdevice.tanh(values)
        tl.store(out_ptr + offsets, result, mask=mask)


def _require_input(value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects a tensor input, got {type(value)!r}")
    if tuple(value.shape) != INPUT_SHAPE:
        raise ValueError(f"{REPRO_ID} expects shape {INPUT_SHAPE}, got {tuple(value.shape)}")
    if value.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects torch.float32 input, got {value.dtype}")
    if value.stride() != OUTPUT_STRIDE:
        raise ValueError(f"{REPRO_ID} expects contiguous stride {OUTPUT_STRIDE}, got {value.stride()}")
    if not value.is_cuda:
        raise ValueError("oracle_layout.py expects CUDA tensor inputs")
    return value


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with one Triton pointwise kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (addmm_2,) = inputs
    addmm_2 = _require_input(addmm_2)

    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm_2.device,
        dtype=torch.float32,
    )
    grid = lambda meta: (triton.cdiv(NUMEL, meta["BLOCK"]),)
    _tanh_f32_kernel[grid](
        addmm_2,
        out,
        TOTAL=NUMEL,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[object] | tuple[object, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        tuple(actual.shape) == tuple(expected.shape) == OUTPUT_SHAPE
        and actual.dtype == expected.dtype == torch.float32
        and actual.stride() == expected.stride() == OUTPUT_STRIDE
        and actual.storage_offset() == expected.storage_offset() == 0
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(actual.shape)} stride={actual.stride()} "
        f"dtype={actual.dtype} storage_offset={actual.storage_offset()})"
    )
    return ok


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
        ok = _check_layout(instance, inputs) and ok
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
