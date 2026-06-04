"""
Full-scope Triton oracle for pointwise_4d236bfe44e3.

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle materializes the
complete torch.cat result for three contiguous f32[768] inputs into the returned
contiguous f32[2304] output with one Triton copy kernel, while tuned Inductor
already reaches the same tiny materialization regime; any remaining difference
is allocation, launch, and 18 KiB of required memory traffic rather than a
missing scheduler transformation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
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

INPUT_NUMEL = 768
OUTPUT_NUMEL = 3 * INPUT_NUMEL
BLOCK = 1024
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _cat3_f32_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        out_ptr,
        n: tl.constexpr,
        stride_a: tl.constexpr,
        stride_b: tl.constexpr,
        stride_c: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        segment = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n

        a = tl.load(a_ptr + offsets * stride_a, mask=mask & (segment == 0), other=0.0)
        b = tl.load(b_ptr + offsets * stride_b, mask=mask & (segment == 1), other=0.0)
        c = tl.load(c_ptr + offsets * stride_c, mask=mask & (segment == 2), other=0.0)
        values = a + b + c
        tl.store(out_ptr + segment * n + offsets, values, mask=mask)


def _validate_input(name: str, tensor: object) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
    if tuple(tensor.shape) != (INPUT_NUMEL,):
        raise ValueError(f"{name} has unexpected shape {tuple(tensor.shape)}")
    if tensor.dtype is not torch.float32:
        raise ValueError(f"{name} has unexpected dtype {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return tensor


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the complete three-input cat materialization from Repro.forward()."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    a = _validate_input("arg205_1", inputs[0])
    b = _validate_input("arg206_1", inputs[1])
    c = _validate_input("arg207_1", inputs[2])
    if not (a.device == b.device == c.device):
        raise ValueError("all inputs must be on the same CUDA device")

    out = torch.empty_strided((OUTPUT_NUMEL,), (1,), device=a.device, dtype=torch.float32)
    _cat3_f32_kernel[(3,)](
        a,
        b,
        c,
        out,
        n=INPUT_NUMEL,
        stride_a=a.stride(0),
        stride_b=b.stride(0),
        stride_c=c.stride(0),
        BLOCK_SIZE=BLOCK,
        num_warps=4,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for correctness check")
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
        with torch.no_grad():
            out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = (
            tuple(out.shape) == (OUTPUT_NUMEL,)
            and out.stride() == (1,)
            and out.dtype is torch.float32
        )
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(out.shape)} stride={out.stride()} dtype={out.dtype})"
        )
        ok = ok and layout_ok
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
