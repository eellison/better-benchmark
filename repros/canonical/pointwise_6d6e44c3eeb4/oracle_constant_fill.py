"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the entire zero-input `aten.full.default([32, 128], 1, dtype=float32, device=cuda:0)` result by allocating the returned contiguous tensor and writing all 4096 elements in one minimal Triton constant-fill launch, whereas Inductor currently lowers the captured full as a generic pointwise fill kernel with standard elementwise scheduling and launch-shape overhead; Inductor cannot do this today because its pattern lowering does not special-case small no-input constant full producers into a scalar-parameterized constant-fill template with a minimal launch shape; the fix is NEW_PATTERN: add an Inductor constant tensor creation lowering for small `aten.full` outputs that emits a specialized fill/memset-style Triton kernel using the exact output layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful.
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

OUT_SHAPE = (32, 128)
OUT_STRIDE = (128, 1)
OUT_NUMEL = OUT_SHAPE[0] * OUT_SHAPE[1]
BLOCK = 256
GRID = (OUT_NUMEL // BLOCK,)
CUDA_DEVICE = torch.device("cuda", 0)


def get_inputs() -> tuple[object, ...]:
    """Load inputs from repro.py."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _fill_one_f32_kernel(
        out_ptr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        tl.store(out_ptr + offsets, 1.0)

    _fill_one_f32_launcher = _fill_one_f32_kernel[GRID]
else:

    def _fill_one_f32_launcher(*_args, **_kwargs):
        raise RuntimeError("Triton is required for this oracle")


@oracle_impl(hardware="H100", shapes="()")
def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Compute exactly Repro()(*make_inputs()) for this no-input full() repro."""
    del inputs

    out = torch.empty(
        OUT_SHAPE,
        device=CUDA_DEVICE,
        dtype=torch.float32,
    )
    _fill_one_f32_launcher(
        out,
        block=BLOCK,
        num_warps=1,
        num_stages=1,
    )
    return out


def _check_layout(out: torch.Tensor) -> bool:
    return (
        tuple(out.shape) == OUT_SHAPE
        and out.stride() == OUT_STRIDE
        and out.dtype == torch.float32
        and out.device.type == "cuda"
        and out.device.index == 0
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify correctness against eager Repro",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark oracle vs torch.compile",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Repetitions for benchmark",
    )
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Benchmark across all shapes from shapes.txt",
    )
    parser.add_argument(
        "--show-hw",
        action="store_true",
        help="Print GPU hardware info and exit",
    )
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
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()} "
            f"dtype={layout_out.dtype} device={layout_out.device})"
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
