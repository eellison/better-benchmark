"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full no-input Repro.forward scope by folding the full(1)->unsqueeze->unsqueeze->fp16 cast->1.0-sub chain into a direct fresh float16 [1, 1, 1, 4096] positive-zero tensor with stride (4096, 4096, 4096, 1), whereas Inductor currently lowers the constant full/view/cast/sub expression as generic scheduled pointwise work that still materializes the tiny output through the normal elementwise kernel path; Inductor cannot do this today because its scheduler/codegen simplifier does not propagate scalar constants through metadata-only views, dtype conversion, and scalar-sub before lowering no-input graphs; the fix is ALGEBRAIC_ELIMINATION: add shape-aware constant propagation across full/view/cast/scalar pointwise chains and emit a direct constant-fill output when the full expression collapses to one scalar."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


OUTPUT_SHAPE = (1, 1, 1, 4096)
OUTPUT_STRIDE = (4096, 4096, 4096, 1)
OUTPUT_NUMEL = 4096


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fill_positive_zero_bits_kernel(
        out_bits_ptr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        zeros = tl.full((BLOCK_SIZE,), 0, tl.uint16)
        tl.store(out_bits_ptr + offsets, zeros, mask=mask)


@oracle_impl(hardware="H100", shapes="()")
def oracle_forward(inputs):
    """Materialize the same f16 [1, 1, 1, 4096] zero mask as Repro.forward()."""
    if inputs:
        raise AssertionError(f"expected no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device="cuda",
        dtype=torch.float16,
    )
    block_size = 1024
    grid = (triton.cdiv(OUTPUT_NUMEL, block_size),)
    _fill_positive_zero_bits_kernel[grid](
        output.view(torch.uint16),
        N=OUTPUT_NUMEL,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output


def _check_layout_and_sign(out):
    return (
        tuple(out.shape) == OUTPUT_SHAPE
        and out.stride() == OUTPUT_STRIDE
        and out.dtype == torch.float16
        and out.device.type == "cuda"
        and not torch.signbit(out).any().item()
    )


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
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout_and_sign(layout_out)
        print(
            f"  output 0 layout/sign: {'PASS' if layout_ok else 'FAIL'} "
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
