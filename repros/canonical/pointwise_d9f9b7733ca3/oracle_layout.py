"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete no-input `aten.full.default([16, 1, 1, 512], -0.0, dtype=float32, device=cuda:0)` result by allocating the fresh contiguous output and writing the negative-zero `0x80000000` bit pattern with one minimal Triton fill kernel, whereas Inductor currently lowers the same constant tensor through generic pointwise full codegen for the tiny static output; Inductor cannot do this today because scheduler/codegen has no sign-bit-preserving constant-fill lowering for no-input `aten.full(-0.0)` that bypasses generic elementwise scheduling; the fix is NEW_PATTERN: add a guarded negative-zero constant-fill lowering that emits bitcast stores directly into the required output layout."""
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


OUT_SHAPE = (16, 1, 1, 512)
OUT_STRIDE = (512, 512, 512, 1)
OUT_NUMEL = 16 * 1 * 1 * 512
FILL_BLOCK = 512
NEG_ZERO_I32 = torch.iinfo(torch.int32).min
CLASSIFICATION = "NEW_PATTERN"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fill_negative_zero_kernel(
        out_ptr,
        numel: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < numel
        bits = tl.full((block,), -2147483648, tl.int32)
        values = bits.to(tl.float32, bitcast=True)
        tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="()")
def oracle_forward(inputs):
    """Materialize the same f32 negative-zero tensor returned by Repro.forward()."""
    if len(inputs) != 0:
        raise AssertionError(f"expected no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    _fill_negative_zero_kernel[(triton.cdiv(OUT_NUMEL, FILL_BLOCK),)](
        output,
        numel=OUT_NUMEL,
        block=FILL_BLOCK,
        num_warps=4,
    )
    if tuple(output.shape) != OUT_SHAPE or output.stride() != OUT_STRIDE:
        raise AssertionError(
            f"oracle layout mismatch: shape={tuple(output.shape)} stride={output.stride()}"
        )
    return output


def _has_exact_negative_zero_bits(tensor: torch.Tensor) -> bool:
    if tensor.dtype != torch.float32:
        return False
    return bool(tensor.view(torch.int32).eq(NEG_ZERO_I32).all().item())


def _check_exact_layout_and_bits(instance, inputs):
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == OUT_SHAPE
        and oracle_out.stride() == OUT_STRIDE
        and oracle_out.dtype == torch.float32
    )
    eager_bits_ok = _has_exact_negative_zero_bits(eager_out)
    oracle_bits_ok = _has_exact_negative_zero_bits(oracle_out)
    bitwise_match = torch.equal(eager_out.view(torch.int32), oracle_out.view(torch.int32))
    bits_ok = eager_bits_ok and oracle_bits_ok and bitwise_match
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} dtype={oracle_out.dtype})"
    )
    print(f"  output 0 negative-zero bits: {'PASS' if bits_ok else 'FAIL'}")
    return layout_ok and bits_ok


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
        ok = ok and _check_exact_layout_and_bits(instance, inputs)
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
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
