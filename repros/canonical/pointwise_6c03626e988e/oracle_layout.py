"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the exact zero-input `aten.full([32, 1, 1, 512], -0.0, dtype=float32, device=cuda:0)` result with one Triton kernel that stores the negative-zero `0x80000000` bit pattern into the final contiguous layout, whereas Inductor currently lowers the no-input constant tensor through its generic pointwise full path and pays the normal wrapper/kernel overhead for a tiny fixed fill; Inductor cannot do this today because there is no dedicated negative-zero constant-fill lowering that preserves sign bits while bypassing generic elementwise scheduling machinery; the fix is NEW_PATTERN: add a specialized CUDA/Triton lowering for sign-sensitive constant fills that emits bitcast stores directly into the required output layout."""
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

OUT_SHAPE = (32, 1, 1, 512)
OUT_STRIDE = (512, 512, 512, 1)
OUT_NUMEL = 32 * 1 * 1 * 512
OUT_DTYPE = torch.float32
OUT_DEVICE = torch.device("cuda", 0)
NEG_ZERO_I32 = torch.iinfo(torch.int32).min


from oracle_harness import (  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
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


if triton is not None:

    @triton.jit
    def _fill_negative_zero_kernel(
        out_ptr,
        total: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < total
        bits = tl.full((block,), -2147483648, tl.int32)
        values = bits.to(tl.float32, bitcast=True)
        tl.store(out_ptr + offsets, values, mask=mask)


def oracle_forward(inputs):
    """Run the full Repro.forward scope with an exact negative-zero fill."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 0:
        raise AssertionError(f"expected no inputs, got {len(inputs)}")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=OUT_DEVICE,
        dtype=OUT_DTYPE,
    )
    block = 256
    _fill_negative_zero_kernel[(triton.cdiv(OUT_NUMEL, block),)](
        out,
        total=OUT_NUMEL,
        block=block,
        num_warps=8,
    )
    return out


def _has_exact_negative_zero_bits(tensor: torch.Tensor) -> bool:
    if tensor.dtype != torch.float32:
        return False
    return bool(tensor.view(torch.int32).eq(NEG_ZERO_I32).all().item())


def _check_layout_and_bits(instance, inputs) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        layout_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(layout_out.shape) == tuple(eager_out.shape) == OUT_SHAPE
        and layout_out.stride() == eager_out.stride() == OUT_STRIDE
        and layout_out.dtype == eager_out.dtype == OUT_DTYPE
        and layout_out.device == eager_out.device == OUT_DEVICE
        and layout_out.storage_offset() == eager_out.storage_offset() == 0
    )
    eager_bits_ok = _has_exact_negative_zero_bits(eager_out)
    layout_bits_ok = _has_exact_negative_zero_bits(layout_out)
    bitwise_match = bool(
        torch.equal(eager_out.view(torch.int32), layout_out.view(torch.int32))
    )
    bits_ok = eager_bits_ok and layout_bits_ok and bitwise_match

    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(layout_out.shape)} stride={layout_out.stride()} "
        f"dtype={layout_out.dtype} device={layout_out.device})"
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
        ok = _check_layout_and_bits(instance, inputs) and ok
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
