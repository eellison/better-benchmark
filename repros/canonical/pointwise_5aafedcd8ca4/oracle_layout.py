"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the exact zero-input `aten.full.default([1, 1, 256, 257], -inf, dtype=float16)` result by allocating the returned contiguous fp16 tensor and storing the packed `-inf` half bit pattern with one Triton fill kernel, whereas Inductor currently lowers the graph through its generic no-load pointwise constant-store kernel for the required fresh output; Inductor cannot do this today because codegen has no dedicated nonfinite fp16 constant-fill lowering that emits packed bitcast stores directly while bypassing generic elementwise scheduling machinery; the fix is NEW_PATTERN: add a specialized constant-full CUDA/Triton lowering for sign-sensitive and nonfinite half constants that writes the final layout with vectorized bit-pattern stores."""
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

OUT_SHAPE = (1, 1, 256, 257)
OUT_STRIDE = (65792, 65792, 257, 1)
OUT_NUMEL = 1 * 1 * 256 * 257
OUT_PACKED_NUMEL = OUT_NUMEL // 2
OUT_DTYPE = torch.float16
OUT_DEVICE = torch.device("cuda", 0)
FILL_BLOCK = 512
NEG_INF_F16X2_BITS = 0xFC00FC00
CLASSIFICATION = "NEW_PATTERN"


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _fill_neg_inf_f16x2_bits_kernel(
        out_packed_bits_ptr,
        neg_inf_packed_bits: tl.constexpr,
        numel: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < numel
        values = tl.full((block,), neg_inf_packed_bits, tl.uint32)
        tl.store(out_packed_bits_ptr + offsets, values, mask=mask)


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full no-input repro and return the exact fp16 `-inf` layout."""
    if inputs:
        raise ValueError(f"{REPRO_ID} expects no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    output = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=OUT_DEVICE,
        dtype=OUT_DTYPE,
    )
    output_packed = output.reshape(-1).view(torch.uint32)
    _fill_neg_inf_f16x2_bits_kernel[(triton.cdiv(OUT_PACKED_NUMEL, FILL_BLOCK),)](
        output_packed,
        neg_inf_packed_bits=NEG_INF_F16X2_BITS,
        numel=OUT_PACKED_NUMEL,
        block=FILL_BLOCK,
        num_warps=1,
        num_stages=1,
    )
    return output


def _check_exact_layout_and_bits(
    instance: torch.nn.Module,
    inputs: list[object] | tuple[object, ...],
) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == OUT_SHAPE
        and oracle_out.stride() == OUT_STRIDE
        and oracle_out.dtype is OUT_DTYPE
        and oracle_out.device.type == "cuda"
        and oracle_out.device.index == 0
        and oracle_out.storage_offset() == 0
    )
    bits_ok = torch.equal(eager_out.view(torch.uint16), oracle_out.view(torch.uint16))
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} "
        f"dtype={oracle_out.dtype} device={oracle_out.device})"
    )
    print(f"  output 0 bit_pattern: {'PASS' if bits_ok else 'FAIL'}")
    return layout_ok and bits_ok


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
        ok = _check_exact_layout_and_bits(instance, inputs) and ok
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
