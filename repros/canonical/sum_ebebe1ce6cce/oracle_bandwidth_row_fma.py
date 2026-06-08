"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Repro.forward scope in one Triton row-reduction kernel, including the [512,128,128] to [32,16,128,128] view semantics, fp32 product, last-dimension sum, explicit prims.fma-equivalent `tl.fma`, sliced broadcast bool mask, scalar fill, and final [512,128,128] contiguous-stride output, whereas Inductor already emits a similar single fused persistent reduction for this graph; Inductor cannot substantially reduce the remaining work without changing the algebra or observable rounding because the kernel must read the two fp32 inputs, accumulate each row, and write every fp32 output element; the fix is BANDWIDTH_BOUND: keep this pattern classified as an at-floor fused row reduction unless a numerics-preserving memory-traffic reduction is found."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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

BATCH = 32
HEADS = 16
SEQ = 128
TOTAL_ROWS = BATCH * HEADS * SEQ
MASK_STRIDE = 2048
ROWS_PER_PROGRAM = 2


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _row_fma_mask_kernel(
        in_ptr0,
        in_ptr1,
        in_ptr2,
        in_ptr3,
        out_ptr1,
        X: tl.constexpr,
    ):
        xindex = tl.program_id(0) * X + tl.arange(0, X)[:, None]
        r0_1 = tl.arange(0, 128)[None, :]
        x2 = xindex % 128
        tmp0 = tl.load(in_ptr0 + xindex * 128 + r0_1, None, eviction_policy="evict_first")
        tmp1 = tl.load(in_ptr1 + xindex * 128 + r0_1, None, eviction_policy="evict_first")
        tmp6 = tl.load(in_ptr2 + x2 * 2048 + r0_1, None, eviction_policy="evict_last").to(tl.int1)
        tmp9 = tl.load(in_ptr3 + 0)
        tmp10 = tl.broadcast_to(tmp9, [1, 1])
        tmp2 = tmp0 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [X, 128])
        tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
        tmp7 = -tmp1
        tmp8 = tl.fma(tmp7, tmp5, tmp2)
        tmp11 = tl.where(tmp6, tmp8, tmp10)
        tl.store(out_ptr1 + xindex * 128 + r0_1, tmp11, None)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("triton is not available")

    if len(inputs) != 6:
        raise ValueError(f"expected 6 repro inputs, got {len(inputs)}")

    bmm_93, arg225_1, arg6_1, full_1, shape0, shape1 = inputs
    if not isinstance(shape0, (list, tuple)) or tuple(shape0) != (BATCH, HEADS, SEQ, SEQ):
        raise ValueError(f"unexpected first shape parameter: {shape0}")
    if not isinstance(shape1, (list, tuple)) or tuple(shape1) != (BATCH * HEADS, SEQ, SEQ):
        raise ValueError(f"unexpected second shape parameter: {shape1}")
    if bmm_93.shape != (BATCH * HEADS, SEQ, SEQ) or bmm_93.stride() != (SEQ * SEQ, SEQ, 1):
        raise ValueError(f"unexpected bmm_93 shape/stride: {tuple(bmm_93.shape)} {bmm_93.stride()}")
    if arg225_1.shape != (BATCH, HEADS, SEQ, SEQ) or arg225_1.stride() != (HEADS * SEQ * SEQ, SEQ * SEQ, SEQ, 1):
        raise ValueError(f"unexpected arg225_1 shape/stride: {tuple(arg225_1.shape)} {arg225_1.stride()}")
    if arg6_1.shape != (1, 1, 2048, 2048) or arg6_1.stride() != (2048 * 2048, 2048 * 2048, 2048, 1):
        raise ValueError(f"unexpected arg6_1 shape/stride: {tuple(arg6_1.shape)} {arg6_1.stride()}")
    if full_1.shape != ():
        raise ValueError(f"unexpected full_1 shape: {tuple(full_1.shape)}")
    if bmm_93.dtype is not torch.float32 or arg225_1.dtype is not torch.float32 or full_1.dtype is not torch.float32:
        raise ValueError("expected fp32 bmm_93, arg225_1, and full_1")
    if arg6_1.dtype is not torch.bool:
        raise ValueError("expected bool arg6_1")
    if bmm_93.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    return bmm_93, arg225_1, arg6_1, full_1


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    bmm_93, arg225_1, arg6_1, full_1 = _validate_inputs(inputs)
    out = torch.empty_strided(
        (BATCH * HEADS, SEQ, SEQ),
        (SEQ * SEQ, SEQ, 1),
        device=bmm_93.device,
        dtype=torch.float32,
    )
    _row_fma_mask_kernel[(triton.cdiv(TOTAL_ROWS, ROWS_PER_PROGRAM),)](
        bmm_93,
        arg225_1,
        arg6_1,
        full_1,
        out,
        X=ROWS_PER_PROGRAM,
    )
    return out


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
