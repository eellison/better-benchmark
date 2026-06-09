"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ALBERT dual-row-sum fragment with a row-resident Triton kernel and returns the exact final permuted-view layout, while Inductor already emits a single fused f32 dual-reduction/epilogue kernel for the same full scope and measures within the harness floor; Inductor cannot materially improve this repro through local scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, or recompute fusion because the required producer reads, two f32 row reductions, full epilogue, and dense output store dominate; the fix is BANDWIDTH_BOUND: record this as at-floor unless broader memory/reduction codegen improvements move both paths."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _row_resident_dual_sum_kernel(
        mm_118_ptr,
        mul_287_ptr,
        mm_120_ptr,
        mm_122_ptr,
        arg12_ptr,
        arg34_ptr,
        arg174_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        offsets = row * N + cols

        tmp0 = tl.load(mul_287_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tmp1 = tl.load(mm_118_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.load(mm_120_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.load(mm_122_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tmp6 = tmp4 + tmp5
        tmp7 = tl.load(arg12_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tmp8 = tmp6 * tmp7

        tmp12 = tl.load(arg34_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum0_terms = tl.where(mask, tmp8, 0.0)
        tmp13 = tmp8 * tmp12
        sum1_terms = tl.where(mask, tmp13, 0.0)
        tmp10 = tl.sum(sum0_terms, axis=0)
        tmp15 = tl.sum(sum1_terms, axis=0)

        tmp17 = tl.load(arg174_ptr + row).to(tl.float32)
        tmp19 = tl.full((BLOCK_N,), 4096.0, tl.float32)
        tmp20 = tmp8 * tmp19
        tmp21 = tmp20 - tmp10
        tmp23 = tmp12 * tmp15
        tmp24 = tmp21 - tmp23
        tmp25 = tmp17 * tmp24
        tl.store(out_ptr + offsets, tmp25, mask=mask)


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> tuple[int, int]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    if len(inputs) != 11:
        raise ValueError(f"expected 11 inputs, got {len(inputs)}")

    mm_118, mul_287, mm_120, mm_122, arg12_1, arg34_1, arg174_1 = inputs[:7]
    shape0, shape1, shape2, shape3 = inputs[7:]
    if not all(isinstance(x, torch.Tensor) for x in inputs[:7]):
        raise TypeError("the first seven repro inputs must be tensors")

    device = mm_118.device
    tensor_inputs = (mm_118, mul_287, mm_120, mm_122, arg12_1, arg34_1, arg174_1)
    if device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if any(t.device != device for t in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if any(t.dtype != torch.float32 for t in tensor_inputs):
        raise ValueError("this oracle expects the captured f32 repro inputs")
    if not all(t.is_contiguous() for t in tensor_inputs):
        raise ValueError("this oracle expects the captured contiguous input layouts")

    if list(shape0) != [8, 512, 4096] or list(shape1) != [8, 512, 4096]:
        raise ValueError(f"unexpected view shapes: {shape0}, {shape1}")
    if list(shape2) != [8, 512, 4096] or list(shape3) != [4096, 4096]:
        raise ValueError(f"unexpected view shapes: {shape2}, {shape3}")
    if tuple(mm_118.shape) != (4096, 4096) or tuple(mm_120.shape) != (4096, 4096):
        raise ValueError("unexpected matrix input shape")
    if tuple(mm_122.shape) != (4096, 4096) or tuple(mul_287.shape) != (8, 512, 4096):
        raise ValueError("unexpected producer input shape")
    if tuple(arg12_1.shape) != (4096,) or tuple(arg34_1.shape) != (8, 512, 4096):
        raise ValueError("unexpected broadcast input shape")
    if tuple(arg174_1.shape) != (8, 512, 1):
        raise ValueError("unexpected final scale input shape")

    return 4096, 4096


@oracle_impl(hardware="H100", shapes="(T([4096, 4096], f32), T([8, 512, 4096], f32), T([4096, 4096], f32), T([4096, 4096], f32), T([4096], f32), T([8, 512, 4096], f32), T([8, 512, 1], f32), S([8, 512, 4096]), S([8, 512, 4096]), S([8, 512, 4096]), S([4096, 4096]))")
def oracle_forward(inputs):
    """Compute the exact same full output as Repro()(*make_inputs())."""
    m, n = _validate_inputs(inputs)
    mm_118, mul_287, mm_120, mm_122, arg12_1, arg34_1, arg174_1 = inputs[:7]

    out_base = torch.empty_strided((m, n), (n, 1), device=mm_118.device, dtype=torch.float32)
    _row_resident_dual_sum_kernel[(m,)](
        mm_118,
        mul_287,
        mm_120,
        mm_122,
        arg12_1,
        arg34_1,
        arg174_1,
        out_base,
        N=n,
        BLOCK_N=4096,
        num_warps=8,
    )
    return torch.ops.aten.permute.default(out_base, [1, 0])


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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
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
