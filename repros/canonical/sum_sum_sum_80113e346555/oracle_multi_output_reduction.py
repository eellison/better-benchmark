"""
Oracle for sum_sum_sum_80113e346555

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the shared masked tensor once, keeps the 16-wide per-channel reductions in registers, and feeds both dependent grouped-reduction branches plus the vector reductions from those values, whereas Inductor materializes and rereads several view/reduction intermediates across a larger multi-output graph; Inductor cannot do this today because its algebraic elimination and reduction chaining do not preserve producer reductions across both downstream grouped branches and final cross-batch reductions; the required change is ALGEBRAIC_ELIMINATION: teach Inductor to recognize these dependent view-sum/mul/sum chains and codegen one fused multi-output reduction plan.
"""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_vectors_kernel(out_v1, out_sw, out_v2, BLOCK: tl.constexpr):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < 128
        z = tl.zeros((BLOCK,), tl.float32)
        tl.store(out_v1 + offsets, z, mask=mask)
        tl.store(out_sw + offsets, z, mask=mask)
        tl.store(out_v2 + offsets, z, mask=mask)

    @triton.jit
    def _multi_output_reduction_kernel(
        where_8,
        getitem_33,
        arg74_1,
        full,
        arg71_1,
        arg17_1,
        arg73_1,
        arg72_1,
        arg68_1,
        arg15_1,
        arg70_1,
        arg69_1,
        out_full_1,
        out_vec_1,
        out_sum_w,
        out_full_2,
        out_vec_2,
        BLOCK_K: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n = pid // 32
        g = pid - n * 32

        ks = tl.arange(0, BLOCK_K)
        ts = tl.arange(0, BLOCK_T)
        channels = g * 4 + ks
        offsets = n * 2048 + channels[:, None] * 16 + ts[None, :]

        a = tl.load(where_8 + offsets)
        b = tl.load(getitem_33 + offsets)
        pred = tl.load(arg74_1 + offsets) <= 0.0
        full_value = tl.load(full)
        w = tl.where(pred, full_value, a + b)

        x1 = tl.load(arg71_1 + offsets)
        x2 = tl.load(arg68_1 + offsets)
        sum_w = tl.sum(w, axis=1)
        sum_x1 = tl.sum(w * x1, axis=1)
        sum_x2 = tl.sum(w * x2, axis=1)

        gamma1 = tl.load(arg17_1 + channels)
        gamma2 = tl.load(arg15_1 + channels)
        p1 = tl.load(arg73_1 + n * 32 + g)
        q1 = tl.load(arg72_1 + n * 32 + g)
        p2 = tl.load(arg70_1 + n * 32 + g)
        q2 = tl.load(arg69_1 + n * 32 + g)

        grouped_w1 = tl.sum(sum_w * gamma1, axis=0)
        grouped_x1 = tl.sum(sum_x1 * gamma1, axis=0)
        grouped_w2 = tl.sum(sum_w * gamma2, axis=0)
        grouped_x2 = tl.sum(sum_x2 * gamma2, axis=0)

        p1_sq = p1 * p1
        p1_cu = p1_sq * p1
        m1 = (grouped_w1 * q1 - grouped_x1) * p1_cu * 0.015625
        bias1 = -m1 * q1 - grouped_w1 * p1 * 0.015625
        out1 = w * (p1 * gamma1)[:, None] + x1 * m1 + bias1
        tl.store(out_full_1 + offsets, out1)

        p2_sq = p2 * p2
        p2_cu = p2_sq * p2
        m2 = (grouped_w2 * q2 - grouped_x2) * p2_cu * 0.015625
        bias2 = -m2 * q2 - grouped_w2 * p2 * 0.015625
        out2 = w * (p2 * gamma2)[:, None] + x2 * m2 + bias2
        tl.store(out_full_2 + offsets, out2)

        tl.atomic_add(out_vec_1 + channels, (sum_x1 - sum_w * q1) * p1, sem="relaxed")
        tl.atomic_add(out_sum_w + channels, sum_w, sem="relaxed")
        tl.atomic_add(out_vec_2 + channels, (sum_x2 - sum_w * q2) * p2, sem="relaxed")


@oracle_impl(hardware="H100", shapes="(T([64, 128, 4, 4], f32), T([64, 128, 4, 4], f32), T([64, 128, 4, 4], f32), T([], f32), T([64, 128, 4, 4], f32), T([128], f32), T([64, 32], f32), T([64, 32], f32), T([64, 128, 4, 4], f32), T([128], f32), T([64, 32], f32), T([64, 32], f32), S([64, 128, 16]), S([64, 128, 16]), S([64, 32, 4]), S([64, 32, 4]), S([1, 32, 4]), S([64, 32, 4, 16]), S([64, 32, 4, 16]), S([64, 128, 4, 4]), S([64, 32, 4]), S([64, 32, 4]), S([128]), S([64, 128, 16]), S([64, 32, 4]), S([64, 32, 4]), S([1, 32, 4]), S([64, 32, 4, 16]), S([64, 128, 4, 4]), S([64, 32, 4]), S([128]))")
def oracle_forward(inputs):
    """Run the full-scope Triton oracle for Repro.forward()."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        where_8,
        getitem_33,
        arg74_1,
        full,
        arg71_1,
        arg17_1,
        arg73_1,
        arg72_1,
        arg68_1,
        arg15_1,
        arg70_1,
        arg69_1,
        *_shape_params,
    ) = inputs

    out_full_1 = torch.empty_like(where_8)
    out_vec_1 = torch.empty((128,), device=where_8.device, dtype=torch.float32)
    out_sum_w = torch.empty((128,), device=where_8.device, dtype=torch.float32)
    out_full_2 = torch.empty_like(where_8)
    out_vec_2 = torch.empty((128,), device=where_8.device, dtype=torch.float32)

    _zero_vectors_kernel[(1,)](out_vec_1, out_sum_w, out_vec_2, BLOCK=128)
    _multi_output_reduction_kernel[(64 * 32,)](
        where_8,
        getitem_33,
        arg74_1,
        full,
        arg71_1,
        arg17_1,
        arg73_1,
        arg72_1,
        arg68_1,
        arg15_1,
        arg70_1,
        arg69_1,
        out_full_1,
        out_vec_1,
        out_sum_w,
        out_full_2,
        out_vec_2,
        BLOCK_K=4,
        BLOCK_T=16,
        num_warps=2,
    )
    return (out_full_1, out_vec_1, out_sum_w, out_full_2, out_vec_2, out_sum_w)


def main():
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
