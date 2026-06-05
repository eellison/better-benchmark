"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full NFNet GELU-backward channel reduction by folding the mm view/expand/divide, GELU-derivative pointwise chain, and [0, 2, 3] sum into one Triton kernel that directly loads mm[n, c], whereas Inductor currently lowers the same graph through its generic reduction scheduling instead of selecting this direct broadcast-producer-in-reduction layout; Inductor cannot do this today because the scheduler/codegen does not canonicalize a broadcasted [N, C, 1, 1] producer as a scalar-per-spatial-position input inside the channel-reduction loop; the fix is SCHEDULER_FUSION: teach reduction fusion/indexing to inline view/expand scalar producers and emit direct strided source loads inside the generated reduction kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_27f8a4b9ab09"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 3072
H = 6
W = 6
HW = H * W
GAMMA_OVER_HW = 1.7015043497085571 / HW
RSQRT2 = 0.7071067811865476
RSQRT2PI = 0.3989422804014327



@triton.jit
def _gelu_grad_flat_channel_kernel(
    mm_ptr,
    x_ptr,
    out_ptr,
    x_stride_n: tl.constexpr,
    x_stride_c: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_w: tl.constexpr,
    C_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    N_HW_: tl.constexpr,
    GAMMA_OVER_HW_: tl.constexpr,
    RSQRT2_: tl.constexpr,
    RSQRT2PI_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < N_HW_
    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_

    x_offsets = n * x_stride_n + c * x_stride_c + h * x_stride_h + w * x_stride_w
    mm_offsets = n * C_ + c
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mm_values = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)

    erf_plus_one = tl.math.erf(x * RSQRT2_) + 1.0
    cdf = 0.5 * erf_plus_one
    pdf_term = x * tl.exp(-0.5 * x * x) * RSQRT2PI_
    value = mm_values * GAMMA_OVER_HW_ * (cdf + pdf_term)
    tl.store(out_ptr + c, tl.sum(tl.where(active, value, 0.0), axis=0))


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def oracle_fused(
    mm: torch.Tensor,
    arg413_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    assert mm.shape == (N, C)
    assert arg413_1.shape == (N, C, H, W)
    assert mm.is_contiguous()

    out = torch.empty((C,), device=mm.device, dtype=torch.float32)
    _gelu_grad_flat_channel_kernel[(C,)](
        mm,
        arg413_1,
        out,
        x_stride_n=arg413_1.stride(0),
        x_stride_c=arg413_1.stride(1),
        x_stride_h=arg413_1.stride(2),
        x_stride_w=arg413_1.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N * HW,
        GAMMA_OVER_HW_=GAMMA_OVER_HW,
        RSQRT2_=RSQRT2,
        RSQRT2PI_=RSQRT2PI,
        BLOCK_K=triton.next_power_of_2(N * HW),
        num_warps=4,
    )
    return out


def oracle_forward(inputs):
    return oracle_fused(*inputs)


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
