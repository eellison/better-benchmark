"""Full-scope oracle for sum_sum_bc10a7742387 (VoVNet BN-backward tail).

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle consumes the same
eight original inputs as repro.py and returns the same contiguous
`[32, 1024, 7, 7]` tensor plus `[1024]` vector. It differs from Inductor by
computing the ReLU-backward `where(mm / 49)` producer, centered activation,
two sibling channel reductions, vector side output, and dependent BN-backward
full-tensor epilogue inside one per-channel Triton program, so the large
activation and gradient producer are not materialized and reread across generic
pointwise/reduction kernels. Inductor cannot do this today because its scheduler
does not form a full-scope multi-output reduction template that shares a masked
pointwise producer across sibling sums and then sinks the finalized sums into a
dependent full-tensor epilogue. The fix class is SCHEDULER_FUSION.
"""
from __future__ import annotations

import argparse
import importlib.util
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



REPRO_ID = "sum_sum_bc10a7742387"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 32
C = 1024
H = 7
W = 7
HW = H * W
TOTAL_REDUCE = N * HW
NUMEL = N * C * HW
AVGPOOL_SCALE = 1.0 / 49.0
BN_SCALE = 1.0 / float(TOTAL_REDUCE)

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]



@triton.jit
def _full_channel_bn_backward_kernel(
    mm_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    vector_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_REDUCE_: tl.constexpr,
    AVGPOOL_SCALE_: tl.constexpr,
    BN_SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < TOTAL_REDUCE_

    n = k // HW_
    hw = k - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    gamma = tl.load(gamma_ptr + c).to(tl.float32)
    beta = tl.load(beta_ptr + c).to(tl.float32)

    source = tl.load(activation_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    grad = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32) * AVGPOOL_SCALE_
    centered = source - mean
    relu_input = centered * invstd * gamma + beta
    where_self = tl.where(relu_input <= 0.0, 0.0, grad)
    where_self = tl.where(active, where_self, 0.0)

    sum1 = tl.sum(where_self, axis=0)
    sum2 = tl.sum(where_self * centered, axis=0)

    variance_term = sum2 * BN_SCALE_ * invstd * invstd
    mean_term = sum1 * BN_SCALE_
    affine_term = invstd * gamma
    result = (where_self - centered * variance_term - mean_term) * affine_term

    tl.store(out_ptr + offsets, result, mask=active)
    tl.store(vector_out_ptr + c, sum2 * invstd)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) and x.device.type != "cuda" else x for x in module.make_inputs())


def oracle_fused(
    mm: torch.Tensor,
    arg216_1: torch.Tensor,
    arg217_1: torch.Tensor,
    arg218_1: torch.Tensor,
    arg87_1: torch.Tensor,
    arg88_1: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm.shape == (N, C)
    assert arg216_1.shape == (N, C, H, W)
    assert arg217_1.shape == (1, C, 1, 1)
    assert arg218_1.shape == (1, C, 1, 1)
    assert arg87_1.shape == (C,)
    assert arg88_1.shape == (C,)
    assert mm.is_contiguous()
    assert arg216_1.is_contiguous()
    assert arg217_1.is_contiguous()
    assert arg218_1.is_contiguous()
    assert arg87_1.is_contiguous()
    assert arg88_1.is_contiguous()

    out = torch.empty((N, C, H, W), device=mm.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=mm.device, dtype=torch.float32)
    block_k = triton.next_power_of_2(TOTAL_REDUCE)
    _full_channel_bn_backward_kernel[(C,)](
        mm,
        arg216_1,
        arg217_1,
        arg218_1,
        arg87_1,
        arg88_1,
        out,
        vector_out,
        C_=C,
        HW_=HW,
        TOTAL_REDUCE_=TOTAL_REDUCE,
        AVGPOOL_SCALE_=AVGPOOL_SCALE,
        BN_SCALE_=BN_SCALE,
        BLOCK_K=block_k,
        num_warps=8,
    )
    return out, vector_out


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
