"""
Oracle for sum_sum_8d73a4b5bade.

Gap diagnosis: Classification: SCHEDULER_FUSION. This oracle computes the complete NFNet small weight-standardization backward tail in one Triton channel kernel, sharing the two per-channel reductions over `[16, 3, 3, 3]` and immediately applying all dependent pointwise epilogue math for both returned outputs, whereas Inductor currently schedules the clone/view inputs, sibling sums, dependent scalar broadcasts, and full tensor epilogue as separate generic reduction/pointwise regions over materialized intermediates; Inductor cannot do this today because its scheduler does not form a tiny multi-output reduction region whose reduction results feed both a vector side output and a full-tensor side output in the same kernel, and the fix is SCHEDULER_FUSION: teach scheduler/codegen to fuse compatible sibling reductions with their dependent epilogues when all live consumers are inside the captured graph.
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_sum_8d73a4b5bade"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

CHANNELS = 16
REDUCE_ELEMS = 27
INV_REDUCE_ELEMS = 1.0 / REDUCE_ELEMS
WS_SCALE = 0.19245008972987526


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _nfnet_ws_backward_kernel(
        grad_ptr,
        weight_ptr,
        mean_ptr,
        invstd_ptr,
        upstream_ptr,
        out_scale_ptr,
        out_grad_ptr,
        REDUCE_ELEMS_: tl.constexpr,
        INV_REDUCE_ELEMS_: tl.constexpr,
        WS_SCALE_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0)
        k = tl.arange(0, BLOCK_K)
        mask = k < REDUCE_ELEMS_
        offsets = c * REDUCE_ELEMS_ + k

        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        upstream = tl.load(upstream_ptr + c).to(tl.float32)

        centered = weight - mean
        sum_grad = tl.sum(tl.where(mask, grad, 0.0), axis=0)
        sum_grad_centered = tl.sum(tl.where(mask, grad * centered, 0.0), axis=0)

        mean_grad = sum_grad * INV_REDUCE_ELEMS_
        variance_term = sum_grad_centered * INV_REDUCE_ELEMS_ * invstd * invstd
        output_scale = upstream * WS_SCALE_
        grad_scale = invstd * output_scale

        out_grad = (grad - centered * variance_term - mean_grad) * grad_scale
        out_channel_scale = sum_grad_centered * invstd * WS_SCALE_

        tl.store(out_scale_ptr + c, out_channel_scale)
        tl.store(out_grad_ptr + offsets, out_grad, mask=mask)


def oracle_forward(inputs):
    """Run the full-scope multi-output reduction oracle."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    getitem_241, arg0_1, arg478_1, arg152_1, arg1_1, *_shape_params = inputs
    assert getitem_241.shape == (CHANNELS, 3, 3, 3)
    assert arg0_1.shape == (CHANNELS, 3, 3, 3)
    assert arg478_1.shape == (1, CHANNELS, 1)
    assert arg152_1.shape == (CHANNELS,)
    assert arg1_1.shape == (CHANNELS, 1, 1, 1)

    grad = getitem_241.contiguous()
    weight = arg0_1.contiguous()
    mean = arg478_1.reshape(CHANNELS)
    invstd = arg152_1.contiguous()
    upstream = arg1_1.reshape(CHANNELS).contiguous()

    mul_tensor_10 = torch.empty((CHANNELS, 1, 1, 1), device=grad.device, dtype=grad.dtype)
    view_default_4 = torch.empty((CHANNELS, 3, 3, 3), device=grad.device, dtype=grad.dtype)

    _nfnet_ws_backward_kernel[(CHANNELS,)](
        grad,
        weight,
        mean,
        invstd,
        upstream,
        mul_tensor_10,
        view_default_4,
        REDUCE_ELEMS_=REDUCE_ELEMS,
        INV_REDUCE_ELEMS_=INV_REDUCE_ELEMS,
        WS_SCALE_=WS_SCALE,
        BLOCK_K=32,
        num_warps=1,
    )
    return mul_tensor_10, view_default_4


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
