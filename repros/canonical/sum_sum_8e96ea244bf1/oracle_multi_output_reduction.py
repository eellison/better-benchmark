"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete `sum_sum_8e96ea244bf1` NFNet weight-standardization backward graph in one Triton multi-output reduction, loading the two channels-last `[16,3,3,3]` inputs through their source strides, computing both per-output-channel reductions over the 27 weight elements, applying all scalar and tensor epilogue math, and returning both `mul_tensor_10` and `reshape_default_4`, whereas Inductor lowers the clone/reshape views, sibling reductions, broadcast arithmetic, and final tensor/vector stores as separate scheduled regions around materialized intermediates; Inductor cannot do this today because its scheduler does not form a full-scope multi-output reduction for tiny strided channels-last producers whose reductions feed both a vector output and a dependent full-tensor epilogue; the required Inductor fix is SCHEDULER_FUSION: teach the scheduler/codegen to fuse compatible small reductions with their dependent broadcast epilogues and multiple returned stores when the producer is a simple memory-format clone/view."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl

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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

OUT_CHANNELS = 16
IN_CHANNELS = 3
KERNEL_H = 3
KERNEL_W = 3
REDUCE_N = IN_CHANNELS * KERNEL_H * KERNEL_W
BLOCK_K = 32
INV_REDUCE_N = 1.0 / REDUCE_N
WEIGHT_SCALE = 0.19245008972987526


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _nfnet_weight_standardization_bwd_kernel(
    getitem_ptr,
    primals2_ptr,
    mean_ptr,
    invstd_ptr,
    gain_ptr,
    out_gain_ptr,
    out_weight_ptr,
    getitem_s0: tl.constexpr,
    getitem_s1: tl.constexpr,
    getitem_s2: tl.constexpr,
    getitem_s3: tl.constexpr,
    primals2_s0: tl.constexpr,
    primals2_s1: tl.constexpr,
    primals2_s2: tl.constexpr,
    primals2_s3: tl.constexpr,
    REDUCE_N_: tl.constexpr,
    KERNEL_HW_: tl.constexpr,
    KERNEL_W_: tl.constexpr,
    INV_REDUCE_N_: tl.constexpr,
    WEIGHT_SCALE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    oc = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    mask = k < REDUCE_N_

    ic = k // KERNEL_HW_
    rem = k - ic * KERNEL_HW_
    kh = rem // KERNEL_W_
    kw = rem - kh * KERNEL_W_

    grad_offsets = (
        oc * getitem_s0
        + ic * getitem_s1
        + kh * getitem_s2
        + kw * getitem_s3
    )
    weight_offsets = (
        oc * primals2_s0
        + ic * primals2_s1
        + kh * primals2_s2
        + kw * primals2_s3
    )

    grad = tl.load(getitem_ptr + grad_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(primals2_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + oc).to(tl.float32)
    invstd = tl.load(invstd_ptr + oc).to(tl.float32)
    gain = tl.load(gain_ptr + oc).to(tl.float32)

    centered = weight - mean
    sum_grad = tl.sum(tl.where(mask, grad, 0.0), axis=0)
    sum_grad_centered = tl.sum(tl.where(mask, grad * centered, 0.0), axis=0)

    mean_grad = sum_grad * INV_REDUCE_N_
    var_term = sum_grad_centered * INV_REDUCE_N_ * invstd * invstd
    output_scale = invstd * gain * WEIGHT_SCALE_
    tensor_out = (grad - centered * var_term - mean_grad) * output_scale
    gain_out = sum_grad_centered * invstd * WEIGHT_SCALE_

    tl.store(out_gain_ptr + oc, gain_out)
    tl.store(out_weight_ptr + oc * REDUCE_N_ + k, tensor_out, mask=mask)


def _check_expected_scope(inputs: tuple[Any, ...] | list[Any]) -> None:
    (
        getitem_355,
        primals_2,
        unsqueeze_506,
        squeeze_1,
        primals_3,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
    ) = inputs
    assert getitem_355.shape == (16, 3, 3, 3)
    assert primals_2.shape == (16, 3, 3, 3)
    assert unsqueeze_506.shape == (1, 16, 1)
    assert squeeze_1.shape == (16,)
    assert primals_3.shape == (16, 1, 1, 1)
    assert list(shape_param_0) == [1, 16, 27]
    assert list(shape_param_1) == [1, 16, 27]
    assert list(shape_param_2) == [16, 1, 1, 1]
    assert list(shape_param_3) == [16, 3, 3, 3]
    assert getitem_355.dtype == torch.float32
    assert primals_2.dtype == torch.float32
    assert unsqueeze_506.dtype == torch.float32
    assert squeeze_1.dtype == torch.float32
    assert primals_3.dtype == torch.float32


@oracle_impl(hardware="H100", shapes="(T([16, 3, 3, 3], f32, stride=(27, 1, 9, 3)), T([16, 3, 3, 3], f32, stride=(27, 1, 9, 3)), T([1, 16, 1], f32), T([16], f32), T([16, 1, 1, 1], f32), S([1, 16, 27]), S([1, 16, 27]), S([16, 1, 1, 1]), S([16, 3, 3, 3]))")
def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full-scope Triton oracle for Repro.forward()."""
    _check_expected_scope(inputs)
    (
        getitem_355,
        primals_2,
        unsqueeze_506,
        squeeze_1,
        primals_3,
        *_shape_params,
    ) = inputs

    if getitem_355.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    out_gain = torch.empty((OUT_CHANNELS, 1, 1, 1), device=getitem_355.device, dtype=torch.float32)
    out_weight = torch.empty(
        (OUT_CHANNELS, IN_CHANNELS, KERNEL_H, KERNEL_W),
        device=getitem_355.device,
        dtype=torch.float32,
    )

    _nfnet_weight_standardization_bwd_kernel[(OUT_CHANNELS,)](
        getitem_355,
        primals_2,
        unsqueeze_506,
        squeeze_1,
        primals_3,
        out_gain,
        out_weight,
        getitem_355.stride(0),
        getitem_355.stride(1),
        getitem_355.stride(2),
        getitem_355.stride(3),
        primals_2.stride(0),
        primals_2.stride(1),
        primals_2.stride(2),
        primals_2.stride(3),
        REDUCE_N_=REDUCE_N,
        KERNEL_HW_=KERNEL_H * KERNEL_W,
        KERNEL_W_=KERNEL_W,
        INV_REDUCE_N_=INV_REDUCE_N,
        WEIGHT_SCALE_=WEIGHT_SCALE,
        BLOCK=BLOCK_K,
        num_warps=1,
    )
    return out_gain, out_weight


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
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Disable auto-detection and skipping of stochastic outputs")
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
                        f"WARNING: oracle is slower than compile for {result['repro_id']} "
                        f"(ratio={result['ratio']:.3f}x)"
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
