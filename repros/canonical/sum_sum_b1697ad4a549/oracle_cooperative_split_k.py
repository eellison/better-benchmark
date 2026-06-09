"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full MobileViT SiLU-gradient plus batch-norm-backward return tuple by cooperatively split-K reducing the two per-channel summaries over the shared `(N, H, W)` domain from the original repro inputs and then recomputing the fused producer in one epilogue that writes both the contiguous input-gradient tensor and `sum2 * rsqrt` vector, whereas Inductor currently schedules the decomposed slice, SiLU derivative producer, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogue as ordinary reduction/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible small-output channel reductions with a finalized-summary tensor epilogue while preserving the target output layout; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible channel reductions across the reduced N/H/W dimension, combine the partial summaries, and fuse the downstream tensor and side-vector epilogues with the SiLU-gradient producer."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

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



REPRO_ID = "sum_sum_b1697ad4a549"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
IN_C = 192
C = 96
SLICE_C_START = 96
H = 32
W = 32
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
SCALE = 7.62939453125e-06



@triton.jit
def _silu_bn_dual_reduce_kernel(
    upstream_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    sum_grad_ptr,
    sum_grad_centered_ptr,
    IN_C_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_C_START_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    upstream_offsets = n * (IN_C_ * HW_) + (c + SLICE_C_START_) * HW_ + hw
    activation_offsets = n * (C_ * HW_) + c * HW_ + hw

    upstream = tl.load(upstream_ptr + upstream_offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + activation_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    centered = activation - mean
    z = centered * invstd * weight + bias
    sigmoid = 1.0 / (tl.exp(-z) + 1.0)
    grad_silu = upstream * sigmoid * (z * (1.0 - sigmoid) + 1.0)

    tl.atomic_add(sum_grad_ptr + c, tl.sum(grad_silu, axis=0), sem="relaxed")
    tl.atomic_add(
        sum_grad_centered_ptr + c,
        tl.sum(grad_silu * centered, axis=0),
        sem="relaxed",
    )


@triton.jit
def _silu_bn_epilogue_kernel(
    upstream_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    sum_grad_ptr,
    sum_grad_centered_ptr,
    out_grad_input_ptr,
    out_weight_grad_ptr,
    IN_C_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_C_START_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)

    upstream_offsets = n * (IN_C_ * HW_) + (c + SLICE_C_START_) * HW_ + hw
    upstream = tl.load(upstream_ptr + upstream_offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_grad = tl.load(sum_grad_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_grad_centered = tl.load(sum_grad_centered_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = activation - mean
    z = centered * invstd * weight + bias
    sigmoid = 1.0 / (tl.exp(-z) + 1.0)
    grad_silu = upstream * sigmoid * (z * (1.0 - sigmoid) + 1.0)

    mean_term = sum_grad * SCALE_
    variance_term = sum_grad_centered * SCALE_ * invstd * invstd
    grad_input = (grad_silu - centered * variance_term - mean_term) * (invstd * weight)

    tl.store(out_grad_input_ptr + offsets, grad_input, mask=mask)

    is_first_element_for_channel = mask & (n == 0) & (hw == 0)
    tl.store(
        out_weight_grad_ptr + c,
        sum_grad_centered * invstd,
        mask=is_first_element_for_channel,
    )


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def oracle_full(
    getitem_73: torch.Tensor,
    arg242_1: torch.Tensor,
    arg243_1: torch.Tensor,
    arg244_1: torch.Tensor,
    arg62_1: torch.Tensor,
    arg63_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert getitem_73.shape == (N, IN_C, H, W)
    assert arg242_1.shape == (N, C, H, W)
    assert arg243_1.shape == (1, C, 1, 1)
    assert arg244_1.shape == (1, C, 1, 1)
    assert arg62_1.shape == (C,)
    assert arg63_1.shape == (C,)
    assert getitem_73.is_contiguous()
    assert arg242_1.is_contiguous()
    assert arg243_1.is_contiguous()
    assert arg244_1.is_contiguous()
    assert arg62_1.is_contiguous()
    assert arg63_1.is_contiguous()

    device = arg242_1.device
    mean = arg243_1.reshape(C)
    invstd = arg244_1.reshape(C)
    sum_grad = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_grad_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    block_k = 2048
    _silu_bn_dual_reduce_kernel[(C, triton.cdiv(TOTAL_SPATIAL, block_k))](
        getitem_73,
        arg242_1,
        mean,
        invstd,
        arg62_1,
        arg63_1,
        sum_grad,
        sum_grad_centered,
        IN_C_=IN_C,
        C_=C,
        SLICE_C_START_=SLICE_C_START,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=8,
    )

    out_grad_input = torch.empty_like(arg242_1)
    out_weight_grad = torch.empty((C,), device=device, dtype=torch.float32)

    block_elems = 256
    _silu_bn_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        getitem_73,
        arg242_1,
        mean,
        invstd,
        arg62_1,
        arg63_1,
        sum_grad,
        sum_grad_centered,
        out_grad_input,
        out_weight_grad,
        IN_C_=IN_C,
        C_=C,
        SLICE_C_START_=SLICE_C_START,
        HW_=HW,
        NUMEL_=NUMEL,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out_grad_input, out_weight_grad


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _compile_default(model: torch.nn.Module, inputs: tuple[object, ...]):
    import torch._dynamo

    torch._dynamo.reset()
    compiled = torch.compile(model)
    for _ in range(3):
        compiled(*inputs)
    torch.cuda.synchronize()
    return compiled


@oracle_impl(hardware="H100", shapes="(T([128, 192, 32, 32], f32), T([128, 96, 32, 32], f32), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32))")
def oracle_forward(inputs):
    return oracle_full(*inputs)


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
