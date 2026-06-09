"""
Canonical oracle for sum_sum_b7f94adef30f (GhostNet BN-backward reductions).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle differs from Inductor by taking the BN-backward pointwise producer as the prepared input and reducing it with a split-K dual-accumulator Triton kernel, so each channel is reduced by many row/spatial tiles that contribute partials for both `sum(add_tensor_1)` and `sum(add_tensor_1 * centered)` before one fused epilogue pass writes the two final outputs; Inductor cannot do this today because its reduction splitting heuristic rejects this small-output, large-reduction BN-backward shape under the no-split threshold and then schedules the sibling reductions/epilogue with too little reduction-dimension parallelism; the fix is COOPERATIVE_SPLIT_K support for compatible multi-output reductions, allowing the scheduler to split the reduction dimension and coordinate partial or atomic accumulation across tiles while preserving the dependent epilogue.
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_b7f94adef30f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C = 72
H = 28
W = 28
HW = H * W
NUMEL = N * C * HW
TOTAL_SPATIAL = N * HW
INV_HW = 1.0 / HW
SCALE = 2.4912308673469386e-06



@triton.jit
def _dual_reduce_split_k_kernel(
    producer_ptr,
    centered_ptr,
    sum1_ptr,
    sum2_ptr,
    total_spatial: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < total_spatial

    n = k // HW_
    hw = k - n * HW_
    full_offsets = n * (C_ * HW_) + c * HW_ + hw
    add_tensor_1 = tl.load(producer_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    centered = tl.load(centered_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)

    tl.atomic_add(sum1_ptr + c, tl.sum(add_tensor_1, axis=0), sem="relaxed")
    tl.atomic_add(sum2_ptr + c, tl.sum(add_tensor_1 * centered, axis=0), sem="relaxed")

@triton.jit
def _bn_backward_epilogue_kernel(
    producer_ptr,
    centered_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    NUMEL_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < NUMEL_

    c = (offsets // HW_) % C_
    add_tensor_1 = tl.load(producer_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered = tl.load(centered_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sum1 = tl.load(sum1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=mask, other=0.0).to(tl.float32)

    mean_term = sum1 * SCALE_
    variance_term = sum2 * SCALE_ * rsqrt * rsqrt
    out = (add_tensor_1 - centered * variance_term - mean_term) * (rsqrt * affine_weight)
    tl.store(out_ptr + offsets, out, mask=mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        arg258_1,
        getitem_219,
        getitem_225,
        arg254_1,
        arg253_1,
        arg255_1,
        arg42_1,
        _shape_param_0,
    ) = inputs
    del _shape_param_0

    return (
        (
            getitem_219
            * (torch.clamp(arg258_1 + 3.0, min=0.0, max=6.0) * 0.16666666666666666)
            + getitem_225 * INV_HW
        ).contiguous(),
        (arg253_1 - arg254_1.reshape(1, C, 1, 1)).contiguous(),
        arg255_1.reshape(C).contiguous(),
        arg42_1.contiguous(),
    )


def oracle_fused(
    producer: torch.Tensor,
    centered: torch.Tensor,
    rsqrt: torch.Tensor,
    affine_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert producer.shape == (N, C, H, W)
    assert centered.shape == (N, C, H, W)
    assert rsqrt.shape == (C,)
    assert affine_weight.shape == (C,)

    sum1 = torch.zeros((C,), device=producer.device, dtype=torch.float32)
    sum2 = torch.zeros((C,), device=producer.device, dtype=torch.float32)

    block_k = 2048
    _dual_reduce_split_k_kernel[(C, triton.cdiv(TOTAL_SPATIAL, block_k))](
        producer,
        centered,
        sum1,
        sum2,
        total_spatial=TOTAL_SPATIAL,
        C_=C,
        HW_=HW,
        BLOCK_K=block_k,
        num_warps=8,
    )

    out = torch.empty_like(producer)
    block_elems = 1024
    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        producer,
        centered,
        rsqrt,
        affine_weight,
        sum1,
        sum2,
        out,
        C_=C,
        HW_=HW,
        SCALE_=SCALE,
        NUMEL_=NUMEL,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, sum2 * rsqrt


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


@oracle_impl(hardware="H100", shapes="(T([512, 72, 1, 1], f32), T([512, 72, 28, 28], f32), T([512, 72, 1, 1], f32), T([1, 72, 1, 1], f32), T([512, 72, 28, 28], f32), T([1, 72, 1, 1], f32), T([72], f32), S([512, 72, 28, 28]))")
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
