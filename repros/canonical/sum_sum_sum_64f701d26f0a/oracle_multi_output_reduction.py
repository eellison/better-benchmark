"""
Oracle kernel for sum_sum_sum_64f701d26f0a (BEiT layer-norm backward).

The repro has one layer-norm backward whose per-row reductions feed three
column-wise gradient reductions and a transposed gradient output:

    row_sum[m] = sum_d mm_92[m, d] * primals_17[d]
    row_dot[m] = sum_d mm_92[m, d] * primals_17[d] * xhat[m, d]

    xhat[m, d] = (cat[m, d] + primals_5[d] * addmm_1[m, d] - mean[m]) * rsqrt[m]

    grad[m, d] = add_129[m, d] + rsqrt[m] / D * (
        mm_92[m, d] * primals_17[d] * D - row_sum[m] - xhat[m, d] * row_dot[m]
    )

Outputs:
    sum(mm_92 * xhat, dim=0)
    sum(mm_92, dim=0)
    sum(grad * addmm_1, dim=0)
    (grad * primals_5).permute(1, 0)
    sum(grad * primals_5, dim=0)

Oracle strategy:
    - Phase 1: one Triton pass over each row computes the two row reductions.
    - Phase 2: one tiled Triton pass recomputes the pointwise terms once,
      writes the returned gradient matrix, and feeds four column accumulators.

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle differs from
Inductor by treating the BEiT layer-norm backward as a scheduled two-phase
multi-output reduction, sharing the per-row dual reduction over
mm_92*primals_17 and mm_92*primals_17*xhat, then sharing the second pass that
computes grad/addmm, grad*primals_5, the returned transpose buffer, and the
three column reductions; Inductor currently sees these as separate aten.sum,
pointwise, reshape, and permute nodes with a reduction dependency between
the row sums and later column sums, so its generic scheduler cannot represent
one fused reduction template with multiple accumulators plus a required
materialized side output; the fix is scheduler/codegen support for dependent
multi-output reduction templates that can carry row-reduction outputs into a
second tiled reduction while fusing same-base column accumulators and side
stores under an autotuned shape guard.
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



REPRO_ID = "sum_sum_sum_64f701d26f0a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

M = 128 * 197
D = 768
INV_D = 1.0 / D


def _row_dual_reduce_kernel(
    mm_ptr,
    addmm_ptr,
    cat_ptr,
    mean_ptr,
    rsqrt_ptr,
    primals17_ptr,
    primals5_ptr,
    row_sum_ptr,
    row_dot_ptr,
    D_: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK_D)
    mask = d < D_
    offsets = row * D_ + d

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cat = tl.load(cat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    p17 = tl.load(primals17_ptr + d, mask=mask, other=0.0).to(tl.float32)
    p5 = tl.load(primals5_ptr + d, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + row).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + row).to(tl.float32)

    grad_weighted = mm * p17
    xhat = (cat + p5 * addmm - mean) * rsqrt

    tl.store(row_sum_ptr + row, tl.sum(grad_weighted, axis=0))
    tl.store(row_dot_ptr + row, tl.sum(grad_weighted * xhat, axis=0))


@triton.jit
def _column_reduce_and_store_kernel(
    mm_ptr,
    addmm_ptr,
    cat_ptr,
    mean_ptr,
    rsqrt_ptr,
    add129_ptr,
    primals17_ptr,
    primals5_ptr,
    row_sum_ptr,
    row_dot_ptr,
    out_sum_mm_xhat_ptr,
    out_sum_mm_ptr,
    out_sum_grad_addmm_ptr,
    out_grad_scaled_md_ptr,
    out_sum_grad_scaled_ptr,
    M_: tl.constexpr,
    D_: tl.constexpr,
    INV_D_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    m_mask = m < M_
    d_mask = d < D_
    mask = m_mask[:, None] & d_mask[None, :]
    offsets = m[:, None] * D_ + d[None, :]

    mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cat = tl.load(cat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add129 = tl.load(add129_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    p17 = tl.load(primals17_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    p5 = tl.load(primals5_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

    p17_b = p17[None, :]
    p5_b = p5[None, :]
    mean_b = mean[:, None]
    rsqrt_b = rsqrt[:, None]
    row_sum_b = row_sum[:, None]
    row_dot_b = row_dot[:, None]

    grad_weighted = mm * p17_b
    xhat = (cat + p5_b * addmm - mean_b) * rsqrt_b
    ln_grad = rsqrt_b * INV_D_ * (grad_weighted * D_ - row_sum_b - xhat * row_dot_b)
    grad = add129 + ln_grad
    grad_scaled = grad * p5_b

    tl.store(out_grad_scaled_md_ptr + offsets, grad_scaled, mask=mask)

    sum_mm_xhat = tl.sum(tl.where(mask, mm * xhat, 0.0), axis=0)
    sum_mm = tl.sum(tl.where(mask, mm, 0.0), axis=0)
    sum_grad_addmm = tl.sum(tl.where(mask, grad * addmm, 0.0), axis=0)
    sum_grad_scaled = tl.sum(tl.where(mask, grad_scaled, 0.0), axis=0)

    tl.atomic_add(out_sum_mm_xhat_ptr + d, sum_mm_xhat, sem="relaxed", mask=d_mask)
    tl.atomic_add(out_sum_mm_ptr + d, sum_mm, sem="relaxed", mask=d_mask)
    tl.atomic_add(out_sum_grad_addmm_ptr + d, sum_grad_addmm, sem="relaxed", mask=d_mask)
    tl.atomic_add(out_sum_grad_scaled_ptr + d, sum_grad_scaled, sem="relaxed", mask=d_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_92,
        primals_17,
        addmm_1,
        primals_5,
        cat,
        getitem_10,
        rsqrt_1,
        add_129,
        *_shape_params,
    ) = inputs

    return (
        mm_92.contiguous(),
        primals_17.contiguous(),
        addmm_1.contiguous(),
        primals_5.contiguous(),
        cat.reshape(M, D).contiguous(),
        getitem_10.reshape(M).contiguous(),
        rsqrt_1.reshape(M).contiguous(),
        add_129.reshape(M, D).contiguous(),
    )


def oracle_fused(
    mm_92: torch.Tensor,
    primals_17: torch.Tensor,
    addmm_1: torch.Tensor,
    primals_5: torch.Tensor,
    cat_md: torch.Tensor,
    mean_m: torch.Tensor,
    rsqrt_m: torch.Tensor,
    add_129_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_92.shape == (M, D)
    assert addmm_1.shape == (M, D)
    assert cat_md.shape == (M, D)
    assert add_129_md.shape == (M, D)
    assert primals_17.shape == (D,)
    assert primals_5.shape == (D,)
    assert mean_m.shape == (M,)
    assert rsqrt_m.shape == (M,)

    device = mm_92.device
    row_sum = torch.empty((M,), device=device, dtype=torch.float32)
    row_dot = torch.empty((M,), device=device, dtype=torch.float32)

    _row_dual_reduce_kernel[(M,)](
        mm_92,
        addmm_1,
        cat_md,
        mean_m,
        rsqrt_m,
        primals_17,
        primals_5,
        row_sum,
        row_dot,
        D_=D,
        BLOCK_D=1024,
    )

    sum_mm_xhat = torch.zeros((D,), device=device, dtype=torch.float32)
    sum_mm = torch.zeros((D,), device=device, dtype=torch.float32)
    sum_grad_addmm = torch.zeros((D,), device=device, dtype=torch.float32)
    grad_scaled_md = torch.empty((M, D), device=device, dtype=torch.float32)
    sum_grad_scaled = torch.zeros((D,), device=device, dtype=torch.float32)

    block_m = 32
    block_d = 32
    grid = (triton.cdiv(M, block_m), triton.cdiv(D, block_d))
    _column_reduce_and_store_kernel[grid](
        mm_92,
        addmm_1,
        cat_md,
        mean_m,
        rsqrt_m,
        add_129_md,
        primals_17,
        primals_5,
        row_sum,
        row_dot,
        sum_mm_xhat,
        sum_mm,
        sum_grad_addmm,
        grad_scaled_md,
        sum_grad_scaled,
        M_=M,
        D_=D,
        INV_D_=INV_D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )

    return (
        sum_mm_xhat,
        sum_mm,
        sum_grad_addmm,
        grad_scaled_md.permute(1, 0),
        sum_grad_scaled,
    )


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
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
