"""
Canonical oracle for sum_sum_sum_f68c9f1fa09b (ConvNeXtV2 GRN/GELU backward).

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle differs from Inductor by reading the large `[128, 2560, 7, 7]` activation and gradient tensors once, computing all shared GELU/GRN terms into five per-`(N,C)` scalar summaries, and deriving the final dependent channel reduction from those summaries instead of materializing `add_tensor_3`, `where_self`, and `mul_tensor_15` or launching separate reductions over the same input; Inductor cannot do this today because the scheduler sees the final `aten.sum` as depending on intermediate reductions (`sum_dim_int_list_2` and the cross-channel correction) and preserves the original pointwise/reduction graph rather than proving that the final spatial pass can be algebraically rewritten into sibling per-`(N,C)` sums; the fix is an ALGEBRAIC_ELIMINATION pass for dependent reduction templates that recognizes this GRN backward form, emits a multi-accumulator spatial summary kernel, reduces the small cross-channel correction, and lowers the remaining channel outputs from the summaries.
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



REPRO_ID = "sum_sum_sum_f68c9f1fa09b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 2560
H = 7
W = 7
HW = H * W
INV_C = 1.0 / C


def _spatial_summary_kernel(
    x_ptr,
    norm_ptr,
    norm_mean_ptr,
    grad_ptr,
    weight_ptr,
    spatial_dot_nc_ptr,
    gelu_gelu_grad_nc_ptr,
    out_grad_gelu_norm_ptr,
    out_grad_ptr,
    base_gelu_grad_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)

    c_mask = c < C_
    hw_mask = hw < HW_
    mask = c_mask[:, None] & hw_mask[None, :]
    offsets = n * C_ * HW_ + c[:, None] * HW_ + hw[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + n * C_ + c, mask=c_mask, other=0.0).to(tl.float32)
    norm_mean = tl.load(norm_mean_ptr + n).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    erf_plus_one = tl.math.erf(x * 0.7071067811865476) + 1.0
    gelu = x * 0.5 * erf_plus_one
    normal_pdf = tl.exp(x * x * -0.5) * 0.3989422804014327
    gelu_grad = 0.5 * erf_plus_one + x * normal_pdf

    norm_ratio = norm / norm_mean
    weighted_norm = weight * norm_ratio

    grad_masked = tl.where(mask, grad, 0.0)
    gelu_masked = tl.where(mask, gelu, 0.0)
    gelu_grad_masked = tl.where(mask, gelu_grad, 0.0)

    sum_grad_gelu_norm = tl.sum(grad_masked * gelu_masked * norm_ratio[:, None], axis=1)
    spatial_dot = tl.sum(grad_masked * gelu_masked * weight[:, None], axis=1)
    base_gelu_grad = tl.sum(
        grad_masked * (1.0 + weighted_norm[:, None]) * gelu_grad_masked,
        axis=1,
    )
    sum_grad = tl.sum(grad_masked, axis=1)
    gelu_gelu_grad = tl.sum(gelu_masked * gelu_grad_masked, axis=1)

    out_offsets = n * C_ + c
    tl.store(spatial_dot_nc_ptr + out_offsets, spatial_dot, mask=c_mask)
    tl.store(gelu_gelu_grad_nc_ptr + out_offsets, gelu_gelu_grad, mask=c_mask)
    tl.atomic_add(out_grad_gelu_norm_ptr + c, sum_grad_gelu_norm, sem="relaxed", mask=c_mask)
    tl.atomic_add(out_grad_ptr + c, sum_grad, sem="relaxed", mask=c_mask)
    tl.atomic_add(base_gelu_grad_ptr + c, base_gelu_grad, sem="relaxed", mask=c_mask)


@triton.jit
def _row_correction_kernel(
    spatial_dot_nc_ptr,
    norm_ptr,
    norm_mean_ptr,
    row_correction_ptr,
    C_: tl.constexpr,
    INV_C_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    mask = c < C_

    spatial_dot = tl.load(spatial_dot_nc_ptr + n * C_ + c, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + n * C_ + c, mask=mask, other=0.0).to(tl.float32)
    norm_mean = tl.load(norm_mean_ptr + n).to(tl.float32)

    correction_terms = -spatial_dot * norm / (norm_mean * norm_mean)
    row_correction = tl.sum(tl.where(mask, correction_terms, 0.0), axis=0) * INV_C_
    tl.store(row_correction_ptr + n, row_correction)


@triton.jit
def _finalize_outputs_kernel(
    spatial_dot_nc_ptr,
    gelu_gelu_grad_nc_ptr,
    norm_ptr,
    norm_mean_ptr,
    row_correction_ptr,
    base_gelu_grad_ptr,
    out_input_grad_ptr,
    N_: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)
    c_mask = c < C_
    n_mask = n < N_
    mask = n_mask[:, None] & c_mask[None, :]
    offsets = n[:, None] * C_ + c[None, :]

    spatial_dot = tl.load(spatial_dot_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gelu_gelu_grad = tl.load(gelu_gelu_grad_nc_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm_mean = tl.load(norm_mean_ptr + n, mask=n_mask, other=0.0).to(tl.float32)
    row_correction = tl.load(row_correction_ptr + n, mask=n_mask, other=0.0).to(tl.float32)
    base_gelu_grad = tl.load(base_gelu_grad_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    inv_norm = tl.where(norm == 0.0, 0.0, 1.0 / norm)
    grn_backprop = spatial_dot / norm_mean[:, None] + row_correction[:, None]
    correction = tl.sum(tl.where(mask, grn_backprop * gelu_gelu_grad * inv_norm, 0.0), axis=0)
    out_input_grad = base_gelu_grad + correction

    tl.store(out_input_grad_ptr + c, out_input_grad, mask=c_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    arg212_1, arg213_1, arg214_1, getitem, arg77_1, *_shape_params = inputs
    return (
        arg212_1.contiguous(),
        arg213_1.contiguous(),
        arg214_1.contiguous(),
        getitem.contiguous(),
        arg77_1.contiguous(),
    )


def oracle_fused(
    arg212_1: torch.Tensor,
    arg213_1: torch.Tensor,
    arg214_1: torch.Tensor,
    getitem: torch.Tensor,
    arg77_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert arg212_1.shape == (N, C, H, W)
    assert arg213_1.shape == (N, C, 1, 1)
    assert arg214_1.shape == (N, 1, 1, 1)
    assert getitem.shape == (N, C, H, W)
    assert arg77_1.shape == (C,)

    device = arg212_1.device
    spatial_dot_nc = torch.empty((N, C), device=device, dtype=torch.float32)
    gelu_gelu_grad_nc = torch.empty((N, C), device=device, dtype=torch.float32)
    out_grad_gelu_norm = torch.zeros((C,), device=device, dtype=torch.float32)
    out_grad = torch.zeros((C,), device=device, dtype=torch.float32)
    base_gelu_grad = torch.zeros((C,), device=device, dtype=torch.float32)

    block_c = 32
    _spatial_summary_kernel[(N, triton.cdiv(C, block_c))](
        arg212_1,
        arg213_1,
        arg214_1,
        getitem,
        arg77_1,
        spatial_dot_nc,
        gelu_gelu_grad_nc,
        out_grad_gelu_norm,
        out_grad,
        base_gelu_grad,
        C_=C,
        HW_=HW,
        BLOCK_HW=64,
        BLOCK_C=block_c,
        num_warps=8,
    )

    row_correction = torch.empty((N,), device=device, dtype=torch.float32)
    _row_correction_kernel[(N,)](
        spatial_dot_nc,
        arg213_1,
        arg214_1,
        row_correction,
        C_=C,
        INV_C_=INV_C,
        BLOCK_C=4096,
        num_warps=8,
    )

    out_input_grad = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_outputs_kernel[(triton.cdiv(C, block_c),)](
        spatial_dot_nc,
        gelu_gelu_grad_nc,
        arg213_1,
        arg214_1,
        row_correction,
        base_gelu_grad,
        out_input_grad,
        N_=N,
        C_=C,
        BLOCK_N=128,
        BLOCK_C=block_c,
        num_warps=8,
    )

    return out_grad_gelu_norm, out_grad, out_input_grad


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
