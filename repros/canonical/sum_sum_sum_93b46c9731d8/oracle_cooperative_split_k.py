"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full RegNet/ResNeSt dual-branch ReLU plus batch-norm-backward return tuple by cooperatively split-K reducing the shared masked pool-gradient producer into one per-channel mask sum and two centered channel sums, then using those finalized summaries in a fused epilogue that writes both returned `[32, 2240, 7, 7]` gradients and `[2240]` scale-gradient vectors, whereas Inductor currently schedules the decomposed pointwise mask, sibling `sum([0, 2, 3])` reductions, and two BN-backward epilogues as ordinary reductions/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible sibling channel reductions and dependent full-tensor epilogues across two BN branches; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible small-output channel reductions across the `(N, H, W)` domain, combine the partials, and fuse the downstream branch epilogues plus side-vector outputs."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_93b46c9731d8"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 32
C = 2240
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_HW = 1.0 / HW
SCALE = 1.0 / TOTAL_SPATIAL



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


@triton.jit
def _dual_bn_reduce_split_k_kernel(
    mm_ptr,
    x1_ptr,
    mean1_ptr,
    rsqrt1_ptr,
    weight1_ptr,
    bias1_ptr,
    x2_ptr,
    mean2_ptr,
    rsqrt2_ptr,
    weight2_ptr,
    bias2_ptr,
    sum_mask_ptr,
    sum_centered1_ptr,
    sum_centered2_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)

    c_mask = c < C_
    k_mask = k < TOTAL_SPATIAL_
    mask = c_mask[:, None] & k_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    x_offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mm_offsets = n[None, :] * C_ + c[:, None]

    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    mean1 = tl.load(mean1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    rsqrt1 = tl.load(rsqrt1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    mean2 = tl.load(mean2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    rsqrt2 = tl.load(rsqrt2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered1 = x1 - mean1[:, None]
    centered2 = x2 - mean2[:, None]
    affine1 = centered1 * rsqrt1[:, None] * weight1[:, None] + bias1[:, None]
    affine2 = centered2 * rsqrt2[:, None] * weight2[:, None] + bias2[:, None]
    masked_grad = tl.where((affine1 + affine2) <= 0.0, 0.0, mm * INV_HW_)

    tl.store(sum_mask_ptr + c, tl.sum(masked_grad, axis=1), mask=c_mask)
    tl.store(sum_centered1_ptr + c, tl.sum(masked_grad * centered1, axis=1), mask=c_mask)
    tl.store(sum_centered2_ptr + c, tl.sum(masked_grad * centered2, axis=1), mask=c_mask)


@triton.jit
def _dual_bn_epilogue_kernel(
    mm_ptr,
    x1_ptr,
    mean1_ptr,
    rsqrt1_ptr,
    weight1_ptr,
    bias1_ptr,
    x2_ptr,
    mean2_ptr,
    rsqrt2_ptr,
    weight2_ptr,
    bias2_ptr,
    sum_mask_ptr,
    sum_centered1_ptr,
    sum_centered2_ptr,
    grad2_ptr,
    weight_grad2_ptr,
    grad1_ptr,
    weight_grad1_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < NUMEL_

    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    mm_offsets = n * C_ + c

    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean1 = tl.load(mean1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt1 = tl.load(rsqrt1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=mask, other=0.0).to(tl.float32)

    mean2 = tl.load(mean2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt2 = tl.load(rsqrt2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c, mask=mask, other=0.0).to(tl.float32)

    sum_mask = tl.load(sum_mask_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_centered1 = tl.load(sum_centered1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_centered2 = tl.load(sum_centered2_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered1 = x1 - mean1
    centered2 = x2 - mean2
    affine1 = centered1 * rsqrt1 * weight1 + bias1
    affine2 = centered2 * rsqrt2 * weight2 + bias2
    masked_grad = tl.where((affine1 + affine2) <= 0.0, 0.0, mm * INV_HW_)

    mean_term = sum_mask * SCALE_
    variance_term1 = sum_centered1 * SCALE_ * rsqrt1 * rsqrt1
    variance_term2 = sum_centered2 * SCALE_ * rsqrt2 * rsqrt2

    grad1 = (masked_grad - centered1 * variance_term1 - mean_term) * (rsqrt1 * weight1)
    grad2 = (masked_grad - centered2 * variance_term2 - mean_term) * (rsqrt2 * weight2)

    tl.store(grad2_ptr + offsets, grad2, mask=mask)
    tl.store(grad1_ptr + offsets, grad1, mask=mask)

    first_element_for_channel = mask & (n == 0) & ((offsets % HW_) == 0)
    tl.store(weight_grad2_ptr + c, sum_centered2 * rsqrt2, mask=first_element_for_channel)
    tl.store(weight_grad1_ptr + c, sum_centered1 * rsqrt1, mask=first_element_for_channel)


def oracle_full(
    mm: torch.Tensor,
    arg438_1: torch.Tensor,
    arg439_1: torch.Tensor,
    arg440_1: torch.Tensor,
    arg179_1: torch.Tensor,
    arg180_1: torch.Tensor,
    arg441_1: torch.Tensor,
    arg442_1: torch.Tensor,
    arg443_1: torch.Tensor,
    arg182_1: torch.Tensor,
    arg183_1: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1

    assert mm.shape == (N, C)
    assert arg438_1.shape == (N, C, H, W)
    assert arg439_1.shape == (1, C, 1, 1)
    assert arg440_1.shape == (1, C, 1, 1)
    assert arg179_1.shape == (C,)
    assert arg180_1.shape == (C,)
    assert arg441_1.shape == (N, C, H, W)
    assert arg442_1.shape == (1, C, 1, 1)
    assert arg443_1.shape == (1, C, 1, 1)
    assert arg182_1.shape == (C,)
    assert arg183_1.shape == (C,)
    assert mm.is_contiguous()
    assert arg438_1.is_contiguous()
    assert arg439_1.is_contiguous()
    assert arg440_1.is_contiguous()
    assert arg179_1.is_contiguous()
    assert arg180_1.is_contiguous()
    assert arg441_1.is_contiguous()
    assert arg442_1.is_contiguous()
    assert arg443_1.is_contiguous()
    assert arg182_1.is_contiguous()
    assert arg183_1.is_contiguous()

    device = arg438_1.device
    sum_mask = torch.empty((C,), device=device, dtype=torch.float32)
    sum_centered1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_centered2 = torch.empty((C,), device=device, dtype=torch.float32)

    block_c = 2
    block_k = 2048
    reduce_grid = (triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))
    _dual_bn_reduce_split_k_kernel[reduce_grid](
        mm,
        arg438_1,
        arg439_1,
        arg440_1,
        arg179_1,
        arg180_1,
        arg441_1,
        arg442_1,
        arg443_1,
        arg182_1,
        arg183_1,
        sum_mask,
        sum_centered1,
        sum_centered2,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=4,
    )

    grad2 = torch.empty_like(arg441_1)
    weight_grad2 = torch.empty((C,), device=device, dtype=torch.float32)
    grad1 = torch.empty_like(arg438_1)
    weight_grad1 = torch.empty((C,), device=device, dtype=torch.float32)

    block_elems = 512
    _dual_bn_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        mm,
        arg438_1,
        arg439_1,
        arg440_1,
        arg179_1,
        arg180_1,
        arg441_1,
        arg442_1,
        arg443_1,
        arg182_1,
        arg183_1,
        sum_mask,
        sum_centered1,
        sum_centered2,
        grad2,
        weight_grad2,
        grad1,
        weight_grad1,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return grad2, weight_grad2, grad1, weight_grad1


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = _as_tuple(oracle_full(*inputs))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_full(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_full(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    print(f"oracle_full cooperative split-k dual BN backward: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("default", {}),
        ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    ]

    for label, config in compile_configs:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            ) * 1000.0
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark oracle")
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
