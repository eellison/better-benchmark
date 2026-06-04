"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full EfficientNet SiLU-gradient plus batch-norm-backward return tuple by split-K reducing the two per-channel BN summaries from the original repro inputs and then recomputing the producer in a fused epilogue that writes both returned outputs, whereas Inductor currently lowers the decomposed pointwise producer, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogue as ordinary reduction/pointwise work with too little reduction-axis parallelism; Inductor cannot do this today because its scheduler/codegen does not have a cooperative split-K multi-output reduction template that can coordinate compatible sibling channel reductions and preserve the dependent materialized tensor epilogue; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible small-output channel reductions across the reduced N/H/W dimension, combine the partials, and fuse the downstream BN-backward epilogue and side vector output."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_04bdad42e9ef"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 32
C = 1280
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_HW = 1.0 / HW
SCALE = 1.0 / TOTAL_SPATIAL



@triton.jit
def _silu_bn_dual_reduce_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    sum1_ptr,
    sum2_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    x_offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c).to(tl.float32)

    centered = x - mean
    z = centered * rsqrt * affine_weight + affine_bias
    sig = 1.0 / (tl.exp(-z) + 1.0)
    producer = (mm * INV_HW_) * sig * (z * (1.0 - sig) + 1.0)

    tl.atomic_add(sum1_ptr + c, tl.sum(producer, axis=0), sem="relaxed")
    tl.atomic_add(sum2_ptr + c, tl.sum(producer * centered, axis=0), sem="relaxed")


@triton.jit
def _silu_bn_epilogue_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    out_weight_grad_ptr,
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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + n * C_ + c, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = x - mean
    z = centered * rsqrt * affine_weight + affine_bias
    sig = 1.0 / (tl.exp(-z) + 1.0)
    producer = (mm * INV_HW_) * sig * (z * (1.0 - sig) + 1.0)

    mean_term = sum1 * SCALE_
    variance_term = sum2 * SCALE_ * rsqrt * rsqrt
    out = (producer - centered * variance_term - mean_term) * (rsqrt * affine_weight)
    tl.store(out_ptr + offsets, out, mask=mask)

    is_first_hw_for_channel = mask & (n == 0) & ((offsets % HW_) == 0)
    tl.store(out_weight_grad_ptr + c, sum2 * rsqrt, mask=is_first_hw_for_channel)


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


def oracle_full(
    mm: torch.Tensor,
    arg405_1: torch.Tensor,
    arg406_1: torch.Tensor,
    arg407_1: torch.Tensor,
    arg162_1: torch.Tensor,
    arg163_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1

    assert mm.shape == (N, C)
    assert arg405_1.shape == (N, C, H, W)
    assert arg406_1.shape == (1, C, 1, 1)
    assert arg407_1.shape == (1, C, 1, 1)
    assert arg162_1.shape == (C,)
    assert arg163_1.shape == (C,)
    assert mm.is_contiguous()
    assert arg405_1.is_contiguous()
    assert arg406_1.is_contiguous()
    assert arg407_1.is_contiguous()
    assert arg162_1.is_contiguous()
    assert arg163_1.is_contiguous()

    sum1 = torch.zeros((C,), device=arg405_1.device, dtype=torch.float32)
    sum2 = torch.zeros((C,), device=arg405_1.device, dtype=torch.float32)

    block_k = 2048
    _silu_bn_dual_reduce_kernel[(C, triton.cdiv(TOTAL_SPATIAL, block_k))](
        mm,
        arg405_1,
        arg406_1,
        arg407_1,
        arg162_1,
        arg163_1,
        sum1,
        sum2,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        BLOCK_K=block_k,
        num_warps=8,
    )

    out = torch.empty_like(arg405_1)
    out_weight_grad = torch.empty((C,), device=arg405_1.device, dtype=torch.float32)

    block_elems = 256
    _silu_bn_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        mm,
        arg405_1,
        arg406_1,
        arg407_1,
        arg162_1,
        arg163_1,
        sum1,
        sum2,
        out,
        out_weight_grad,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, out_weight_grad


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, tuple):
        return out
    return (out,)


def _as_tuple(out: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


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
        output_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == ref.stride()
        ok = ok and output_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} stride_match={stride_ok}"
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

    print(f"oracle_full cooperative split-k Silu+BN backward: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("default", {}),
        ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
        (
            "combo_looped_cd",
            {
                "combo_kernels": True,
                "combo_kernel_per_subkernel_blocks": True,
                "coordinate_descent_tuning": True,
                "benchmark_combo_kernel": True,
                "triton.multi_kernel": 3,
            },
        ),
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
