"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full MobileViT SiLU-gradient plus batch-norm-backward return tuple by cooperatively split-K reducing the two per-channel summaries over the shared `(N, H, W)` domain from the original repro inputs and then recomputing the fused producer in one epilogue that writes both the contiguous input-gradient tensor and `sum2 * rsqrt` vector, whereas Inductor currently schedules the decomposed slice, SiLU derivative producer, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogue as ordinary reduction/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible small-output channel reductions with a finalized-summary tensor epilogue while preserving the target output layout; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible channel reductions across the reduced N/H/W dimension, combine the partial summaries, and fuse the downstream tensor and side-vector epilogues with the SiLU-gradient producer."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


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

sys.path.insert(0, str(REPO_ROOT))


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


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


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
        print(f"output_count: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        offset_ok = got.storage_offset() == ref.storage_offset()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs = float("inf")
            max_rel = float("inf")
            value_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and offset_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} storage_offset={got.storage_offset()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} offset_match={offset_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_default(model: torch.nn.Module, inputs: tuple[object, ...]):
    import torch._dynamo

    torch._dynamo.reset()
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
        oracle_us = (
            triton.testing.do_bench(
                lambda: oracle_full(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            )
            * 1000.0
        )

    print(f"oracle_full cooperative split-k SiLU+BN backward: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        compiled = _compile_default(model, inputs)
        compiled_us = (
            triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            )
            * 1000.0
        )
    print(f"torch.compile default: {compiled_us:.3f} us")


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
