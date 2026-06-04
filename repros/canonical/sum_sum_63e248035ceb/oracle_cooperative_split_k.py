"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full GhostNet batch-norm-backward return tuple by cooperatively split-K reducing the value-equivalent high-channel add/copy/clone slice into both per-channel summaries, then using those finalized summaries in a fused epilogue that writes the channels-last `[512, 80, 7, 7]` input-gradient output and the `[80]` scale-gradient vector, whereas Inductor currently schedules the memory-format copy/slice producer, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogues as ordinary reductions and pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible channel reductions across the `(N, H, W)` domain while preserving a dependent full-tensor epilogue with channels-last output layout; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible small-output channel reductions across the reduced `(N, H, W)` domain, combine the partial summaries, and fuse the downstream tensor and vector epilogues with memory-format-aware stores."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_63e248035ceb"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C_IN = 160
C = 80
SLICE_START = 80
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
SCALE = 3.985969387755102e-05
OUT_STRIDE = (C * HW, 1, W * C, C)
REDUCE_BLOCK_C = 2
REDUCE_BLOCK_K = 1024
EPILOGUE_BLOCK_ELEMS = 1024



@triton.jit
def _dual_reduce_split_k_kernel(
    clone_ptr,
    getitem_ptr,
    rhs_ptr,
    mean_ptr,
    sum_x_ptr,
    sum_x_rhs_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
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
    x_offsets = n[None, :] * (C_IN_ * HW_) + (SLICE_START_ + c[:, None]) * HW_ + hw[None, :]
    rhs_offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]

    x = (
        tl.load(clone_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    )
    rhs = (
        tl.load(rhs_ptr + rhs_offsets, mask=mask, other=0.0).to(tl.float32)
        - tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    )

    tl.atomic_add(sum_x_ptr + c, tl.sum(x, axis=1), sem="relaxed", mask=c_mask)
    tl.atomic_add(sum_x_rhs_ptr + c, tl.sum(x * rhs, axis=1), sem="relaxed", mask=c_mask)


@triton.jit
def _bn_backward_epilogue_kernel(
    clone_ptr,
    getitem_ptr,
    rhs_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    sum_x_ptr,
    sum_x_rhs_ptr,
    out_ptr,
    out_scale_grad_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < NUMEL_

    c = linear % C_
    hw = (linear // C_) % HW_
    n = linear // (C_ * HW_)

    x_offsets = n * (C_IN_ * HW_) + (SLICE_START_ + c) * HW_ + hw
    rhs_offsets = n * (C_ * HW_) + c * HW_ + hw

    x = (
        tl.load(clone_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    )
    rhs = (
        tl.load(rhs_ptr + rhs_offsets, mask=active, other=0.0).to(tl.float32)
        - tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    )
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_x_rhs = tl.load(sum_x_rhs_ptr + c, mask=active, other=0.0).to(tl.float32)

    mean_term = sum_x * SCALE_
    variance_term = sum_x_rhs * SCALE_ * rsqrt * rsqrt
    out = (x - rhs * variance_term - mean_term) * (rsqrt * affine_weight)
    tl.store(out_ptr + linear, out, mask=active)

    tl.store(out_scale_grad_ptr + c, sum_x_rhs * rsqrt, mask=active & (n == 0) & (hw == 0))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_full(
    clone_2: torch.Tensor,
    getitem_51: torch.Tensor,
    arg422_1: torch.Tensor,
    arg488_1: torch.Tensor,
    arg423_1: torch.Tensor,
    arg159_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert clone_2.shape == (N, C_IN, H, W)
    assert getitem_51.shape == (N, C_IN, H, W)
    assert arg422_1.shape == (N, C, H, W)
    assert arg488_1.shape == (1, C, 1, 1)
    assert arg423_1.shape == (C,)
    assert arg159_1.shape == (C,)
    assert clone_2.is_contiguous()
    assert getitem_51.is_contiguous()
    assert arg422_1.is_contiguous()
    assert arg488_1.is_contiguous()
    assert arg423_1.is_contiguous()
    assert arg159_1.is_contiguous()

    sum_x = torch.zeros((C,), device=clone_2.device, dtype=torch.float32)
    sum_x_rhs = torch.zeros((C,), device=clone_2.device, dtype=torch.float32)

    block_c = REDUCE_BLOCK_C
    block_k = REDUCE_BLOCK_K
    _dual_reduce_split_k_kernel[(triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))](
        clone_2,
        getitem_51,
        arg422_1,
        arg488_1,
        sum_x,
        sum_x_rhs,
        C_IN_=C_IN,
        C_=C,
        SLICE_START_=SLICE_START,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=4,
    )

    out = torch.empty_strided((N, C, H, W), OUT_STRIDE, device=clone_2.device, dtype=torch.float32)
    out_scale_grad = torch.empty((C,), device=clone_2.device, dtype=torch.float32)

    block_elems = EPILOGUE_BLOCK_ELEMS
    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        clone_2,
        getitem_51,
        arg422_1,
        arg488_1,
        arg423_1,
        arg159_1,
        sum_x,
        sum_x_rhs,
        out,
        out_scale_grad,
        C_IN_=C_IN,
        C_=C,
        SLICE_START_=SLICE_START,
        HW_=HW,
        NUMEL_=NUMEL,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, out_scale_grad


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


def _compile_model(model: torch.nn.Module, inputs: tuple[object, ...]):
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

    print(f"oracle_full cooperative split-k: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        compiled = _compile_model(model, inputs)
        compiled_us = triton.testing.do_bench(
            lambda: compiled(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"torch.compile default: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--warmup", type=int, default=25, help="benchmark warmup iterations")
    parser.add_argument("--rep", type=int, default=100, help="benchmark repetitions")
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the Triton oracle")
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
