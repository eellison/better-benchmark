"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full MobileNetV2 hardtanh-gated batch-norm-backward return tuple by cooperatively split-K reducing the two per-channel summaries over the shared `(N, H, W)` domain and then using the finalized summaries in a fused epilogue that writes both the `[128, 1280, 7, 7]` input-gradient tensor and `[1280]` scale-gradient vector, whereas Inductor currently schedules the gated producer, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogue as ordinary reduction and pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible channel reductions across the reduced `(N, H, W)` domain while preserving the dependent full-tensor epilogue; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible small-output channel reductions across the reduced `(N, H, W)` domain, combine the partial summaries, and fuse the downstream tensor and vector epilogues with the gated producer."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_6e3dcc264797"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
C = 1280
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_HW = 1.0 / HW
SCALE = 0.00015943877551020407
REDUCE_BLOCK_C = 2
REDUCE_BLOCK_K = 512
EPILOGUE_BLOCK_ELEMS = 512

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _gated_bn_dual_reduce_split_k_kernel(
    gate_input_ptr,
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
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
    nc_offsets = n[None, :] * C_ + c[:, None]

    gate_input = tl.load(gate_input_ptr + nc_offsets, mask=mask, other=0).to(tl.float32)
    mm = tl.load(mm_ptr + nc_offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = x - mean[:, None]
    affine = centered * rsqrt[:, None] * affine_weight[:, None] + affine_bias[:, None]
    in_open_interval = (affine > 0.0) & (affine < 6.0)
    where_self = tl.where(mask & in_open_interval, mm * gate_input * 1.25 * INV_HW_, 0.0)

    tl.atomic_add(sum_where_ptr + c, tl.sum(where_self, axis=1), sem="relaxed", mask=c_mask)
    tl.atomic_add(
        sum_where_centered_ptr + c,
        tl.sum(where_self * centered, axis=1),
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _bn_backward_epilogue_kernel(
    gate_input_ptr,
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_ptr,
    out_scale_grad_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < NUMEL_

    hw = linear % HW_
    c = (linear // HW_) % C_
    n = linear // (C_ * HW_)
    nc_offsets = n * C_ + c

    gate_input = tl.load(gate_input_ptr + nc_offsets, mask=active, other=0).to(tl.float32)
    mm = tl.load(mm_ptr + nc_offsets, mask=active, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + linear, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(sum_where_centered_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = x - mean
    affine = centered * rsqrt * affine_weight + affine_bias
    in_open_interval = (affine > 0.0) & (affine < 6.0)
    where_self = tl.where(in_open_interval, mm * gate_input * 1.25 * INV_HW_, 0.0)

    mean_term = sum_where * SCALE_
    variance_term = sum_where_centered * SCALE_ * rsqrt * rsqrt
    out = (where_self - centered * variance_term - mean_term) * (rsqrt * affine_weight)
    tl.store(out_ptr + linear, out, mask=active)

    tl.store(out_scale_grad_ptr + c, sum_where_centered * rsqrt, mask=active & (n == 0) & (hw == 0))


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
    arg331_1: torch.Tensor,
    mm: torch.Tensor,
    arg328_1: torch.Tensor,
    arg329_1: torch.Tensor,
    arg330_1: torch.Tensor,
    arg138_1: torch.Tensor,
    arg139_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1

    assert arg331_1.shape == (N, C)
    assert mm.shape == (N, C)
    assert arg328_1.shape == (N, C, H, W)
    assert arg329_1.shape == (1, C, 1, 1)
    assert arg330_1.shape == (1, C, 1, 1)
    assert arg138_1.shape == (C,)
    assert arg139_1.shape == (C,)
    assert arg331_1.is_contiguous()
    assert mm.is_contiguous()
    assert arg328_1.is_contiguous()
    assert arg329_1.is_contiguous()
    assert arg330_1.is_contiguous()
    assert arg138_1.is_contiguous()
    assert arg139_1.is_contiguous()

    sum_where = torch.zeros((C,), device=arg328_1.device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=arg328_1.device, dtype=torch.float32)

    _gated_bn_dual_reduce_split_k_kernel[
        (triton.cdiv(C, REDUCE_BLOCK_C), triton.cdiv(TOTAL_SPATIAL, REDUCE_BLOCK_K))
    ](
        arg331_1,
        mm,
        arg328_1,
        arg329_1,
        arg330_1,
        arg138_1,
        arg139_1,
        sum_where,
        sum_where_centered,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        BLOCK_C=REDUCE_BLOCK_C,
        BLOCK_K=REDUCE_BLOCK_K,
        num_warps=4,
    )

    out = torch.empty_like(arg328_1)
    out_scale_grad = torch.empty((C,), device=arg328_1.device, dtype=torch.float32)

    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK_ELEMS),)](
        arg331_1,
        mm,
        arg328_1,
        arg329_1,
        arg330_1,
        arg138_1,
        arg139_1,
        sum_where,
        sum_where_centered,
        out,
        out_scale_grad,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS=EPILOGUE_BLOCK_ELEMS,
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
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = shape_ok and dtype_ok and value_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} stride={got.stride()} "
            f"expected_stride={ref.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok}"
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

    print(f"oracle_full cooperative split-k hardtanh+BN backward: {oracle_us:.3f} us")

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
