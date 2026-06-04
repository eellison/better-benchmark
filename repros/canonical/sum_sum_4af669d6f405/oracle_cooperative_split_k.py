"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ShuffleNet channel-shuffle plus batch-norm-backward return tuple from Repro.forward by reconstructing the shuffled high-half slice from the original cat/view/permute/clone producer, cooperatively split-K reducing both per-channel summaries over the shared N/H/W domain, and using the finalized summaries in one epilogue that writes the full input-gradient tensor plus scale-gradient side output, whereas Inductor currently schedules the channel shuffle, ReLU-gated producer, sibling sums, and dependent BN-backward epilogues as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible channel reductions with a dependent full-tensor side-output epilogue fed by a layout-changing producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN channel reductions across the reduced N/H/W dimension, combine partial summaries once, and fuse the channel-shuffle producer plus downstream tensor/vector epilogues."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_4af669d6f405"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C_IN = 116
C = 58
H = 28
W = 28
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
HIGH_HALF = 58
SCALE = 2.4912308673469386e-06
REDUCE_BLOCK_K = 2048
EPILOGUE_BLOCK_ELEMS = 1024

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _load_shuffled_high_half(
    view_28_ptr,
    getitem_138_ptr,
    n,
    c,
    hw,
    active,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
):
    # After cat([view_28[:, :58], getitem_138], 1).view(N, 58, 2, H, W)
    # and permute(0, 2, 1, 3, 4), output channel 58 + c reads cat channel
    # 2*c + 1. Channels below 58 come from view_28; the rest from getitem_138.
    cat_c = c * 2 + 1
    from_view = cat_c < C_
    view_offset = n * (C_IN_ * HW_) + cat_c * HW_ + hw
    get_c = tl.where(from_view, 0, cat_c - C_)
    getitem_offset = n * (C_ * HW_) + get_c * HW_ + hw

    view_value = tl.load(
        view_28_ptr + view_offset,
        mask=active & from_view,
        other=0.0,
    ).to(tl.float32)
    getitem_value = tl.load(
        getitem_138_ptr + getitem_offset,
        mask=active & ~from_view,
        other=0.0,
    ).to(tl.float32)
    return tl.where(from_view, view_value, getitem_value)


@triton.jit
def _shuffled_bn_dual_reduce_kernel(
    view_28_ptr,
    getitem_138_ptr,
    arg162_ptr,
    arg163_ptr,
    arg164_ptr,
    arg21_ptr,
    arg22_ptr,
    full_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    bn_offsets = n * (C_ * HW_) + c * HW_ + hw

    source = _load_shuffled_high_half(
        view_28_ptr,
        getitem_138_ptr,
        n,
        c,
        hw,
        active,
        C_IN_,
        C_,
        HW_,
    )
    bn_input = tl.load(arg162_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg163_ptr + c).to(tl.float32)
    invstd = tl.load(arg164_ptr + c).to(tl.float32)
    affine_weight = tl.load(arg21_ptr + c).to(tl.float32)
    affine_bias = tl.load(arg22_ptr + c).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    centered = bn_input - mean
    normalized = _mul_rn_f32(centered, invstd)
    relu_input = _add_rn_f32(_mul_rn_f32(normalized, affine_weight), affine_bias)
    where_self = tl.where(active, tl.where(relu_input <= 0.0, full_value, source), 0.0)

    tl.atomic_add(sum_where_ptr + c, tl.sum(where_self, axis=0), sem="relaxed")
    tl.atomic_add(
        sum_where_centered_ptr + c,
        tl.sum(_mul_rn_f32(where_self, centered), axis=0),
        sem="relaxed",
    )


@triton.jit
def _bn_backward_epilogue_kernel(
    view_28_ptr,
    getitem_138_ptr,
    arg162_ptr,
    arg163_ptr,
    arg164_ptr,
    arg21_ptr,
    arg22_ptr,
    full_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_grad_ptr,
    out_scale_grad_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < NUMEL_

    hw = linear % HW_
    c = (linear // HW_) % C_
    n = linear // (C_ * HW_)
    bn_offsets = linear

    source = _load_shuffled_high_half(
        view_28_ptr,
        getitem_138_ptr,
        n,
        c,
        hw,
        active,
        C_IN_,
        C_,
        HW_,
    )
    bn_input = tl.load(arg162_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg163_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(arg164_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(arg21_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_bias = tl.load(arg22_ptr + c, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(
        sum_where_centered_ptr + c,
        mask=active,
        other=0.0,
    ).to(tl.float32)

    centered = bn_input - mean
    normalized = _mul_rn_f32(centered, invstd)
    relu_input = _add_rn_f32(_mul_rn_f32(normalized, affine_weight), affine_bias)
    where_self = tl.where(relu_input <= 0.0, full_value, source)

    mean_term = _mul_rn_f32(sum_where, SCALE_)
    scaled_centered_sum = _mul_rn_f32(sum_where_centered, SCALE_)
    invstd_sq = _mul_rn_f32(invstd, invstd)
    variance_term = _mul_rn_f32(scaled_centered_sum, invstd_sq)
    affine_term = _mul_rn_f32(invstd, affine_weight)

    centered_variance = _mul_rn_f32(centered, variance_term)
    sub_mean = _add_rn_f32(_add_rn_f32(where_self, -centered_variance), -mean_term)
    out_grad = _mul_rn_f32(sub_mean, affine_term)
    tl.store(out_grad_ptr + linear, out_grad, mask=active)

    tl.store(
        out_scale_grad_ptr + c,
        _mul_rn_f32(sum_where_centered, invstd),
        mask=active & (n == 0) & (hw == 0),
    )


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
    view_28: torch.Tensor,
    getitem_138: torch.Tensor,
    arg162_1: torch.Tensor,
    arg163_1: torch.Tensor,
    arg164_1: torch.Tensor,
    arg21_1: torch.Tensor,
    arg22_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert _shape_param_0 == [N, C, 2, H, W]
    assert _shape_param_1 == [N, C_IN, H, W]
    assert view_28.shape == (N, C_IN, H, W)
    assert getitem_138.shape == (N, C, H, W)
    assert arg162_1.shape == (N, C, H, W)
    assert arg163_1.shape == (1, C, 1, 1)
    assert arg164_1.shape == (1, C, 1, 1)
    assert arg21_1.shape == (C,)
    assert arg22_1.shape == (C,)
    assert full.shape == ()
    assert view_28.is_contiguous()
    assert getitem_138.is_contiguous()
    assert arg162_1.is_contiguous()
    assert arg163_1.is_contiguous()
    assert arg164_1.is_contiguous()
    assert arg21_1.is_contiguous()
    assert arg22_1.is_contiguous()
    assert full.is_contiguous()

    device = arg162_1.device
    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    _shuffled_bn_dual_reduce_kernel[(C, triton.cdiv(TOTAL_SPATIAL, REDUCE_BLOCK_K))](
        view_28,
        getitem_138,
        arg162_1,
        arg163_1,
        arg164_1,
        arg21_1,
        arg22_1,
        full,
        sum_where,
        sum_where_centered,
        C_IN_=C_IN,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=REDUCE_BLOCK_K,
        num_warps=8,
    )

    out_grad = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    out_scale_grad = torch.empty((C,), device=device, dtype=torch.float32)

    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK_ELEMS),)](
        view_28,
        getitem_138,
        arg162_1,
        arg163_1,
        arg164_1,
        arg21_1,
        arg22_1,
        full,
        sum_where,
        sum_where_centered,
        out_grad,
        out_scale_grad,
        C_IN_=C_IN,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        SCALE_=SCALE,
        BLOCK_ELEMS=EPILOGUE_BLOCK_ELEMS,
        num_warps=4,
    )

    return out_grad, out_scale_grad


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
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs = float("inf")
            max_rel = float("inf")
            value_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _bench_us(fn, warmup: int, rep: int) -> float:
    return (
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="min",
        )
        * 1000.0
    )


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


def run_bench(rep: int, warmup: int, compile_inductor: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    module = _load_repro_module()
    eager_model = module.Repro().cuda()

    with torch.no_grad():
        oracle_full(*inputs)
        eager_model(*inputs)
        torch.cuda.synchronize()

        oracle_us = _bench_us(lambda: oracle_full(*inputs), warmup=warmup, rep=rep)
        eager_us = _bench_us(lambda: eager_model(*inputs), warmup=warmup, rep=rep)

    print(f"oracle_full full-scope cooperative split-k: {oracle_us:.3f} us")
    print(f"eager Repro.forward: {eager_us:.3f} us")

    if not compile_inductor:
        return

    compile_configs = [
        ("default", {}),
        ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    ]
    for label, config in compile_configs:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = _bench_us(lambda: compiled(*inputs), warmup=warmup, rep=rep)
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument(
        "--compile-inductor",
        action="store_true",
        help="also benchmark torch.compile variants",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, compile_inductor=args.compile_inductor)


if __name__ == "__main__":
    with torch.no_grad():
        main()
