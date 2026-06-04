"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle recomputes the full GhostNet hard-sigmoid slice producer, ReLU mask, paired `[0, 2, 3]` batch-norm-backward channel reductions, final input-gradient tensor, and scale-gradient side output from the original `Repro.forward` inputs, whereas Inductor currently schedules the hard-sigmoid/expand/slice producer, mask, sibling sums, and dependent BN-backward epilogues as separate pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible small-output channel reductions with a dependent full-tensor epilogue fed by the same reconstructed producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN channel reductions across the reduced `N,H,W` dimension, combine the partial summaries, and fuse the downstream tensor and vector epilogues with the original producer."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_384d663c041d"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C_IN = 120
C = 60
SLICE_START = 60
H = 28
W = 28
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_HW = 1.0 / HW
SCALE = 2.4912308673469386e-06
REDUCE_BLOCK_K = 2048
EPILOGUE_BLOCK_ELEMS = 1024



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
def _full_producer_dual_reduce_kernel(
    arg280_ptr,
    getitem_195_ptr,
    getitem_201_ptr,
    arg274_ptr,
    arg275_ptr,
    arg276_ptr,
    arg57_ptr,
    arg58_ptr,
    full_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    src_c = SLICE_START_ + c
    source_offsets = n * (C_IN_ * HW_) + src_c * HW_ + hw
    squeeze_offsets = n * C_IN_ + src_c
    bn_offsets = n * (C_ * HW_) + c * HW_ + hw

    hsigmoid_in = tl.load(arg280_ptr + squeeze_offsets, mask=active, other=0.0).to(tl.float32) + 3.0
    hsigmoid_in = tl.where(hsigmoid_in < 0.0, 0.0, hsigmoid_in)
    hsigmoid_in = tl.where(hsigmoid_in > 6.0, 6.0, hsigmoid_in)
    hsigmoid = hsigmoid_in * 0.16666666666666666

    source = (
        tl.load(getitem_195_ptr + source_offsets, mask=active, other=0.0).to(tl.float32) * hsigmoid
        + tl.load(getitem_201_ptr + squeeze_offsets, mask=active, other=0.0).to(tl.float32) * INV_HW_
    )

    arg274 = tl.load(arg274_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg275_ptr + c).to(tl.float32)
    invstd = tl.load(arg276_ptr + c).to(tl.float32)
    affine_weight = tl.load(arg57_ptr + c).to(tl.float32)
    affine_bias = tl.load(arg58_ptr + c).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    centered = arg274 - mean
    relu_input = _add_rn_f32(_mul_rn_f32(_mul_rn_f32(centered, invstd), affine_weight), affine_bias)
    where_self = tl.where(active, tl.where(relu_input <= 0.0, full_value, source), 0.0)

    tl.atomic_add(sum_where_ptr + c, tl.sum(where_self, axis=0), sem="relaxed")
    tl.atomic_add(sum_where_centered_ptr + c, tl.sum(where_self * centered, axis=0), sem="relaxed")


@triton.jit
def _bn_backward_epilogue_kernel(
    arg280_ptr,
    getitem_195_ptr,
    getitem_201_ptr,
    arg274_ptr,
    arg275_ptr,
    arg276_ptr,
    arg57_ptr,
    arg58_ptr,
    full_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_ptr,
    out_scale_grad_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    SLICE_START_: tl.constexpr,
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
    src_c = SLICE_START_ + c
    source_offsets = n * (C_IN_ * HW_) + src_c * HW_ + hw
    squeeze_offsets = n * C_IN_ + src_c
    bn_offsets = linear

    hsigmoid_in = tl.load(arg280_ptr + squeeze_offsets, mask=active, other=0.0).to(tl.float32) + 3.0
    hsigmoid_in = tl.where(hsigmoid_in < 0.0, 0.0, hsigmoid_in)
    hsigmoid_in = tl.where(hsigmoid_in > 6.0, 6.0, hsigmoid_in)
    hsigmoid = hsigmoid_in * 0.16666666666666666

    source = (
        tl.load(getitem_195_ptr + source_offsets, mask=active, other=0.0).to(tl.float32) * hsigmoid
        + tl.load(getitem_201_ptr + squeeze_offsets, mask=active, other=0.0).to(tl.float32) * INV_HW_
    )

    arg274 = tl.load(arg274_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg275_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(arg276_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(arg57_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_bias = tl.load(arg58_ptr + c, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(sum_where_centered_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = arg274 - mean
    relu_input = _add_rn_f32(_mul_rn_f32(_mul_rn_f32(centered, invstd), affine_weight), affine_bias)
    where_self = tl.where(relu_input <= 0.0, full_value, source)

    mean_term = sum_where * SCALE_
    variance_term = sum_where_centered * SCALE_ * invstd * invstd
    out = (where_self - centered * variance_term - mean_term) * (invstd * affine_weight)
    tl.store(out_ptr + linear, out, mask=active)

    tl.store(
        out_scale_grad_ptr + c,
        sum_where_centered * invstd,
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
    arg280_1: torch.Tensor,
    getitem_195: torch.Tensor,
    getitem_201: torch.Tensor,
    arg274_1: torch.Tensor,
    arg275_1: torch.Tensor,
    arg276_1: torch.Tensor,
    arg57_1: torch.Tensor,
    arg58_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert _shape_param_0 == [N, C_IN, H, W]
    assert arg280_1.shape == (N, C_IN, 1, 1)
    assert getitem_195.shape == (N, C_IN, H, W)
    assert getitem_201.shape == (N, C_IN, 1, 1)
    assert arg274_1.shape == (N, C, H, W)
    assert arg275_1.shape == (1, C, 1, 1)
    assert arg276_1.shape == (1, C, 1, 1)
    assert arg57_1.shape == (C,)
    assert arg58_1.shape == (C,)
    assert full.shape == ()
    assert arg280_1.is_contiguous()
    assert getitem_195.is_contiguous()
    assert getitem_201.is_contiguous()
    assert arg274_1.is_contiguous()
    assert arg275_1.is_contiguous()
    assert arg276_1.is_contiguous()
    assert arg57_1.is_contiguous()
    assert arg58_1.is_contiguous()
    assert full.is_contiguous()

    sum_where = torch.zeros((C,), device=arg274_1.device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=arg274_1.device, dtype=torch.float32)

    _full_producer_dual_reduce_kernel[(C, triton.cdiv(TOTAL_SPATIAL, REDUCE_BLOCK_K))](
        arg280_1,
        getitem_195,
        getitem_201,
        arg274_1,
        arg275_1,
        arg276_1,
        arg57_1,
        arg58_1,
        full,
        sum_where,
        sum_where_centered,
        C_IN_=C_IN,
        C_=C,
        SLICE_START_=SLICE_START,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        BLOCK_K=REDUCE_BLOCK_K,
        num_warps=8,
    )

    out = torch.empty((N, C, H, W), device=arg274_1.device, dtype=torch.float32)
    out_scale_grad = torch.empty((C,), device=arg274_1.device, dtype=torch.float32)

    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK_ELEMS),)](
        arg280_1,
        getitem_195,
        getitem_201,
        arg274_1,
        arg275_1,
        arg276_1,
        arg57_1,
        arg58_1,
        full,
        sum_where,
        sum_where_centered,
        out,
        out_scale_grad,
        C_IN_=C_IN,
        C_=C,
        SLICE_START_=SLICE_START,
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
        ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} allclose={value_ok}"
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
        oracle_us = (
            triton.testing.do_bench(
                lambda: oracle_full(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            )
            * 1000.0
        )

    print(f"oracle_full full-scope cooperative split-k: {oracle_us:.3f} us")

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
            compiled_us = (
                triton.testing.do_bench(
                    lambda: compiled(*inputs),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                )
                * 1000.0
            )
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
