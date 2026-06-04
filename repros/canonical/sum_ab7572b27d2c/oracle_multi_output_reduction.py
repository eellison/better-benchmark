"""
Full-scope Triton oracle for sum_ab7572b27d2c (LayoutLM attention layout +
feature sum).

Gap diagnosis (classification: SCHEDULER_FUSION): This oracle covers the full
compiled repro.py scope, not just the reduction subset: it consumes the original
strided `[32, 12, 512, 64]` input, writes the contiguous clone backing storage
for the returned `[768, 16384]` strided view, and computes the `[768]` feature
sum from the same input pass. Inductor currently emits this as a layout clone
plus a separate reduction because the scheduler does not fuse a materializing
layout-copy output with a reduction output that share the same source iteration
space but expose different result layouts. This is a SCHEDULER_FUSION gap if
the full-scope Triton timing beats both required compile configurations; if it
does not, keep this artifact as diagnosis-only rather than a true floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_ab7572b27d2c"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
HEADS = 12
SEQ = 512
HEAD_DIM = 64
ROWS = BATCH * SEQ
FEATURES = HEADS * HEAD_DIM



@triton.jit
def _copy_reduce_partials_kernel(
    x_ptr,
    clone_ptr,
    partials_ptr,
    x_stride_b: tl.constexpr,
    x_stride_h: tl.constexpr,
    x_stride_s: tl.constexpr,
    x_stride_d: tl.constexpr,
    SEQ_: tl.constexpr,
    HEAD_DIM_: tl.constexpr,
    ROWS_: tl.constexpr,
    FEATURES_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    feature_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    batch = row_offsets // SEQ_
    seq = row_offsets - batch * SEQ_
    head = feature_offsets // HEAD_DIM_
    dim = feature_offsets - head * HEAD_DIM_

    x_offsets = (
        batch[:, None] * x_stride_b
        + head[None, :] * x_stride_h
        + seq[:, None] * x_stride_s
        + dim[None, :] * x_stride_d
    )
    clone_offsets = row_offsets[:, None] * FEATURES_ + feature_offsets[None, :]
    mask = (row_offsets[:, None] < ROWS_) & (feature_offsets[None, :] < FEATURES_)

    values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(clone_ptr + clone_offsets, values, mask=mask)

    partial = tl.sum(values, axis=0)
    tl.store(
        partials_ptr + pid_m * FEATURES_ + feature_offsets,
        partial,
        mask=feature_offsets < FEATURES_,
    )


@triton.jit
def _finish_partials_kernel(
    partials_ptr,
    sum_ptr,
    num_row_blocks: tl.constexpr,
    FEATURES_: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    feature = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_R)
    mask = row_blocks < num_row_blocks
    values = tl.load(
        partials_ptr + row_blocks * FEATURES_ + feature,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(sum_ptr + feature, tl.sum(values, axis=0))


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


def oracle_fused(
    getitem_45: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert getitem_45.shape == (BATCH, HEADS, SEQ, HEAD_DIM)
    assert getitem_45.dtype is torch.float32

    clone_storage = torch.empty((ROWS, FEATURES), device=getitem_45.device, dtype=torch.float32)
    sums = torch.empty((FEATURES,), device=getitem_45.device, dtype=torch.float32)

    block_m = 64
    block_n = 32
    num_row_blocks = triton.cdiv(ROWS, block_m)
    partials = torch.empty((num_row_blocks, FEATURES), device=getitem_45.device, dtype=torch.float32)

    _copy_reduce_partials_kernel[(num_row_blocks, triton.cdiv(FEATURES, block_n))](
        getitem_45,
        clone_storage,
        partials,
        x_stride_b=getitem_45.stride(0),
        x_stride_h=getitem_45.stride(1),
        x_stride_s=getitem_45.stride(2),
        x_stride_d=getitem_45.stride(3),
        SEQ_=SEQ,
        HEAD_DIM_=HEAD_DIM,
        ROWS_=ROWS,
        FEATURES_=FEATURES,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )
    _finish_partials_kernel[(FEATURES,)](
        partials,
        sums,
        num_row_blocks=num_row_blocks,
        FEATURES_=FEATURES,
        BLOCK_R=triton.next_power_of_2(num_row_blocks),
        num_warps=8,
    )

    layout_view = torch.as_strided(clone_storage, (FEATURES, ROWS), (1, FEATURES))
    return layout_view, sums


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


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
        ref = reference_outputs(inputs)
        actual = oracle_fused(*inputs)
        torch.cuda.synchronize()

    ok = True
    for idx, (actual_tensor, ref_tensor) in enumerate(zip(actual, ref, strict=True)):
        max_abs, max_rel = _max_diff(actual_tensor, ref_tensor)
        output_ok = torch.allclose(actual_tensor.float(), ref_tensor.float(), rtol=rtol, atol=atol)
        stride_ok = actual_tensor.stride() == ref_tensor.stride()
        shape_ok = actual_tensor.shape == ref_tensor.shape
        dtype_ok = actual_tensor.dtype == ref_tensor.dtype
        tensor_ok = output_ok and stride_ok and shape_ok and dtype_ok
        ok = ok and tensor_ok
        print(
            f"output[{idx}]: shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} "
            f"dtype={actual_tensor.dtype} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} shape_match={shape_ok} stride_match={stride_ok} "
            f"dtype_match={dtype_ok}"
        )
    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def _compile_with_config(model: torch.nn.Module, inputs: tuple[object, ...], config: dict[str, object]):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        for _ in range(3):
            compiled(*inputs)
        torch.cuda.synchronize()
    return compiled


def run_bench(rep: int, warmup: int, no_compile: bool) -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    timings: dict[str, float] = {}
    with torch.no_grad():
        oracle_fused(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    timings["oracle_fused"] = oracle_us
    print(f"oracle_fused full-scope layout+sum: {oracle_us:.3f} us")

    if no_compile:
        return timings

    module = _load_repro_module()
    compile_configs = [
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
        timings[label] = compiled_us
        print(f"torch.compile {label}: {compiled_us:.3f} us")

    if oracle_us < min(timings["coordinate_descent_tuning"], timings["combo_looped_cd"]):
        print("Valid floor: yes")
    else:
        print("Valid floor: no (diagnosis-only; oracle does not beat both compile configs)")
    return timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true", help="only benchmark the oracle")
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
