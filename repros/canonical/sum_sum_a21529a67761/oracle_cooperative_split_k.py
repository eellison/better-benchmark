"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full MNASNet dropout/avgpool/ReLU plus batch-norm-backward return tuple by cooperatively split-K reducing the two per-channel BN summaries over the shared `(N, H, W)` domain and feeding the finalized summaries into a fused epilogue that writes both the contiguous `[256, 1280, 7, 7]` grad-input tensor and `[1280]` scale-gradient vector, whereas Inductor currently schedules the dropout/avgpool producer, ReLU mask, sibling channel reductions, and dependent BN-backward epilogue as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that can coordinate compatible small-output channel reductions with a dependent full-tensor epilogue while preserving returned layouts; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible channel reductions across the reduced N/H/W domain, combine partial summaries, and fuse the downstream vector and full-tensor epilogues without materializing the masked producer."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_a21529a67761"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 256
C = 1280
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_DROPOUT_KEEP = 1.25
INV_HW = 1.0 / HW
SCALE = 1.0 / TOTAL_SPATIAL

BLOCK_K = 1024
FINAL_BLOCK_C = 8
EPILOGUE_BLOCK_ELEMS = 256

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _masked_bn_partial_reduce_kernel(
    dropout_mask_ptr,
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    partial_sum_dy_ptr,
    partial_sum_dy_centered_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_DROPOUT_KEEP_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K_ + tl.arange(0, BLOCK_K_)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    x_offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    dropout_mask = tl.load(dropout_mask_ptr + mm_offsets, mask=active, other=0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    centered = x - mean
    affine = centered * rsqrt * weight + bias
    upstream = (mm * (dropout_mask * INV_DROPOUT_KEEP_)) * INV_HW_
    dy = tl.where(active, tl.where(affine <= 0.0, 0.0, upstream), 0.0)

    partial_offset = tile * C_ + c
    tl.store(partial_sum_dy_ptr + partial_offset, tl.sum(dy, axis=0))
    tl.store(partial_sum_dy_centered_ptr + partial_offset, tl.sum(dy * centered, axis=0))


@triton.jit
def _finalize_summaries_kernel(
    partial_sum_dy_ptr,
    partial_sum_dy_centered_ptr,
    rsqrt_ptr,
    sum_dy_ptr,
    sum_dy_centered_ptr,
    out_weight_grad_ptr,
    C_: tl.constexpr,
    NUM_TILES_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    tiles = tl.arange(0, BLOCK_TILES_)
    mask = (tiles[:, None] < NUM_TILES_) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    sum_dy = tl.sum(tl.load(partial_sum_dy_ptr + offsets, mask=mask, other=0.0), axis=0)
    sum_dy_centered = tl.sum(
        tl.load(partial_sum_dy_centered_ptr + offsets, mask=mask, other=0.0),
        axis=0,
    )

    c_mask = c < C_
    rsqrt = tl.load(rsqrt_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(sum_dy_ptr + c, sum_dy, mask=c_mask)
    tl.store(sum_dy_centered_ptr + c, sum_dy_centered, mask=c_mask)
    tl.store(out_weight_grad_ptr + c, sum_dy_centered * rsqrt, mask=c_mask)


@triton.jit
def _bn_input_epilogue_kernel(
    dropout_mask_ptr,
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    bias_ptr,
    sum_dy_ptr,
    sum_dy_centered_ptr,
    out_grad_input_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    INV_DROPOUT_KEEP_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS_: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
    active = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    mm_offsets = n * C_ + c

    dropout_mask = tl.load(dropout_mask_ptr + mm_offsets, mask=active, other=0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_dy = tl.load(sum_dy_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_dy_centered = tl.load(sum_dy_centered_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = x - mean
    affine = centered * rsqrt * weight + bias
    upstream = (mm * (dropout_mask * INV_DROPOUT_KEEP_)) * INV_HW_
    dy = tl.where(affine <= 0.0, 0.0, upstream)

    mean_term = sum_dy * SCALE_
    variance_term = sum_dy_centered * SCALE_ * rsqrt * rsqrt
    grad_input = (dy - centered * variance_term - mean_term) * (rsqrt * weight)
    tl.store(out_grad_input_ptr + offsets, grad_input, mask=active)


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
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def oracle_full(
    arg263_1: torch.Tensor,
    mm: torch.Tensor,
    arg260_1: torch.Tensor,
    arg261_1: torch.Tensor,
    arg262_1: torch.Tensor,
    arg104_1: torch.Tensor,
    arg105_1: torch.Tensor,
    _shape_param_0: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")
    if arg260_1.device.type != "cuda":
        raise RuntimeError("oracle_full expects CUDA inputs")
    assert arg263_1.shape == (N, C)
    assert mm.shape == (N, C)
    assert arg260_1.shape == (N, C, H, W)
    assert arg261_1.shape == (1, C, 1, 1)
    assert arg262_1.shape == (1, C, 1, 1)
    assert arg104_1.shape == (C,)
    assert arg105_1.shape == (C,)
    assert arg263_1.is_contiguous()
    assert mm.is_contiguous()
    assert arg260_1.is_contiguous()
    assert arg261_1.is_contiguous()
    assert arg262_1.is_contiguous()
    assert arg104_1.is_contiguous()
    assert arg105_1.is_contiguous()

    device = arg260_1.device
    mean = arg261_1.reshape(C)
    rsqrt = arg262_1.reshape(C)
    num_tiles = triton.cdiv(TOTAL_SPATIAL, BLOCK_K)
    block_tiles = triton.next_power_of_2(num_tiles)

    partial_sum_dy = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_sum_dy_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    sum_dy = torch.empty((C,), device=device, dtype=torch.float32)
    sum_dy_centered = torch.empty((C,), device=device, dtype=torch.float32)
    out_grad_input = torch.empty_like(arg260_1)
    out_weight_grad = torch.empty((C,), device=device, dtype=torch.float32)

    _masked_bn_partial_reduce_kernel[(C, num_tiles)](
        arg263_1,
        mm,
        arg260_1,
        mean,
        rsqrt,
        arg104_1,
        arg105_1,
        partial_sum_dy,
        partial_sum_dy_centered,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_DROPOUT_KEEP_=INV_DROPOUT_KEEP,
        INV_HW_=INV_HW,
        BLOCK_K_=BLOCK_K,
        num_warps=4,
    )

    _finalize_summaries_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_sum_dy,
        partial_sum_dy_centered,
        rsqrt,
        sum_dy,
        sum_dy_centered,
        out_weight_grad,
        C_=C,
        NUM_TILES_=num_tiles,
        BLOCK_TILES_=block_tiles,
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=4,
    )

    _bn_input_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK_ELEMS),)](
        arg263_1,
        mm,
        arg260_1,
        mean,
        rsqrt,
        arg104_1,
        arg105_1,
        sum_dy,
        sum_dy_centered,
        out_grad_input,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_DROPOUT_KEEP_=INV_DROPOUT_KEEP,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS_=EPILOGUE_BLOCK_ELEMS,
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
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

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
    print(f"oracle_full cooperative split-k BN backward: {oracle_us:.3f} us")

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
    parser.add_argument("--check", action="store_true", help="compare oracle against Repro.forward")
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
    if args.warmup < 0 or args.rep <= 0:
        parser.error("--warmup must be non-negative and --rep must be positive")

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(rep=args.rep, warmup=args.warmup, no_compile=args.no_compile)


if __name__ == "__main__":
    with torch.no_grad():
        main()
