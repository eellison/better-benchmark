"""
Full-scope oracle for sum_sum_8bc26e2e1883 (DenseNet BN-backward tail).

Gap diagnosis (classification: SCHEDULER_FUSION): The timed oracle consumes the
same 20 original inputs as repro.py and returns the same `[608]` vector plus
`[64, 32, 14, 14]` residual-add slice. It fuses the ReLU-mask `where`, channel
centering, and two sibling channel reductions across all C=608 BN channels into
one split-K Triton reduction, finalizes both sums while writing the returned
vector, then computes only the live epilogue channels 576:608 and adds the
thirteen upstream residual slices in the same returned output. Inductor today
schedules the residual slice chain, sibling reductions, dependent BN-backward
epilogue, and final slice/add as ordinary graph nodes, so it cannot form one
full-scope multi-output reduction template that shares the masked/centered
producer and sinks the side output to the live channel range. The fix is
SCHEDULER_FUSION: teach the scheduler/codegen to fuse compatible sibling
reductions with a dependent, slice-limited epilogue and keep the residual slice
chain inside that live epilogue.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_8bc26e2e1883"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 608
H = 14
W = 14
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 576
SLICE_C = C - SLICE_START
SCALE = 7.971938775510203e-05



@triton.jit
def _dual_reduce_partial_kernel(
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    num_tiles: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    offsets = n * (C_ * HW_) + c * HW_ + hw

    mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)

    where_self = tl.where(mask_input <= 0.0, full_value, source)
    where_self = tl.where(active, where_self, 0.0)
    centered = centered_source - mean

    partial_offset = c * num_tiles + tile
    tl.store(partial_sum1_ptr + partial_offset, tl.sum(where_self, axis=0))
    tl.store(partial_sum2_ptr + partial_offset, tl.sum(where_self * centered, axis=0))


@triton.jit
def _finalize_reduce_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    scale_ptr,
    sum1_ptr,
    sum2_ptr,
    vector_out_ptr,
    num_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    tile_mask = tiles < num_tiles
    partial_offsets = c * num_tiles + tiles

    sum1_values = tl.load(partial_sum1_ptr + partial_offsets, mask=tile_mask, other=0.0).to(tl.float32)
    sum2_values = tl.load(partial_sum2_ptr + partial_offsets, mask=tile_mask, other=0.0).to(tl.float32)
    sum1 = tl.sum(sum1_values, axis=0)
    sum2 = tl.sum(sum2_values, axis=0)

    scale_value = tl.load(scale_ptr + c).to(tl.float32)
    tl.store(sum1_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(vector_out_ptr + c, sum2 * scale_value)


@triton.jit
def _slice_add_epilogue_kernel(
    residual0_ptr,
    residual1_ptr,
    residual2_ptr,
    residual3_ptr,
    residual4_ptr,
    residual5_ptr,
    residual6_ptr,
    residual7_ptr,
    residual8_ptr,
    residual9_ptr,
    residual10_ptr,
    residual11_ptr,
    residual12_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    scale_ptr,
    affine_weight_ptr,
    sum1_ptr,
    sum2_ptr,
    add_out_ptr,
    NUMEL_SLICE_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = out_offsets < NUMEL_SLICE_

    hw = out_offsets % HW_
    slice_c = (out_offsets // HW_) % SLICE_C_
    n = out_offsets // (SLICE_C_ * HW_)
    c = SLICE_START_ + slice_c
    input_offsets = n * (C_ * HW_) + c * HW_ + hw

    residual_offsets0 = n * (1024 * HW_) + c * HW_ + hw
    residual_offsets1 = n * (992 * HW_) + c * HW_ + hw
    residual_offsets2 = n * (960 * HW_) + c * HW_ + hw
    residual_offsets3 = n * (928 * HW_) + c * HW_ + hw
    residual_offsets4 = n * (896 * HW_) + c * HW_ + hw
    residual_offsets5 = n * (864 * HW_) + c * HW_ + hw
    residual_offsets6 = n * (832 * HW_) + c * HW_ + hw
    residual_offsets7 = n * (800 * HW_) + c * HW_ + hw
    residual_offsets8 = n * (768 * HW_) + c * HW_ + hw
    residual_offsets9 = n * (736 * HW_) + c * HW_ + hw
    residual_offsets10 = n * (704 * HW_) + c * HW_ + hw
    residual_offsets11 = n * (672 * HW_) + c * HW_ + hw
    residual_offsets12 = n * (640 * HW_) + c * HW_ + hw

    residual = tl.load(residual0_ptr + residual_offsets0, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual1_ptr + residual_offsets1, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual2_ptr + residual_offsets2, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual3_ptr + residual_offsets3, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual4_ptr + residual_offsets4, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual5_ptr + residual_offsets5, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual6_ptr + residual_offsets6, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual7_ptr + residual_offsets7, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual8_ptr + residual_offsets8, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual9_ptr + residual_offsets9, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual10_ptr + residual_offsets10, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual11_ptr + residual_offsets11, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual12_ptr + residual_offsets12, mask=active, other=0.0).to(tl.float32)

    mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    source = tl.load(source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    scale_value = tl.load(scale_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)

    where_self = tl.where(mask_input <= 0.0, full_value, source)
    centered = centered_source - mean
    mean_term = sum1 * SCALE_
    variance_term = sum2 * SCALE_ * scale_value * scale_value
    affine_term = scale_value * affine_weight
    epilogue = (where_self - centered * variance_term - mean_term) * affine_term
    tl.store(add_out_ptr + out_offsets, residual + epilogue, mask=active)


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
    mul_306: torch.Tensor,
    mul_324: torch.Tensor,
    mul_342: torch.Tensor,
    mul_360: torch.Tensor,
    mul_378: torch.Tensor,
    mul_396: torch.Tensor,
    mul_414: torch.Tensor,
    mul_432: torch.Tensor,
    mul_450: torch.Tensor,
    mul_468: torch.Tensor,
    mul_486: torch.Tensor,
    mul_504: torch.Tensor,
    mul_522: torch.Tensor,
    arg432_1: torch.Tensor,
    full: torch.Tensor,
    getitem_174: torch.Tensor,
    arg430_1: torch.Tensor,
    arg670_1: torch.Tensor,
    arg431_1: torch.Tensor,
    arg124_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residuals = (
        mul_306,
        mul_324,
        mul_342,
        mul_360,
        mul_378,
        mul_396,
        mul_414,
        mul_432,
        mul_450,
        mul_468,
        mul_486,
        mul_504,
        mul_522,
    )
    expected_residual_channels = (1024, 992, 960, 928, 896, 864, 832, 800, 768, 736, 704, 672, 640)
    for residual, residual_c in zip(residuals, expected_residual_channels):
        assert residual.shape == (N, residual_c, H, W)
        assert residual.is_contiguous()
    assert arg432_1.shape == (N, C, H, W)
    assert getitem_174.shape == (N, C, H, W)
    assert arg430_1.shape == (N, C, H, W)
    assert arg670_1.shape == (1, C, 1, 1)
    assert arg431_1.shape == (C,)
    assert arg124_1.shape == (C,)
    assert arg432_1.is_contiguous()
    assert getitem_174.is_contiguous()
    assert arg430_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((C, num_tiles), device=arg432_1.device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, num_tiles), device=arg432_1.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=arg432_1.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=arg432_1.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=arg432_1.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg432_1,
        full,
        getitem_174,
        arg430_1,
        arg670_1,
        partial_sum1,
        partial_sum2,
        num_tiles=num_tiles,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=4,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _finalize_reduce_kernel[(C,)](
        partial_sum1,
        partial_sum2,
        arg431_1,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )

    add_out = torch.empty((N, SLICE_C, H, W), device=arg432_1.device, dtype=torch.float32)
    block_elems = 256
    numel_slice = N * SLICE_C * HW
    _slice_add_epilogue_kernel[(triton.cdiv(numel_slice, block_elems),)](
        mul_306,
        mul_324,
        mul_342,
        mul_360,
        mul_378,
        mul_396,
        mul_414,
        mul_432,
        mul_450,
        mul_468,
        mul_486,
        mul_504,
        mul_522,
        arg432_1,
        full,
        getitem_174,
        arg430_1,
        arg670_1,
        arg431_1,
        arg124_1,
        sum1,
        sum2,
        add_out,
        NUMEL_SLICE_=numel_slice,
        C_=C,
        HW_=HW,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return vector_out, add_out


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
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == expected.stride()
        ok = ok and output_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={output_ok} stride_match={stride_ok}"
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


def run_bench(rep: int, warmup: int, no_compile: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        oracle_fused(*inputs)
        torch.cuda.synchronize()
        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0
    print(f"oracle_fused full-scope dual reduction + 13-slice epilogue: {oracle_us:.3f} us")

    if no_compile:
        return

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
        print(f"torch.compile {label}: {compiled_us:.3f} us")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check against repro.py")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
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
