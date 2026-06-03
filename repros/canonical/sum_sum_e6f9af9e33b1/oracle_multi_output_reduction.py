"""
Full-scope oracle for sum_sum_e6f9af9e33b1 (DenseNet BN-backward tail).

Gap diagnosis (classification: SCHEDULER_FUSION): The timed oracle consumes the same 15 original inputs as repro.py and returns the same `[256]` vector plus `[64, 32, 28, 28]` residual-add slice; it differs from Inductor by fusing the ReLU-mask `where`, channel centering, and two sibling channel reductions across all C=256 BN channels, finalizing both sums while writing the returned vector, and then computing only live epilogue channels 224:256 while adding the eight upstream residual slices. Inductor cannot do this today because its scheduler keeps the residual slice chain, sibling reductions, dependent BN-backward epilogue, and final slice/add as separate schedulable regions rather than one multi-output reduction with a dependent slice-limited epilogue. The fix class is SCHEDULER_FUSION: teach the scheduler/codegen to fuse compatible sibling reductions with a dependent live-channel epilogue and keep the residual slice chain inside that epilogue.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_e6f9af9e33b1"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 256
H = 28
W = 28
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 224
SLICE_C = C - SLICE_START
SCALE = 1.992984693877551e-05

sys.path.insert(0, str(REPO_ROOT))


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
    invstd_ptr,
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

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    tl.store(sum1_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(vector_out_ptr + c, sum2 * invstd)


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
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
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

    residual_offsets0 = n * (512 * HW_) + c * HW_ + hw
    residual_offsets1 = n * (480 * HW_) + c * HW_ + hw
    residual_offsets2 = n * (448 * HW_) + c * HW_ + hw
    residual_offsets3 = n * (416 * HW_) + c * HW_ + hw
    residual_offsets4 = n * (384 * HW_) + c * HW_ + hw
    residual_offsets5 = n * (352 * HW_) + c * HW_ + hw
    residual_offsets6 = n * (320 * HW_) + c * HW_ + hw
    residual_offsets7 = n * (288 * HW_) + c * HW_ + hw

    residual = tl.load(residual0_ptr + residual_offsets0, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual1_ptr + residual_offsets1, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual2_ptr + residual_offsets2, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual3_ptr + residual_offsets3, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual4_ptr + residual_offsets4, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual5_ptr + residual_offsets5, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual6_ptr + residual_offsets6, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual7_ptr + residual_offsets7, mask=active, other=0.0).to(tl.float32)

    mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    source = tl.load(source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)

    where_self = tl.where(mask_input <= 0.0, full_value, source)
    centered = centered_source - mean
    mean_term = sum1 * SCALE_
    variance_term = sum2 * SCALE_ * invstd * invstd
    affine_term = invstd * affine_weight
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
    mul_747: torch.Tensor,
    mul_765: torch.Tensor,
    mul_783: torch.Tensor,
    mul_801: torch.Tensor,
    mul_819: torch.Tensor,
    mul_837: torch.Tensor,
    mul_855: torch.Tensor,
    mul_873: torch.Tensor,
    arg314_1: torch.Tensor,
    full: torch.Tensor,
    getitem_291: torch.Tensor,
    arg312_1: torch.Tensor,
    arg709_1: torch.Tensor,
    arg313_1: torch.Tensor,
    arg46_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    residuals = (
        mul_747,
        mul_765,
        mul_783,
        mul_801,
        mul_819,
        mul_837,
        mul_855,
        mul_873,
    )
    expected_residual_channels = (512, 480, 448, 416, 384, 352, 320, 288)
    for residual, residual_c in zip(residuals, expected_residual_channels):
        assert residual.shape == (N, residual_c, H, W)
        assert residual.is_contiguous()
    assert arg314_1.shape == (N, C, H, W)
    assert getitem_291.shape == (N, C, H, W)
    assert arg312_1.shape == (N, C, H, W)
    assert arg709_1.shape == (1, C, 1, 1)
    assert arg313_1.shape == (C,)
    assert arg46_1.shape == (C,)
    assert arg314_1.is_contiguous()
    assert getitem_291.is_contiguous()
    assert arg312_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((C, num_tiles), device=arg314_1.device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, num_tiles), device=arg314_1.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=arg314_1.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=arg314_1.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=arg314_1.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg314_1,
        full,
        getitem_291,
        arg312_1,
        arg709_1,
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
        arg313_1,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )

    add_out = torch.empty((N, SLICE_C, H, W), device=arg314_1.device, dtype=torch.float32)
    block_elems = 1024
    numel_slice = N * SLICE_C * HW
    _slice_add_epilogue_kernel[(triton.cdiv(numel_slice, block_elems),)](
        mul_747,
        mul_765,
        mul_783,
        mul_801,
        mul_819,
        mul_837,
        mul_855,
        mul_873,
        arg314_1,
        full,
        getitem_291,
        arg312_1,
        arg709_1,
        arg313_1,
        arg46_1,
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
    print(f"oracle_fused full-scope dual reduction + 8-slice epilogue: {oracle_us:.3f} us")

    if no_compile:
        return

    module = _load_repro_module()
    compile_configs = [
        ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
        (
            (
                "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
                "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3"
            ),
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
