"""
Full-scope oracle for sum_sum_100fff569d2e (DenseNet BN-backward tail).

Gap diagnosis (classification: SCHEDULER_FUSION): the timed oracle consumes the
same eight original inputs and returns the same `[224]` vector plus
`[64, 32, 56, 56]` add slice as repro.py. It fuses the relu-mask `where`,
channel centering, and sibling channel reductions across all C=224 BN channels
into one split-K Triton reduction, finalizes both channel sums while writing the
returned vector, then computes only the live epilogue channels 192:224 and adds
`mul_972[:, 192:224]`. Inductor currently schedules the residual slice, sibling
reductions, dependent BN-backward epilogue, and final slice/add as ordinary graph
nodes rather than one full-scope multi-output reduction template that shares the
masked/centered producer and sinks the side output to the returned channel range;
if local measurements show this full-scope oracle slower than tuned compile,
report it as diagnosis-only rather than a performance floor.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_100fff569d2e"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
RESIDUAL_C = 256
C = 224
H = 56
W = 56
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 192
SLICE_C = C - SLICE_START
SCALE = 4.982461734693877e-06

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
    residual_ptr,
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
    RESIDUAL_C_: tl.constexpr,
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
    residual_offsets = n * (RESIDUAL_C_ * HW_) + c * HW_ + hw

    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0).to(tl.float32)
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
    mul_972: torch.Tensor,
    arg280_1: torch.Tensor,
    full: torch.Tensor,
    getitem_324: torch.Tensor,
    arg278_1: torch.Tensor,
    arg720_1: torch.Tensor,
    arg279_1: torch.Tensor,
    arg24_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mul_972.shape == (N, RESIDUAL_C, H, W)
    assert arg280_1.shape == (N, C, H, W)
    assert getitem_324.shape == (N, C, H, W)
    assert arg278_1.shape == (N, C, H, W)
    assert arg720_1.shape == (1, C, 1, 1)
    assert arg279_1.shape == (C,)
    assert arg24_1.shape == (C,)
    assert mul_972.is_contiguous()
    assert arg280_1.is_contiguous()
    assert getitem_324.is_contiguous()
    assert arg278_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((C, num_tiles), device=arg280_1.device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, num_tiles), device=arg280_1.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=arg280_1.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=arg280_1.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=arg280_1.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg280_1,
        full,
        getitem_324,
        arg278_1,
        arg720_1,
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
        arg279_1,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
    )

    add_out = torch.empty((N, SLICE_C, H, W), device=arg280_1.device, dtype=torch.float32)
    block_elems = 256
    numel_slice = N * SLICE_C * HW
    _slice_add_epilogue_kernel[(triton.cdiv(numel_slice, block_elems),)](
        mul_972,
        arg280_1,
        full,
        getitem_324,
        arg278_1,
        arg720_1,
        arg279_1,
        arg24_1,
        sum1,
        sum2,
        add_out,
        NUMEL_SLICE_=numel_slice,
        RESIDUAL_C_=RESIDUAL_C,
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
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={output_ok}"
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
    print(f"oracle_fused full-scope dual reduction + residual slice epilogue: {oracle_us:.3f} us")

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
