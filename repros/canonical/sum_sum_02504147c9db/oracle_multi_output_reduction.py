"""Full-scope oracle for sum_sum_02504147c9db (DenseNet BN-backward + pool tail).

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle consumes the same
nine original inputs as repro.py and returns the same `[64]` vector plus
contiguous `[128, 64, 32, 32]` avg-pool-backward tensor. It differs from
Inductor by sharing the masked ReLU-backward producer and centered activation
producer across two sibling channel reductions, then using the finalized sums in
a dependent epilogue that fuses the slice add and writes the structured
avg-pool-backward output directly. Inductor cannot do this today because the
scheduler does not form one multi-output reduction template spanning sibling
reductions, dependent BN-backward pointwise work, a channel slice, and the final
pool-backward layout expansion; the fix class is SCHEDULER_FUSION.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_02504147c9db"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
ADD_C = 80
C = 64
H = 16
W = 16
OUT_H = 32
OUT_W = 32
HW = H * W
OUT_HW = OUT_H * OUT_W
TOTAL_SPATIAL = N * HW
NUMEL_16 = N * C * HW
SCALE = 3.0517578125e-05

COMPILE_CONFIGS = [
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
    offsets = c * num_tiles + tiles

    sum1_values = tl.load(partial_sum1_ptr + offsets, mask=tile_mask, other=0.0).to(tl.float32)
    sum2_values = tl.load(partial_sum2_ptr + offsets, mask=tile_mask, other=0.0).to(tl.float32)
    sum1 = tl.sum(sum1_values, axis=0)
    sum2 = tl.sum(sum2_values, axis=0)
    scale_value = tl.load(scale_ptr + c).to(tl.float32)

    tl.store(sum1_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(vector_out_ptr + c, sum2 * scale_value)


@triton.jit
def _pool_epilogue_kernel(
    add_input_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    scale_ptr,
    affine_weight_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    NUMEL_16_: tl.constexpr,
    ADD_C_: tl.constexpr,
    C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    OUT_W_: tl.constexpr,
    HW_: tl.constexpr,
    OUT_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < NUMEL_16_

    hw = linear % HW_
    w = hw % W_
    h = hw // W_
    c = (linear // HW_) % C_
    n = linear // (C_ * HW_)

    input_offsets = n * (C_ * HW_) + c * HW_ + hw
    add_offsets = n * (ADD_C_ * HW_) + (c + 16) * HW_ + hw

    mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    source = tl.load(source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    residual = tl.load(add_input_ptr + add_offsets, mask=active, other=0.0).to(tl.float32)
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
    add_tensor = residual + (where_self - centered * variance_term - mean_term) * affine_term
    pooled_grad = add_tensor * 0.25

    out_base = n * (C_ * OUT_HW_) + c * OUT_HW_ + (h * 2) * OUT_W_ + w * 2
    tl.store(out_ptr + out_base, pooled_grad, mask=active)
    tl.store(out_ptr + out_base + 1, pooled_grad, mask=active)
    tl.store(out_ptr + out_base + OUT_W_, pooled_grad, mask=active)
    tl.store(out_ptr + out_base + OUT_W_ + 1, pooled_grad, mask=active)


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
    inputs = module.make_inputs()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) and x.device.type != "cuda" else x for x in inputs)


def oracle_fused(
    add_17: torch.Tensor,
    arg149_1: torch.Tensor,
    full: torch.Tensor,
    getitem_111: torch.Tensor,
    arg147_1: torch.Tensor,
    arg304_1: torch.Tensor,
    arg148_1: torch.Tensor,
    arg28_1: torch.Tensor,
    arg146_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert add_17.shape == (N, ADD_C, H, W)
    assert arg149_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_111.shape == (N, C, H, W)
    assert arg147_1.shape == (N, C, H, W)
    assert arg304_1.shape == (1, C, 1, 1)
    assert arg148_1.shape == (C,)
    assert arg28_1.shape == (C,)
    assert arg146_1.shape == (N, C, OUT_H, OUT_W)
    assert add_17.is_contiguous()
    assert arg149_1.is_contiguous()
    assert getitem_111.is_contiguous()
    assert arg147_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((C, num_tiles), device=add_17.device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, num_tiles), device=add_17.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=add_17.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=add_17.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=add_17.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg149_1,
        full,
        getitem_111,
        arg147_1,
        arg304_1,
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
        arg148_1,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    out = torch.empty((N, C, OUT_H, OUT_W), device=add_17.device, dtype=torch.float32)
    block_elems = 256
    _pool_epilogue_kernel[(triton.cdiv(NUMEL_16, block_elems),)](
        add_17,
        arg149_1,
        full,
        getitem_111,
        arg147_1,
        arg304_1,
        arg148_1,
        arg28_1,
        sum1,
        sum2,
        out,
        NUMEL_16_=NUMEL_16,
        ADD_C_=ADD_C,
        C_=C,
        H_=H,
        W_=W,
        OUT_W_=OUT_W,
        HW_=HW,
        OUT_HW_=OUT_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return vector_out, out


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
        output_ok = (
            got.shape == expected.shape
            and got.dtype == expected.dtype
            and got.stride() == expected.stride()
            and torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        )
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"dtype={got.dtype} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok}"
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
    print(f"oracle_fused full-scope dual reduction + pool epilogue: {oracle_us:.3f} us")

    if no_compile:
        return

    compile_times: list[tuple[str, float]] = []
    module = _load_repro_module()
    for label, config in COMPILE_CONFIGS:
        model = module.Repro().cuda()
        with torch.no_grad():
            compiled = _compile_with_config(model, inputs, config)
            compiled_us = triton.testing.do_bench(
                lambda: compiled(*inputs),
                warmup=warmup,
                rep=rep,
                return_mode="min",
            ) * 1000.0
        compile_times.append((label, compiled_us))
        print(f"torch.compile {label}: {compiled_us:.3f} us")

    if compile_times and not all(oracle_us < compiled_us for _, compiled_us in compile_times):
        print("diagnosis_only: oracle is not a true floor because a required compile config is faster")
    elif compile_times:
        print("valid_floor: oracle is faster than both required compile configs")


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
