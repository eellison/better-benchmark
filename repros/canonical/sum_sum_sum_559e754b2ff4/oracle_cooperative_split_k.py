"""
Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle split-K-reduces the GPT-J layer-norm backward row summaries over the 4096-wide hidden dimension, then has the gradient producer write the returned `[4096, 128]` transpose source while cooperatively accumulating the three `[4096]` column reductions, whereas Inductor currently emits generic row reductions, pointwise materialization/permute work, and separate column reductions over the same layer-norm gradient; Inductor cannot do this today because the scheduler has no split-K multi-output reduction template that can coordinate row-summary partials and row-tiled side-output producers with sibling column accumulators; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward reduction dimensions, finalize shared row partials, and fuse side-output stores with cooperative column reductions.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_559e754b2ff4"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

M = 128
D = 4096


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@triton.jit
def _row_dual_partial_kernel(
    mm0_ptr,
    mm1_ptr,
    mm2_ptr,
    mm3_ptr,
    weight_ptr,
    xhat_ptr,
    partial_row_sum_ptr,
    partial_row_dot_ptr,
    D_: tl.constexpr,
    NUM_D_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    d = tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d < D_
    offsets = row * D_ + d

    mm0 = tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm1 = tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm2 = tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm3 = tl.load(mm3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + d, mask=mask, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    summed = mm0 + mm1 + mm2 + mm3
    weighted = summed * weight

    out_offset = row * NUM_D_TILES + tile
    tl.store(partial_row_sum_ptr + out_offset, tl.sum(weighted, axis=0))
    tl.store(partial_row_dot_ptr + out_offset, tl.sum(weighted * xhat, axis=0))


@triton.jit
def _finalize_row_partials_kernel(
    partial_row_sum_ptr,
    partial_row_dot_ptr,
    row_sum_ptr,
    row_dot_ptr,
    NUM_D_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK_TILES)
    mask = tile < NUM_D_TILES
    offsets = row * NUM_D_TILES + tile

    row_sum = tl.load(partial_row_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_dot = tl.load(partial_row_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(row_sum_ptr + row, tl.sum(row_sum, axis=0))
    tl.store(row_dot_ptr + row, tl.sum(row_dot, axis=0))


@triton.jit
def _grad_store_and_column_reduce_kernel(
    mm0_ptr,
    mm1_ptr,
    mm2_ptr,
    mm3_ptr,
    weight_ptr,
    xhat_ptr,
    scale_ptr,
    residual_ptr,
    row_sum_ptr,
    row_dot_ptr,
    out_sum_x_xhat_ptr,
    out_sum_x_ptr,
    grad_md_ptr,
    out_sum_grad_ptr,
    M_: tl.constexpr,
    D_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    m_mask = m < M_
    d_mask = d < D_
    mask = m_mask[:, None] & d_mask[None, :]
    offsets = m[:, None] * D_ + d[None, :]

    mm0 = tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm1 = tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm2 = tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm3 = tl.load(mm3_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + m, mask=m_mask, other=0.0).to(tl.float32)

    summed = mm0 + mm1 + mm2 + mm3
    weighted = summed * weight[None, :]
    grad = residual + scale[:, None] * (
        weighted * D_ - row_sum[:, None] - xhat * row_dot[:, None]
    )

    tl.store(grad_md_ptr + offsets, grad, mask=mask)

    sum_x_xhat = tl.sum(tl.where(mask, summed * xhat, 0.0), axis=0)
    sum_x = tl.sum(tl.where(mask, summed, 0.0), axis=0)
    sum_grad = tl.sum(tl.where(mask, grad, 0.0), axis=0)

    tl.atomic_add(out_sum_x_xhat_ptr + d, sum_x_xhat, sem="relaxed", mask=d_mask)
    tl.atomic_add(out_sum_x_ptr + d, sum_x, sem="relaxed", mask=d_mask)
    tl.atomic_add(out_sum_grad_ptr + d, sum_grad, sem="relaxed", mask=d_mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_316,
        mm_321,
        mm_323,
        mm_325,
        arg8_1,
        arg216_1,
        arg628_1,
        add_364,
        *_shape_params,
    ) = inputs

    return (
        mm_316.contiguous(),
        mm_321.contiguous(),
        mm_323.contiguous(),
        mm_325.contiguous(),
        arg8_1.contiguous(),
        arg216_1.reshape(M, D).contiguous(),
        arg628_1.reshape(M).contiguous(),
        add_364.reshape(M, D).contiguous(),
    )


def oracle_fused(
    mm_316: torch.Tensor,
    mm_321: torch.Tensor,
    mm_323: torch.Tensor,
    mm_325: torch.Tensor,
    weight: torch.Tensor,
    xhat_md: torch.Tensor,
    scale_m: torch.Tensor,
    residual_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_316.shape == (M, D)
    assert mm_321.shape == (M, D)
    assert mm_323.shape == (M, D)
    assert mm_325.shape == (M, D)
    assert weight.shape == (D,)
    assert xhat_md.shape == (M, D)
    assert scale_m.shape == (M,)
    assert residual_md.shape == (M, D)

    device = mm_316.device

    block_row_d = 256
    num_d_tiles = triton.cdiv(D, block_row_d)
    partial_row_sum = torch.empty((M, num_d_tiles), device=device, dtype=torch.float32)
    partial_row_dot = torch.empty((M, num_d_tiles), device=device, dtype=torch.float32)
    _row_dual_partial_kernel[(M, num_d_tiles)](
        mm_316,
        mm_321,
        mm_323,
        mm_325,
        weight,
        xhat_md,
        partial_row_sum,
        partial_row_dot,
        D_=D,
        NUM_D_TILES=num_d_tiles,
        BLOCK_D=block_row_d,
        num_warps=8,
    )

    row_sum = torch.empty((M,), device=device, dtype=torch.float32)
    row_dot = torch.empty((M,), device=device, dtype=torch.float32)
    _finalize_row_partials_kernel[(M,)](
        partial_row_sum,
        partial_row_dot,
        row_sum,
        row_dot,
        NUM_D_TILES=num_d_tiles,
        BLOCK_TILES=16,
        num_warps=1,
    )

    sum_x_xhat = torch.zeros((D,), device=device, dtype=torch.float32)
    sum_x = torch.zeros((D,), device=device, dtype=torch.float32)
    grad_md = torch.empty((M, D), device=device, dtype=torch.float32)
    sum_grad = torch.zeros((D,), device=device, dtype=torch.float32)

    block_m = 16
    block_d = 64
    grid = (triton.cdiv(M, block_m), triton.cdiv(D, block_d))
    _grad_store_and_column_reduce_kernel[grid](
        mm_316,
        mm_321,
        mm_323,
        mm_325,
        weight,
        xhat_md,
        scale_m,
        residual_md,
        row_sum,
        row_dot,
        sum_x_xhat,
        sum_x,
        grad_md,
        sum_grad,
        M_=M,
        D_=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )

    return sum_x_xhat, sum_x, grad_md.permute(1, 0), sum_grad


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
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
        oracle_inputs = prepare_oracle_inputs(*inputs)
        actual = oracle_fused(*oracle_inputs)
        torch.cuda.synchronize()

    ok = True
    for idx, (got, expected) in enumerate(zip(actual, ref)):
        max_abs, max_rel = _max_diff(got, expected)
        output_ok = torch.allclose(got.float(), expected.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == expected.stride()
        ok = ok and output_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={output_ok} stride_match={stride_ok}"
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

    inputs = make_inputs()
    with torch.no_grad():
        oracle_inputs = prepare_oracle_inputs(*inputs)
        oracle_fused(*oracle_inputs)
        torch.cuda.synchronize()

        oracle_us = triton.testing.do_bench(
            lambda: oracle_fused(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    print(f"oracle_fused cooperative layernorm reduction: {oracle_us:.3f} us")

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
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=3e-2)
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
