"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` BERT layer-norm/dropout-backward return tuple by row-tiling the `[16384, 768]` producer, preserving the returned `[768, 16384]` transpose-view stride, and cooperatively accumulating all three returned `[768]` column reductions from the same row tiles, whereas Inductor currently schedules the row-local sums, dropout/mask gradient materialization and permute, and sibling `sum([0, 1])`/`sum([0])` reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that combines row-local reductions, a required transposed side-output store, and multiple sibling column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer mode that writes required side outputs while emitting partial or atomic column accumulators and finalizes them for compatible layer-norm-backward reductions."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps CPU-only syntax checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_1c9995bc990f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_bertformaskedlm_train_001_72fdc3f1"

BATCH = 32
SEQ = 512
M = BATCH * SEQ
D = 768
DROPOUT_SCALE = 1.1111111111111112

ROW_SPLIT = 29
XBLOCK = 1
BLOCK_D = 1024
FINAL_BLOCK_D = 16
FINAL_BLOCK_TILES = 1024
ROW_NUM_WARPS = 8
FINAL_NUM_WARPS = 8

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) and torch.cuda.is_available() else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    model = module.Repro().to(first_tensor.device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm_138: torch.Tensor,
    mul_330: torch.Tensor,
    arg8_1: torch.Tensor,
    arg115_1: torch.Tensor,
    arg318_1: torch.Tensor,
    arg114_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    add = mm_138.view(_shape_param_0) + mul_330
    weighted = add * arg8_1
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * arg115_1).sum(dim=2, keepdim=True)
    grad = arg318_1 * (weighted * D - row_sum - arg115_1 * row_dot)
    grad = grad * arg114_1.to(torch.float32) * DROPOUT_SCALE

    out_add_rhs = (add * arg115_1).sum(dim=(0, 1))
    out_add = add.sum(dim=(0, 1))
    grad_md = grad.view(_shape_param_1)
    out_grad_sum = grad_md.sum(dim=0, keepdim=True).view(_shape_param_2)
    return out_add_rhs, out_add, grad_md.permute(1, 0), out_grad_sum


if triton is not None:

    @triton.jit
    def _row_tile_kernel(
        mm_ptr,
        residual_ptr,
        gamma_ptr,
        rhs_ptr,
        row_scale_ptr,
        keep_ptr,
        partial_add_rhs_ptr,
        partial_add_ptr,
        partial_grad_ptr,
        grad_md_ptr,
        M_: tl.constexpr,
        D_: tl.constexpr,
        DROPOUT_SCALE_: tl.constexpr,
        ROW_SPLIT_: tl.constexpr,
        XBLOCK_: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        d = tl.arange(0, BLOCK_D)
        d_mask = d < D_

        gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        acc_add_rhs = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_add = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
            m = row_block * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
            m_mask = m < M_
            mask = m_mask[:, None] & d_mask[None, :]
            offsets = m[:, None] * D_ + d[None, :]

            mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            row_scale = tl.load(row_scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
            keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)

            add = mm + residual
            weighted = add * gamma[None, :]
            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
            grad = row_scale[:, None] * (
                weighted * D_ - row_sum[:, None] - rhs * row_dot[:, None]
            )
            grad = grad * keep * DROPOUT_SCALE_

            tl.store(grad_md_ptr + offsets, grad, mask=mask)
            acc_add_rhs += tl.sum(tl.where(mask, add * rhs, 0.0), axis=0)
            acc_add += tl.sum(tl.where(mask, add, 0.0), axis=0)
            acc_grad += tl.sum(tl.where(mask, grad, 0.0), axis=0)

        partial_offsets = row_block * D_ + d
        tl.store(partial_add_rhs_ptr + partial_offsets, acc_add_rhs, mask=d_mask)
        tl.store(partial_add_ptr + partial_offsets, acc_add, mask=d_mask)
        tl.store(partial_grad_ptr + partial_offsets, acc_grad, mask=d_mask)


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_add_rhs_ptr,
        partial_add_ptr,
        partial_grad_ptr,
        out_add_rhs_ptr,
        out_add_ptr,
        out_grad_ptr,
        NUM_M_TILES: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d < D_
        tiles = tl.arange(0, BLOCK_TILES)

        acc_add_rhs = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_add = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for tile_base in range(0, NUM_M_TILES, BLOCK_TILES):
            tile_ids = tile_base + tiles
            mask = (tile_ids[:, None] < NUM_M_TILES) & d_mask[None, :]
            offsets = tile_ids[:, None] * D_ + d[None, :]
            acc_add_rhs += tl.sum(
                tl.load(partial_add_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )
            acc_add += tl.sum(
                tl.load(partial_add_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )
            acc_grad += tl.sum(
                tl.load(partial_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
                axis=0,
            )

        tl.store(out_add_rhs_ptr + d, acc_add_rhs, mask=d_mask)
        tl.store(out_add_ptr + d, acc_add, mask=d_mask)
        tl.store(out_grad_ptr + d, acc_grad, mask=d_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_138,
        mul_330,
        arg8_1,
        arg115_1,
        arg318_1,
        arg114_1,
        *_shape_params,
    ) = inputs

    return (
        mm_138.reshape(M, D).contiguous(),
        mul_330.reshape(M, D).contiguous(),
        arg8_1.contiguous(),
        arg115_1.reshape(M, D).contiguous(),
        arg318_1.reshape(M).contiguous(),
        arg114_1.reshape(M, D).contiguous(),
    )


def oracle_triton_prepared(
    mm_md: torch.Tensor,
    residual_md: torch.Tensor,
    gamma_d: torch.Tensor,
    rhs_md: torch.Tensor,
    row_scale_m: torch.Tensor,
    keep_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm_md.shape == (M, D)
    assert residual_md.shape == (M, D)
    assert gamma_d.shape == (D,)
    assert rhs_md.shape == (M, D)
    assert row_scale_m.shape == (M,)
    assert keep_md.shape == (M, D)
    assert mm_md.is_contiguous()
    assert residual_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert rhs_md.is_contiguous()
    assert row_scale_m.is_contiguous()
    assert keep_md.is_contiguous()

    device = mm_md.device
    num_m_tiles = triton.cdiv(M, ROW_SPLIT)
    partial_add_rhs = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_add = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_grad = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    grad_md = torch.empty((M, D), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_m_tiles,)](
        mm_md,
        residual_md,
        gamma_d,
        rhs_md,
        row_scale_m,
        keep_md,
        partial_add_rhs,
        partial_add,
        partial_grad,
        grad_md,
        M_=M,
        D_=D,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_D=BLOCK_D,
        num_warps=ROW_NUM_WARPS,
    )

    out_add_rhs = torch.empty((D,), device=device, dtype=torch.float32)
    out_add = torch.empty((D,), device=device, dtype=torch.float32)
    out_grad = torch.empty((D,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_add_rhs,
        partial_add,
        partial_grad,
        out_add_rhs,
        out_add,
        out_grad,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=FINAL_NUM_WARPS,
    )

    return out_add_rhs, out_add, grad_md.t(), out_grad


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(impl: str, rtol: float, atol: float) -> bool:
    if impl in ("auto", "triton") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = _as_tuple(oracle_full(*inputs, impl=impl))
        synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = shape_ok and value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} allclose={value_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize()

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize()
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def run_bench(impl: str, warmup: int, rep: int) -> None:
    if impl in ("auto", "triton") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    actual_impl = impl
    if actual_impl == "auto":
        first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
        actual_impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"

    with torch.no_grad():
        oracle_full(*inputs, impl=actual_impl)
        synchronize()
        if actual_impl == "triton":
            oracle_us = (
                triton.testing.do_bench(
                    lambda: oracle_full(*inputs, impl=actual_impl),
                    warmup=warmup,
                    rep=rep,
                    return_mode="min",
                )
                * 1000.0
            )
        else:
            oracle_us = benchmark(lambda: oracle_full(*inputs, impl=actual_impl), warmup, rep)

    logical_read_bytes = (M * D * 5 + D + M) * 4 + M * D
    logical_write_bytes = (M * D + D * 3) * 4
    print(
        f"oracle_full cooperative split-k BERT layernorm/dropout backward: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL}"
    )
    print(f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    if args.check and not run_check(impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
