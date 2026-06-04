"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full DistillGPT2 layer-norm-backward/dropout-mask return tuple by row-tiling the `[16384, 768]` producer, reducing each row's hidden-dimension summaries for the masked gradient path, and cooperatively accumulating all three returned `[768]` column sums from the same row-tiled pass, whereas Inductor currently emits separate generic reductions and pointwise kernels for the row sums, sibling `sum([0, 1])` reductions, mask conversion/multiplication, and final masked-gradient column sum over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local reductions feeding a dependent column reduction with sibling column reductions in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward reductions across row tiles, keep row-local summaries in the producer, and finalize multiple column accumulators including dependent masked-gradient sums."""
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


REPRO_ID = "sum_sum_sum_117551af918e"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_distillgpt2_train_003_72fdc3f1"

BATCH = 32
SEQ = 512
M = BATCH * SEQ
D = 768

TILE_M = 8
TILE_D = 1024
FINAL_BLOCK_D = 16
FINAL_BLOCK_TILES = 256



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    moved: list[object] = []
    for value in module.make_inputs():
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm_1: torch.Tensor,
    arg150_1: torch.Tensor,
    arg38_1: torch.Tensor,
    arg117_1: torch.Tensor,
    arg119_1: torch.Tensor,
    arg116_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    add = arg150_1 + mm_1.view(_shape_param_0)
    weighted = add * arg38_1
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * arg117_1).sum(dim=2, keepdim=True)
    grad = arg119_1 * (weighted * D - row_sum - arg117_1 * row_dot)

    out0 = (add * arg117_1).sum(dim=(0, 1))
    out1 = add.sum(dim=(0, 1))
    masked = grad * arg116_1.to(torch.float32)
    out2 = masked.view(_shape_param_1).sum(dim=0, keepdim=True).view(_shape_param_2)
    return out0, out1, out2


if triton is not None:

    @triton.jit
    def _row_partial_reduce_kernel(
        mm_ptr,
        residual_ptr,
        gamma_ptr,
        xhat_ptr,
        scale_ptr,
        mask_ptr,
        partial_add_xhat_ptr,
        partial_add_ptr,
        partial_masked_grad_ptr,
        M_: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        tile_m = tl.program_id(0)
        m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d = tl.arange(0, BLOCK_D)
        m_mask = m < M_
        d_mask = d < D_
        mask = m_mask[:, None] & d_mask[None, :]
        offsets = m[:, None] * D_ + d[None, :]

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        xhat = tl.load(xhat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
        keep = tl.load(mask_ptr + offsets, mask=mask, other=0).to(tl.float32)

        add = residual + mm
        weighted = add * gamma[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * xhat, 0.0), axis=1)
        grad = scale[:, None] * (
            weighted * D_ - row_sum[:, None] - xhat * row_dot[:, None]
        )

        sum_add_xhat = tl.sum(tl.where(mask, add * xhat, 0.0), axis=0)
        sum_add = tl.sum(tl.where(mask, add, 0.0), axis=0)
        sum_masked_grad = tl.sum(tl.where(mask, grad * keep, 0.0), axis=0)

        partial_offsets = tile_m * D_ + d
        tl.store(partial_add_xhat_ptr + partial_offsets, sum_add_xhat, mask=d_mask)
        tl.store(partial_add_ptr + partial_offsets, sum_add, mask=d_mask)
        tl.store(
            partial_masked_grad_ptr + partial_offsets,
            sum_masked_grad,
            mask=d_mask,
        )


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_add_xhat_ptr,
        partial_add_ptr,
        partial_masked_grad_ptr,
        out_add_xhat_ptr,
        out_add_ptr,
        out_masked_grad_ptr,
        NUM_M_TILES: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d < D_
        tile_offsets = tl.arange(0, BLOCK_TILES)

        acc_add_xhat = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_add = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_masked_grad = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for tile_start in range(0, NUM_M_TILES, BLOCK_TILES):
            tiles = tile_start + tile_offsets
            mask = (tiles[:, None] < NUM_M_TILES) & d_mask[None, :]
            offsets = tiles[:, None] * D_ + d[None, :]

            add_xhat = tl.load(
                partial_add_xhat_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            add = tl.load(partial_add_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            masked_grad = tl.load(
                partial_masked_grad_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)

            acc_add_xhat += tl.sum(add_xhat, axis=0)
            acc_add += tl.sum(add, axis=0)
            acc_masked_grad += tl.sum(masked_grad, axis=0)

        tl.store(out_add_xhat_ptr + d, acc_add_xhat, mask=d_mask)
        tl.store(out_add_ptr + d, acc_add, mask=d_mask)
        tl.store(out_masked_grad_ptr + d, acc_masked_grad, mask=d_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_1,
        arg150_1,
        arg38_1,
        arg117_1,
        arg119_1,
        arg116_1,
        *_shape_params,
    ) = inputs

    return (
        mm_1.reshape(M, D).contiguous(),
        arg150_1.reshape(M, D).contiguous(),
        arg38_1.contiguous(),
        arg117_1.reshape(M, D).contiguous(),
        arg119_1.reshape(M).contiguous(),
        arg116_1.reshape(M, D).contiguous(),
    )


def oracle_triton_prepared(
    mm_md: torch.Tensor,
    residual_md: torch.Tensor,
    gamma_d: torch.Tensor,
    xhat_md: torch.Tensor,
    scale_m: torch.Tensor,
    mask_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm_md.shape == (M, D)
    assert residual_md.shape == (M, D)
    assert gamma_d.shape == (D,)
    assert xhat_md.shape == (M, D)
    assert scale_m.shape == (M,)
    assert mask_md.shape == (M, D)
    assert mm_md.is_contiguous()
    assert residual_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert xhat_md.is_contiguous()
    assert scale_m.is_contiguous()
    assert mask_md.is_contiguous()

    device = mm_md.device
    num_m_tiles = triton.cdiv(M, TILE_M)
    partial_add_xhat = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_add = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_masked_grad = torch.empty(
        (num_m_tiles, D), device=device, dtype=torch.float32
    )

    _row_partial_reduce_kernel[(num_m_tiles,)](
        mm_md,
        residual_md,
        gamma_d,
        xhat_md,
        scale_m,
        mask_md,
        partial_add_xhat,
        partial_add,
        partial_masked_grad,
        M_=M,
        D_=D,
        BLOCK_M=TILE_M,
        BLOCK_D=TILE_D,
        num_warps=8,
    )

    out_add_xhat = torch.empty((D,), device=device, dtype=torch.float32)
    out_add = torch.empty((D,), device=device, dtype=torch.float32)
    out_masked_grad = torch.empty((D,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_add_xhat,
        partial_add,
        partial_masked_grad,
        out_add_xhat,
        out_add,
        out_masked_grad,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=8,
    )

    return out_add_xhat, out_add, out_masked_grad


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = _as_tuple(oracle_full(*inputs, impl=impl))
        synchronize(device)

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs, max_rel = math.inf, math.inf
            value_ok = False
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        output_ok = shape_ok and value_ok and dtype_ok and stride_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} allclose={value_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def run_bench(device: torch.device, impl: str, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = impl
    if actual_impl == "auto":
        actual_impl = "triton" if device.type == "cuda" and triton is not None else "torch"

    logical_read_bytes = (
        M * D * 4
        + M * D * 4
        + D * 4
        + M * D * 4
        + M * 4
        + M * D
    )
    logical_write_bytes = D * 4 * 3

    with torch.no_grad():
        oracle_full(*inputs, impl=actual_impl)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_full(*inputs, impl=actual_impl),
            device,
            warmup,
            rep,
        )

    print(
        f"oracle_full cooperative split-k DistillGPT2 tuple: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device}"
    )
    print(f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(
        device=device,
        impl=args.impl,
        rtol=args.rtol,
        atol=args.atol,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
