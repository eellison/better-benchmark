"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Longformer layer-norm-backward return tuple by row-tiling the `[8,1024,768]` producer, folding the three `view([1024,8,768]).permute(1,0,2)` matmul buffers into each tile, writing the returned `[768,8192]` transposed masked-gradient side output, and cooperatively accumulating the two pre-mask `[768]` column sums plus the dependent masked-gradient `[768]` sum from the same row tiles, whereas Inductor currently materializes/schedules the permuted three-matmul add, row-local reductions, mask multiply, transpose side output, and sibling `sum([0,1])`/`sum([0])` reductions as separate generic pointwise and reduction kernels; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines non-contiguous row-tile producers, row-local layer-norm scalars, a required transposed side-output store, and multiple column accumulators in one coordinated producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: teach Inductor to tile compatible layer-norm-backward producers across token rows, emit/finalize shared column partials, and fuse the layout-mapped producer plus masked transpose store."""
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


REPRO_ID = "sum_sum_sum_3213336f4c0b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "hf_allenailongformerbase_train_005_f64748a8"

BATCH = 8
SEQ = 1024
M = BATCH * SEQ
D = 768

ROW_SPLIT = 16
XBLOCK = 1
BLOCK_D = 1024
FINAL_BLOCK_D = 16
FINAL_BLOCK_TILES = 1024

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


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    moved: list[object] = []
    for value in module.make_inputs():
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm_127: torch.Tensor,
    mm_129: torch.Tensor,
    mm_131: torch.Tensor,
    mul_279: torch.Tensor,
    arg8_1: torch.Tensor,
    arg109_1: torch.Tensor,
    arg297_1: torch.Tensor,
    arg108_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mm_sum = (
        mm_127.view(_shape_param_0)
        + mm_129.view(_shape_param_1)
        + mm_131.view(_shape_param_2)
    )
    x = mul_279 + mm_sum.permute(1, 0, 2)
    weighted = x * arg8_1
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * arg109_1).sum(dim=2, keepdim=True)
    grad = arg297_1 * (weighted * D - row_sum - arg109_1 * row_dot)
    masked_grad = grad * arg108_1.to(torch.float32)

    out_x_rhs = (x * arg109_1).sum(dim=(0, 1))
    out_x = x.sum(dim=(0, 1))
    grad_md = masked_grad.view(_shape_param_3)
    out_grad = grad_md.sum(dim=0, keepdim=True).view(_shape_param_4)
    return out_x_rhs, out_x, grad_md.t(), out_grad


if triton is not None:

    @triton.jit
    def _row_split_store_and_reduce_kernel(
        mm0_ptr,
        mm1_ptr,
        mm2_ptr,
        residual_ptr,
        gamma_ptr,
        rhs_ptr,
        scale_ptr,
        keep_ptr,
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_grad_ptr,
        grad_md_ptr,
        M_: tl.constexpr,
        D_: tl.constexpr,
        BATCH_: tl.constexpr,
        SEQ_: tl.constexpr,
        ROW_SPLIT_: tl.constexpr,
        XBLOCK_: tl.constexpr,
        BLOCK_D_: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        d = tl.arange(0, BLOCK_D_)
        d_mask = d < D_
        gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

        acc_x_rhs = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_D_,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_D_,), dtype=tl.float32)

        for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
            m = row_block * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
            m_mask = m < M_
            elem_mask = m_mask[:, None] & d_mask[None, :]

            batch = m // SEQ_
            seq = m - batch * SEQ_
            permuted_mm_row = seq * BATCH_ + batch
            mm_offsets = permuted_mm_row[:, None] * D_ + d[None, :]
            bs_offsets = m[:, None] * D_ + d[None, :]

            mm0 = tl.load(mm0_ptr + mm_offsets, mask=elem_mask, other=0.0).to(
                tl.float32
            )
            mm1 = tl.load(mm1_ptr + mm_offsets, mask=elem_mask, other=0.0).to(
                tl.float32
            )
            mm2 = tl.load(mm2_ptr + mm_offsets, mask=elem_mask, other=0.0).to(
                tl.float32
            )
            residual = tl.load(
                residual_ptr + bs_offsets, mask=elem_mask, other=0.0
            ).to(tl.float32)
            rhs = tl.load(rhs_ptr + bs_offsets, mask=elem_mask, other=0.0).to(
                tl.float32
            )
            scale = tl.load(scale_ptr + m, mask=m_mask, other=0.0).to(tl.float32)
            keep = tl.load(keep_ptr + bs_offsets, mask=elem_mask, other=0).to(
                tl.float32
            )

            mm_sum = (mm0 + mm1) + mm2
            x = residual + mm_sum
            weighted = x * gamma[None, :]
            row_sum = tl.sum(tl.where(elem_mask, weighted, 0.0), axis=1)
            row_dot = tl.sum(tl.where(elem_mask, weighted * rhs, 0.0), axis=1)
            grad = scale[:, None] * (
                weighted * D_ - row_sum[:, None] - rhs * row_dot[:, None]
            )
            masked_grad = grad * keep

            tl.store(grad_md_ptr + bs_offsets, masked_grad, mask=elem_mask)

            acc_x_rhs += tl.sum(tl.where(elem_mask, x * rhs, 0.0), axis=0)
            acc_x += tl.sum(tl.where(elem_mask, x, 0.0), axis=0)
            acc_grad += tl.sum(tl.where(elem_mask, masked_grad, 0.0), axis=0)

        partial_offsets = row_block * D_ + d
        tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=d_mask)
        tl.store(partial_x_ptr + partial_offsets, acc_x, mask=d_mask)
        tl.store(partial_grad_ptr + partial_offsets, acc_grad, mask=d_mask)


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_grad_ptr,
        out_x_rhs_ptr,
        out_x_ptr,
        out_grad_ptr,
        NUM_M_TILES: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d < D_
        tile_offsets = tl.arange(0, BLOCK_TILES)

        acc_x_rhs = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_grad = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for tile_start in range(0, NUM_M_TILES, BLOCK_TILES):
            tiles = tile_start + tile_offsets
            mask = (tiles[:, None] < NUM_M_TILES) & d_mask[None, :]
            offsets = tiles[:, None] * D_ + d[None, :]

            x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            grad = tl.load(partial_grad_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )

            acc_x_rhs += tl.sum(x_rhs, axis=0)
            acc_x += tl.sum(x, axis=0)
            acc_grad += tl.sum(grad, axis=0)

        tl.store(out_x_rhs_ptr + d, acc_x_rhs, mask=d_mask)
        tl.store(out_x_ptr + d, acc_x, mask=d_mask)
        tl.store(out_grad_ptr + d, acc_grad, mask=d_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_127,
        mm_129,
        mm_131,
        mul_279,
        arg8_1,
        arg109_1,
        arg297_1,
        arg108_1,
        *_shape_params,
    ) = inputs

    return (
        mm_127.contiguous(),
        mm_129.contiguous(),
        mm_131.contiguous(),
        mul_279.reshape(M, D).contiguous(),
        arg8_1.contiguous(),
        arg109_1.reshape(M, D).contiguous(),
        arg297_1.reshape(M).contiguous(),
        arg108_1.reshape(M, D).contiguous(),
    )


def oracle_triton_prepared(
    mm0_md_seq_major: torch.Tensor,
    mm1_md_seq_major: torch.Tensor,
    mm2_md_seq_major: torch.Tensor,
    residual_md: torch.Tensor,
    gamma_d: torch.Tensor,
    rhs_md: torch.Tensor,
    scale_m: torch.Tensor,
    keep_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm0_md_seq_major.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm0_md_seq_major.shape == (M, D)
    assert mm1_md_seq_major.shape == (M, D)
    assert mm2_md_seq_major.shape == (M, D)
    assert residual_md.shape == (M, D)
    assert gamma_d.shape == (D,)
    assert rhs_md.shape == (M, D)
    assert scale_m.shape == (M,)
    assert keep_md.shape == (M, D)
    assert mm0_md_seq_major.is_contiguous()
    assert mm1_md_seq_major.is_contiguous()
    assert mm2_md_seq_major.is_contiguous()
    assert residual_md.is_contiguous()
    assert gamma_d.is_contiguous()
    assert rhs_md.is_contiguous()
    assert scale_m.is_contiguous()
    assert keep_md.is_contiguous()

    device = mm0_md_seq_major.device
    num_m_tiles = triton.cdiv(M, ROW_SPLIT)
    partial_x_rhs = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    partial_grad = torch.empty((num_m_tiles, D), device=device, dtype=torch.float32)
    grad_md = torch.empty((M, D), device=device, dtype=torch.float32)

    _row_split_store_and_reduce_kernel[(num_m_tiles,)](
        mm0_md_seq_major,
        mm1_md_seq_major,
        mm2_md_seq_major,
        residual_md,
        gamma_d,
        rhs_md,
        scale_m,
        keep_md,
        partial_x_rhs,
        partial_x,
        partial_grad,
        grad_md,
        M_=M,
        D_=D,
        BATCH_=BATCH,
        SEQ_=SEQ,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_D_=BLOCK_D,
        num_warps=8,
    )

    out_x_rhs = torch.empty((D,), device=device, dtype=torch.float32)
    out_x = torch.empty((D,), device=device, dtype=torch.float32)
    out_grad = torch.empty((D,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_x_rhs,
        partial_x,
        partial_grad,
        out_x_rhs,
        out_x,
        out_grad,
        NUM_M_TILES=num_m_tiles,
        D_=D,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_D=FINAL_BLOCK_D,
        num_warps=8,
    )

    return out_x_rhs, out_x, grad_md.t(), out_grad


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
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
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

    with torch.no_grad():
        oracle_full(*inputs, impl=actual_impl)
        synchronize(device)
        if actual_impl == "triton":
            assert triton is not None
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
            oracle_us = benchmark(
                lambda: oracle_full(*inputs, impl=actual_impl),
                device,
                warmup,
                rep,
            )

    logical_read_bytes = (M * D * 6 + D + M) * 4 + M * D
    logical_write_bytes = (M * D + D * 3) * 4
    print(
        f"oracle_full cooperative split-k Longformer triple-mm layernorm backward: "
        f"{oracle_us:.3f} us impl={actual_impl} shape={SHAPE_LABEL} device={device}"
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
