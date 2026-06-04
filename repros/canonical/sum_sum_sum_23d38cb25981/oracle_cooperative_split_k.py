"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Swin MLP layernorm-backward/drop-path/roll/window-reverse return tuple for the `[401408, 128]` producer, including the two pre-drop `[128]` column reductions, the row-local 128-wide layernorm-backward reductions, residual add, per-batch drop-path scale, generated `(arange(56)+3)%56` roll indices, returned non-contiguous `[128, 401408]` transpose side output, and final `[128]` reduction over that side output, whereas Inductor currently schedules the sibling column sums, row reductions, stochastic-depth pointwise chain, generated-index roll, clone/permute/reshape layout work, transpose view, and downstream reduction as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output reduction template that coordinates row-local reductions, generated-index roll/window layout stores, and multiple dependent column accumulators in one producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible column reductions across row/window tiles, keep row-local layernorm summaries in registers, and fuse the drop-path roll/window transpose store with finalization of all column sums."""
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
except ModuleNotFoundError:  # pragma: no cover - keeps syntax/import checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_23d38cb25981"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_44226644"

BATCH = 128
H = 56
W = 56
TOKENS = H * W
WINDOW = 7
WINDOWS_H = H // WINDOW
WINDOWS_W = W // WINDOW
WINDOWS_PER_BATCH = WINDOWS_H * WINDOWS_W
WINDOW_AREA = WINDOW * WINDOW
C = 128
M = BATCH * TOKENS
ROLL_SHIFT = 3
KEEP_PROB = 0.9956521736457944

TILE_M = 16
TILE_C = 128
FINAL_BLOCK_C = 16
FINAL_BLOCK_TILES = 256
KERNEL_WARPS = 4

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
    if device.type != "cuda":
        raise RuntimeError("repro.py creates its roll index on CUDA; use --device cuda")
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm_189: torch.Tensor,
    primals_29: torch.Tensor,
    mul_14: torch.Tensor,
    div_119: torch.Tensor,
    view_1351: torch.Tensor,
    lt: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    )

    x = mm_189.reshape(BATCH, TOKENS, C)
    weighted = x * primals_29
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * mul_14).sum(dim=2, keepdim=True)
    grad_delta = div_119 * (weighted * C - row_sum - mul_14 * row_dot)

    out_x_rhs = (x * mul_14).sum(dim=(0, 1))
    out_x = x.sum(dim=(0, 1))

    dropped = (view_1351 + grad_delta).reshape(BATCH, H, W, C)
    dropped = dropped * (lt.to(torch.float32) / KEEP_PROB)
    roll_index = (
        torch.arange(H, device=mm_189.device, dtype=torch.int64) + ROLL_SHIFT
    ) % H
    indexed = torch.ops.aten.index.Tensor(dropped, [None, None, roll_index])
    indexed = torch.ops.aten.index.Tensor(indexed, [None, roll_index])
    windows = indexed.reshape(BATCH, WINDOWS_H, WINDOW, WINDOWS_W, WINDOW, C)
    flat = (
        windows.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(BATCH * WINDOWS_PER_BATCH, WINDOW_AREA, C)
        .reshape(M, C)
    )
    out_transposed = flat.permute(1, 0)
    out_final_sum = flat.sum(dim=0)
    return out_x_rhs, out_x, out_transposed, out_final_sum


if triton is not None:

    @triton.jit
    def _row_tile_store_and_reduce_kernel(
        x_ptr,
        weight_ptr,
        rhs_ptr,
        scale_ptr,
        residual_ptr,
        drop_mask_ptr,
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_final_ptr,
        out_transposed_ptr,
        M_: tl.constexpr,
        C_: tl.constexpr,
        H_: tl.constexpr,
        W_: tl.constexpr,
        WINDOW_: tl.constexpr,
        WINDOW_AREA_: tl.constexpr,
        WINDOWS_PER_BATCH_: tl.constexpr,
        KEEP_PROB_: tl.constexpr,
        ROLL_SHIFT_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        tile_m = tl.program_id(0)
        m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
        c = tl.arange(0, BLOCK_C)
        m_mask = m < M_
        c_mask = c < C_
        mask = m_mask[:, None] & c_mask[None, :]

        q = m // WINDOW_AREA_
        inner = m - q * WINDOW_AREA_
        inner_h = inner // WINDOW_
        inner_w = inner - inner_h * WINDOW_
        batch = q // WINDOWS_PER_BATCH_
        window_id = q - batch * WINDOWS_PER_BATCH_
        block_h = window_id // (W_ // WINDOW_)
        block_w = window_id - block_h * (W_ // WINDOW_)
        indexed_h = block_h * WINDOW_ + inner_h
        indexed_w = block_w * WINDOW_ + inner_w
        source_h = (indexed_h + ROLL_SHIFT_) % H_
        source_w = (indexed_w + ROLL_SHIFT_) % W_
        source_m = batch * (H_ * W_) + source_h * W_ + source_w

        source_offsets = source_m[:, None] * C_ + c[None, :]
        x = tl.load(x_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + source_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + source_m, mask=m_mask, other=0.0).to(tl.float32)
        keep = tl.load(drop_mask_ptr + batch, mask=m_mask, other=0).to(tl.float32)
        keep_scale = keep / KEEP_PROB_

        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        grad_delta = scale[:, None] * (
            weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
        )
        final = (residual + grad_delta) * keep_scale[:, None]

        transposed_offsets = m[:, None] * C_ + c[None, :]
        tl.store(out_transposed_ptr + transposed_offsets, final, mask=mask)

        partial_x_rhs = tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        partial_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        partial_final = tl.sum(tl.where(mask, final, 0.0), axis=0)
        partial_offsets = tile_m * C_ + c
        tl.store(partial_x_rhs_ptr + partial_offsets, partial_x_rhs, mask=c_mask)
        tl.store(partial_x_ptr + partial_offsets, partial_x, mask=c_mask)
        tl.store(partial_final_ptr + partial_offsets, partial_final, mask=c_mask)


    @triton.jit
    def _finalize_column_sums_kernel(
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_final_ptr,
        out_x_rhs_ptr,
        out_x_ptr,
        out_final_ptr,
        NUM_M_TILES: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c < C_
        tile_offsets = tl.arange(0, BLOCK_TILES)

        acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_final = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for tile_start in range(0, NUM_M_TILES, BLOCK_TILES):
            tiles = tile_start + tile_offsets
            mask = (tiles[:, None] < NUM_M_TILES) & c_mask[None, :]
            offsets = tiles[:, None] * C_ + c[None, :]
            x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            final = tl.load(partial_final_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            acc_x_rhs += tl.sum(x_rhs, axis=0)
            acc_x += tl.sum(x, axis=0)
            acc_final += tl.sum(final, axis=0)

        tl.store(out_x_rhs_ptr + c, acc_x_rhs, mask=c_mask)
        tl.store(out_x_ptr + c, acc_x, mask=c_mask)
        tl.store(out_final_ptr + c, acc_final, mask=c_mask)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_189,
        primals_29,
        mul_14,
        div_119,
        view_1351,
        lt,
        *_shape_params,
    ) = inputs

    return (
        mm_189.reshape(M, C).contiguous(),
        primals_29.contiguous(),
        mul_14.reshape(M, C).contiguous(),
        div_119.reshape(M).contiguous(),
        view_1351.reshape(M, C).contiguous(),
        lt.reshape(BATCH).contiguous(),
    )


def oracle_triton_prepared(
    x_mc: torch.Tensor,
    weight_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    scale_m: torch.Tensor,
    residual_mc: torch.Tensor,
    drop_mask_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if x_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_mc.shape == (M, C)
    assert weight_c.shape == (C,)
    assert rhs_mc.shape == (M, C)
    assert scale_m.shape == (M,)
    assert residual_mc.shape == (M, C)
    assert drop_mask_b.shape == (BATCH,)
    assert x_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert scale_m.is_contiguous()
    assert residual_mc.is_contiguous()
    assert drop_mask_b.is_contiguous()

    device = x_mc.device
    num_m_tiles = triton.cdiv(M, TILE_M)
    partial_x_rhs = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    partial_final = torch.empty((num_m_tiles, C), device=device, dtype=torch.float32)
    out_transposed = torch.empty_strided(
        (C, M),
        (1, C),
        device=device,
        dtype=torch.float32,
    )

    _row_tile_store_and_reduce_kernel[(num_m_tiles,)](
        x_mc,
        weight_c,
        rhs_mc,
        scale_m,
        residual_mc,
        drop_mask_b,
        partial_x_rhs,
        partial_x,
        partial_final,
        out_transposed,
        M_=M,
        C_=C,
        H_=H,
        W_=W,
        WINDOW_=WINDOW,
        WINDOW_AREA_=WINDOW_AREA,
        WINDOWS_PER_BATCH_=WINDOWS_PER_BATCH,
        KEEP_PROB_=KEEP_PROB,
        ROLL_SHIFT_=ROLL_SHIFT,
        BLOCK_M=TILE_M,
        BLOCK_C=TILE_C,
        num_warps=KERNEL_WARPS,
    )

    out_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_final = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_final,
        out_x_rhs,
        out_x,
        out_final,
        NUM_M_TILES=num_m_tiles,
        C_=C,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=KERNEL_WARPS,
    )

    return out_x_rhs, out_x, out_transposed, out_final


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
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs, max_rel = math.inf, math.inf
            value_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
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

    logical_read_bytes = (M * C * 4) * 3 + C * 4 + (M * 4) + BATCH
    logical_write_bytes = (M * C * 4) + (C * 4) * 3

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
        f"oracle_full cooperative split-k Swin LN/roll/window tuple: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device}"
    )
    print(f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=2e-1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
