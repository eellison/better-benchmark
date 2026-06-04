"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer index_put/sum/transpose Repro.forward tuple by initializing the as_strided output storage from full_15, applying the duplicate-index bmm scatter-adds in Triton, reducing the finalized [2048,768] storage to the [768] side output, and returning the required non-contiguous [768,2048] transpose view, whereas Inductor currently lowers the accumulate=True index_put as a generic atomic scatter and schedules the following as_strided/view/permute/div/sum/transpose consumers as separate layout and reduction kernels; Inductor cannot do this today because its scheduler/codegen does not recognize a one-dimensional indexed scatter feeding both a reduction epilogue and a required transposed aliasing side-output through as_strided layout views; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that fuses scale, duplicate-index scatter accumulation, hidden-dimension reduction, and transposed output-layout emission for this Longformer pattern."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch._inductor.inductor_prims  # noqa: F401
import triton
import triton.language as tl


REPRO_ID = "sum_79c6d0673ca2"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

sys.path.insert(0, str(REPO_ROOT))

UPDATES = 72 * 512 * 64
STORAGE = 1_572_864
ROWS = 2048
HIDDEN = 768
SCALE = 0.125
INIT_BLOCK = 1024
SCATTER_BLOCK = 512
REDUCE_BLOCK = 2048


@triton.jit
def _init_scaled_storage_kernel(
    full_ptr,
    out_storage_ptr,
    STORAGE_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < STORAGE_
    full = tl.load(full_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_storage_ptr + offsets, full * SCALE_, mask=mask)


@triton.jit
def _scatter_scaled_updates_kernel(
    bmm_ptr,
    index_ptr,
    out_storage_ptr,
    UPDATES_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < UPDATES_
    indices = tl.load(index_ptr + offsets, mask=mask, other=0)
    values = tl.load(bmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32) * SCALE_
    tl.atomic_add(out_storage_ptr + indices, values, sem="relaxed", mask=mask)


@triton.jit
def _reduce_hidden_kernel(
    out_storage_ptr,
    reduced_ptr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    hidden = tl.program_id(0)
    rows = tl.arange(0, BLOCK_ROWS)
    mask = rows < ROWS_
    values = tl.load(
        out_storage_ptr + rows * HIDDEN_ + hidden,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(reduced_ptr + hidden, tl.sum(values, axis=0))


def _load_repro_module() -> Any:
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def make_inputs() -> tuple[Any, ...]:
    module = _load_repro_module()
    return tuple(
        x.cuda() if isinstance(x, torch.Tensor) and not x.is_cuda else x
        for x in module.make_inputs()
    )


def reference_outputs(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _check_shape_params(
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
    _shape_param_4: object,
    _shape_param_5: object,
) -> None:
    assert list(_shape_param_0) == [24, 3, 512, 64, 1]
    assert list(_shape_param_1) == [24, 1024, 64]
    assert list(_shape_param_2) == [2, 12, 1024, 64]
    assert list(_shape_param_3) == [1024, 2, 768]
    assert list(_shape_param_4) == [768]
    assert list(_shape_param_5) == [2048, 768]


def _check_inputs(
    bmm_47: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
) -> None:
    assert bmm_47.is_cuda and full_15.is_cuda and view_38.is_cuda
    assert bmm_47.shape == (72, 512, 64) and bmm_47.dtype == torch.float32
    assert full_15.shape == (STORAGE,) and full_15.dtype == torch.float32
    assert view_38.shape == (UPDATES,) and view_38.dtype == torch.int64
    assert bmm_47.is_contiguous()
    assert full_15.is_contiguous()
    assert view_38.is_contiguous()


def _launch_index_put_sum_transpose(
    bmm_47: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
    reduced: torch.Tensor,
    transposed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert reduced.shape == (HIDDEN,) and reduced.dtype == torch.float32
    assert transposed.shape == (HIDDEN, ROWS) and transposed.dtype == torch.float32
    assert transposed.stride() == (1, HIDDEN)

    _init_scaled_storage_kernel[(triton.cdiv(STORAGE, INIT_BLOCK),)](
        full_15,
        transposed,
        STORAGE_=STORAGE,
        SCALE_=SCALE,
        BLOCK=INIT_BLOCK,
        num_warps=4,
    )
    _scatter_scaled_updates_kernel[(triton.cdiv(UPDATES, SCATTER_BLOCK),)](
        bmm_47,
        view_38,
        transposed,
        UPDATES_=UPDATES,
        SCALE_=SCALE,
        BLOCK=SCATTER_BLOCK,
        num_warps=4,
    )
    _reduce_hidden_kernel[(HIDDEN,)](
        transposed,
        reduced,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_ROWS=REDUCE_BLOCK,
        num_warps=8,
    )
    return reduced, transposed


def oracle_index_put_sum_transpose(
    bmm_47: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
    _shape_param_4: object,
    _shape_param_5: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_shape_params(
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
    )
    _check_inputs(bmm_47, full_15, view_38)

    transposed = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=bmm_47.device,
        dtype=torch.float32,
    )
    reduced = torch.empty((HIDDEN,), device=bmm_47.device, dtype=torch.float32)

    return _launch_index_put_sum_transpose(
        bmm_47,
        full_15,
        view_38,
        reduced,
        transposed,
    )


def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return oracle_index_put_sum_transpose(*inputs)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    finite = diff[torch.isfinite(diff)]
    max_abs = finite.max().item() if finite.numel() else float("nan")
    rel = diff / expected.float().abs().clamp_min(1e-8)
    finite_rel = rel[torch.isfinite(rel)]
    max_rel = finite_rel.max().item() if finite_rel.numel() else float("nan")
    return max_abs, max_rel


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = _as_tuple(oracle_index_put_sum_transpose(*inputs))
        torch.cuda.synchronize()

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        max_abs, max_rel = _max_diff(got, ref) if shape_ok else (float("inf"), float("inf"))
        value_ok = (
            torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol, equal_nan=True)
            if shape_ok
            else False
        )
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"expected_shape={list(ref.shape)} expected_dtype={ref.dtype} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={value_ok} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return bool(ok)


def _time_cuda_us(fn: Callable[[], object], warmup: int, rep: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    times.sort()
    return times[0], times[len(times) // 2]


def run_bench(warmup: int, rep: int) -> tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    bmm_47, full_15, view_38, *_ = inputs
    transposed = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=bmm_47.device,
        dtype=torch.float32,
    )
    reduced = torch.empty((HIDDEN,), device=bmm_47.device, dtype=torch.float32)
    with torch.no_grad():
        _launch_index_put_sum_transpose(bmm_47, full_15, view_38, reduced, transposed)
        torch.cuda.synchronize()
        best_us, median_us = _time_cuda_us(
            lambda: _launch_index_put_sum_transpose(
                bmm_47,
                full_15,
                view_38,
                reduced,
                transposed,
            ),
            warmup=warmup,
            rep=rep,
        )

    logical_bytes = (
        STORAGE * 4
        + STORAGE * 4
        + UPDATES * (4 + 8 + 4)
        + STORAGE * 4
        + HIDDEN * 4
    )
    print(
        f"oracle_index_put_sum_transpose full-scope Triton: best={best_us:.3f} us "
        f"median={median_us:.3f} us warmup={warmup} rep={rep}"
    )
    print(
        f"shape: updates=f32[{UPDATES}], storage=f32[{STORAGE}], "
        f"outputs=([768], [768,2048]), logical_traffic={logical_bytes / 1e6:.1f} MB"
    )
    return best_us, median_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare oracle against Repro.forward")
    parser.add_argument("--bench", action="store_true", help="run oracle timing benchmark")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
