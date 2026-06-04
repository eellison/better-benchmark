"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete Longformer duplicate-index index_put(accumulate=True) into the live [8192, 768] layout, its scaled [768] hidden sum, and scaled [768, 8192] transposed side output directly with Triton initialization, scatter, and reduction kernels, whereas Inductor currently lowers the duplicate-index index_put as a generic atomic scatter and then schedules the as_strided/view/permute/div/sum/permute consumers as separate materialization, layout, and reduction kernels; Inductor cannot do this today because scheduler/codegen does not recognize a one-dimensional indexed scatter-add whose as_strided alias feeds both a reduction epilogue and a layout-changing full side-output store; the fix is SCATTER_REDUCE: add an indexed scatter-reduce lowering that targets the captured live layout, applies the scale, emits the transposed side store, and accumulates the hidden-dimension sum from the same scatter/load tile."""
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_1457ecad8f5c"
SHAPE_LABEL = "hf_allenailongformerbase_train_005_e98f6a22"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

SOURCE_BLOCKS = 288
SOURCE_SEQ = 512
HEAD_DIM = 64
SOURCE_NUMEL = SOURCE_BLOCKS * SOURCE_SEQ * HEAD_DIM
ROWS = 8192
HIDDEN = 768
FULL_NUMEL = ROWS * HIDDEN
SCALE = 0.125
BLOCK_COPY = 256
BLOCK_SCATTER = 256
BLOCK_ROWS = 128
BLOCK_HIDDEN = 16
NUM_ROW_TILES = math.ceil(ROWS / BLOCK_ROWS)



if triton is not None:

    @triton.jit
    def _copy_full_kernel(
        full_ptr,
        out_ptr,
        FULL_NUMEL_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < FULL_NUMEL_
        values = tl.load(full_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, values * SCALE_, mask=active)

    @triton.jit
    def _scatter_raw_kernel(
        bmm_ptr,
        index_ptr,
        out_ptr,
        SOURCE_NUMEL_: tl.constexpr,
        FULL_NUMEL_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        active = offsets < SOURCE_NUMEL_
        target = tl.load(index_ptr + offsets, mask=active, other=0).to(tl.int64)
        values = tl.load(bmm_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        valid = active & (target >= 0) & (target < FULL_NUMEL_)
        tl.atomic_add(out_ptr + target, values * SCALE_, sem="relaxed", mask=valid)

    @triton.jit
    def _partial_reduce_kernel(
        out_ptr,
        partial_ptr,
        ROWS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HIDDEN_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_r = tl.program_id(1)
        rows = pid_r * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hidden = pid_h * BLOCK_HIDDEN_ + tl.arange(0, BLOCK_HIDDEN_)
        mask = (rows[:, None] < ROWS_) & (hidden[None, :] < HIDDEN_)
        offsets = rows[:, None] * HIDDEN_ + hidden[None, :]
        values = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(
            partial_ptr + pid_r * HIDDEN_ + hidden,
            sums,
            mask=hidden < HIDDEN_,
        )

    @triton.jit
    def _finalize_reduce_kernel(
        partial_ptr,
        sum_ptr,
        HIDDEN_: tl.constexpr,
        NUM_ROW_TILES_: tl.constexpr,
        BLOCK_HIDDEN_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        hidden = pid_h * BLOCK_HIDDEN_ + tl.arange(0, BLOCK_HIDDEN_)
        tiles = tl.arange(0, BLOCK_TILES_)
        mask = (hidden[None, :] < HIDDEN_) & (tiles[:, None] < NUM_ROW_TILES_)
        offsets = tiles[:, None] * HIDDEN_ + hidden[None, :]
        values = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(sum_ptr + hidden, sums, mask=hidden < HIDDEN_)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        inputs = _load_repro_module().make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def oracle_longformer_index_put_sum_transpose(
    bmm_47: torch.Tensor,
    full_15: torch.Tensor,
    view_38: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5
    if triton is None:
        raise RuntimeError("triton is not available")
    if bmm_47.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    if bmm_47.shape != (SOURCE_BLOCKS, SOURCE_SEQ, HEAD_DIM):
        raise ValueError(f"unexpected bmm_47 shape: {tuple(bmm_47.shape)}")
    if full_15.numel() != FULL_NUMEL or view_38.numel() != SOURCE_NUMEL:
        raise ValueError("unexpected Longformer scatter buffer/index size")
    if not bmm_47.is_contiguous() or not full_15.is_contiguous() or not view_38.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layouts")

    out_base = torch.empty((ROWS, HIDDEN), device=bmm_47.device, dtype=torch.float32)
    _copy_full_kernel[(triton.cdiv(FULL_NUMEL, BLOCK_COPY),)](
        full_15,
        out_base,
        FULL_NUMEL_=FULL_NUMEL,
        SCALE_=SCALE,
        BLOCK=BLOCK_COPY,
        num_warps=4,
    )

    _scatter_raw_kernel[(triton.cdiv(SOURCE_NUMEL, BLOCK_SCATTER),)](
        bmm_47,
        view_38,
        out_base,
        SOURCE_NUMEL_=SOURCE_NUMEL,
        FULL_NUMEL_=FULL_NUMEL,
        SCALE_=SCALE,
        BLOCK=BLOCK_SCATTER,
        num_warps=4,
    )

    partial = torch.empty((NUM_ROW_TILES, HIDDEN), device=bmm_47.device, dtype=torch.float32)
    grid = (triton.cdiv(HIDDEN, BLOCK_HIDDEN), NUM_ROW_TILES)
    _partial_reduce_kernel[grid](
        out_base,
        partial,
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        num_warps=4,
    )

    out_sum = torch.empty((HIDDEN,), device=bmm_47.device, dtype=torch.float32)
    _finalize_reduce_kernel[(triton.cdiv(HIDDEN, BLOCK_HIDDEN),)](
        partial,
        out_sum,
        HIDDEN_=HIDDEN,
        NUM_ROW_TILES_=NUM_ROW_TILES,
        BLOCK_HIDDEN_=BLOCK_HIDDEN,
        BLOCK_TILES_=triton.next_power_of_2(NUM_ROW_TILES),
        num_warps=1,
    )
    return out_sum, out_base.t()


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = _load_repro_module().Repro().to(device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_longformer_index_put_sum_transpose(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} "
            f"max_rel={max_rel:.6e} shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def run_bench(device: torch.device, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    partial_numel = NUM_ROW_TILES * HIDDEN
    logical_read_bytes = (
        FULL_NUMEL * 4
        + SOURCE_NUMEL * 4
        + SOURCE_NUMEL * 8
        + FULL_NUMEL * 4
        + partial_numel * 4
    )
    logical_write_bytes = (
        FULL_NUMEL * 4
        + SOURCE_NUMEL * 4
        + partial_numel * 4
        + HIDDEN * 4
    )
    print(
        f"oracle shape: bmm=f32[{SOURCE_BLOCKS}, {SOURCE_SEQ}, {HEAD_DIM}], "
        f"full=f32[{FULL_NUMEL}], rows={ROWS}, hidden={HIDDEN} "
        f"shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_longformer_index_put_sum_transpose(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_longformer_index_put_sum_transpose(*inputs),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_longformer_index_put_sum_transpose: {oracle_us:.3f} us "
        f"impl=triton shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs to repro.py")
    parser.add_argument("--bench", action="store_true", help="time the Triton oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
