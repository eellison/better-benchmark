"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Demucs gated two-branch cat-and-sum return by loading `arg13_1`'s paired channel halves and `getitem_30` once per `(batch,time,channel)` tile, accumulating both output halves in one cooperative split-K Triton producer and finalizing the contiguous `[4096]` result, whereas Inductor currently lowers the slice/sigmoid/sub/mul/cat/sum graph as a generic fused reduction over the logical cat domain; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template for sibling reductions that share a sigmoid producer but write disjoint slices of one returned vector; the fix is COOPERATIVE_SPLIT_K: add a split-K reduction template that accumulates multiple branch outputs from one producer tile, finalizes partials, and writes concatenated output layouts directly."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps syntax-only checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_031c8d6c51f7"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 64
C_HALF = 2048
C_TOTAL = 4096
TIME = 90
K_TOTAL = BATCH * TIME

BLOCK_C = 16
BLOCK_K = 128



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(device: torch.device) -> tuple[object, ...]:
    torch.manual_seed(0)
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


if triton is not None:

    @triton.jit
    def _demucs_partial_kernel(
        arg13_ptr,
        getitem_ptr,
        partial_ptr,
        C_HALF_: tl.constexpr,
        C_TOTAL_: tl.constexpr,
        TIME_: tl.constexpr,
        K_TOTAL_: tl.constexpr,
        NUM_K_TILES: tl.constexpr,
        BLOCK_K_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        tile_c = tl.program_id(0)
        tile_k = tl.program_id(1)

        k = tile_k * BLOCK_K_ + tl.arange(0, BLOCK_K_)
        c = tile_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        k_mask = k < K_TOTAL_
        c_mask = c < C_HALF_

        batch = k // TIME_
        time = k - batch * TIME_

        arg13_base = batch[:, None] * (C_TOTAL_ * TIME_) + c[None, :] * TIME_ + time[:, None]
        getitem_base = batch[:, None] * (C_HALF_ * TIME_) + c[None, :] * TIME_ + time[:, None]
        mask = k_mask[:, None] & c_mask[None, :]

        first = tl.load(arg13_ptr + arg13_base, mask=mask, other=0.0).to(tl.float32)
        second = tl.load(arg13_ptr + arg13_base + C_HALF_ * TIME_, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(getitem_ptr + getitem_base, mask=mask, other=0.0).to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-second))
        out_first_half = sig * grad
        out_second_half = sig * (1.0 - sig) * first * grad

        sum_first_half = tl.sum(out_first_half, axis=0)
        sum_second_half = tl.sum(out_second_half, axis=0)

        partial_offsets = tile_k * C_HALF_ + c
        tl.store(partial_ptr + partial_offsets, sum_first_half, mask=c_mask)
        tl.store(
            partial_ptr + NUM_K_TILES * C_HALF_ + partial_offsets,
            sum_second_half,
            mask=c_mask,
        )


    @triton.jit
    def _demucs_finalize_kernel(
        partial_ptr,
        out_ptr,
        C_HALF_: tl.constexpr,
        NUM_K_TILES: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        tile_c = tl.program_id(0)
        tile = tl.arange(0, BLOCK_TILES)
        c = tile_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        c_mask = c < C_HALF_
        mask = (tile[:, None] < NUM_K_TILES) & c_mask[None, :]
        offsets = tile[:, None] * C_HALF_ + c[None, :]

        partial_first = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
        partial_second = tl.load(
            partial_ptr + NUM_K_TILES * C_HALF_ + offsets,
            mask=mask,
            other=0.0,
        )

        sum_first = tl.sum(partial_first, axis=0)
        sum_second = tl.sum(partial_second, axis=0)
        tl.store(out_ptr + c, sum_first, mask=c_mask)
        tl.store(out_ptr + C_HALF_ + c, sum_second, mask=c_mask)


def oracle_full(arg13_1: torch.Tensor, getitem_30: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if arg13_1.device.type != "cuda" or getitem_30.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA tensors")
    if arg13_1.shape != (BATCH, C_TOTAL, TIME):
        raise ValueError(f"expected arg13_1 shape {(BATCH, C_TOTAL, TIME)}, got {tuple(arg13_1.shape)}")
    if getitem_30.shape != (BATCH, C_HALF, TIME):
        raise ValueError(f"expected getitem_30 shape {(BATCH, C_HALF, TIME)}, got {tuple(getitem_30.shape)}")
    if not arg13_1.is_contiguous() or not getitem_30.is_contiguous():
        raise ValueError("this oracle expects the captured contiguous input layouts")

    device = arg13_1.device
    num_k_tiles = triton.cdiv(K_TOTAL, BLOCK_K)
    block_tiles = triton.next_power_of_2(num_k_tiles)
    partial = torch.empty((2, num_k_tiles, C_HALF), device=device, dtype=torch.float32)
    out = torch.empty((C_TOTAL,), device=device, dtype=torch.float32)

    grid = (triton.cdiv(C_HALF, BLOCK_C), num_k_tiles)
    _demucs_partial_kernel[grid](
        arg13_1,
        getitem_30,
        partial,
        C_HALF_=C_HALF,
        C_TOTAL_=C_TOTAL,
        TIME_=TIME,
        K_TOTAL_=K_TOTAL,
        NUM_K_TILES=num_k_tiles,
        BLOCK_K_=BLOCK_K,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )
    _demucs_finalize_kernel[(triton.cdiv(C_HALF, BLOCK_C),)](
        partial,
        out,
        C_HALF_=C_HALF,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )
    return out


def _compare_outputs(
    got_outputs: tuple[torch.Tensor, ...],
    ref_outputs: tuple[torch.Tensor, ...],
    rtol: float,
    atol: float,
) -> bool:
    if len(got_outputs) != len(ref_outputs):
        print(f"output_count: got={len(got_outputs)} expected={len(ref_outputs)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(got_outputs, ref_outputs)):
        shape_ok = tuple(got.shape) == tuple(ref.shape)
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got, ref, rtol=rtol, atol=atol)
        diff = (got - ref).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        denom = ref.abs().clamp_min(1e-12)
        max_rel = (diff / denom).max().item() if diff.numel() else 0.0
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"expected_shape={list(ref.shape)} expected_dtype={ref.dtype} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} allclose={value_ok} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    inputs = make_inputs(device)
    with torch.no_grad():
        ref = reference_outputs(inputs, device)
        got = _as_tuple(oracle_full(*inputs))
    synchronize(device)
    return _compare_outputs(got, ref, rtol=rtol, atol=atol)


def _time_cuda_us(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> tuple[float, float]:
    if device.type != "cuda":
        raise RuntimeError("benchmarking requires a CUDA device")
    for _ in range(warmup):
        fn()
    synchronize(device)

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
    median = times[len(times) // 2]
    return times[0], median


def run_bench(device: torch.device, warmup: int, rep: int) -> None:
    inputs = make_inputs(device)
    module = _load_repro_module()
    model = module.Repro().to(device)
    compiled = torch.compile(model)

    with torch.no_grad():
        oracle_full(*inputs)
        compiled(*inputs)
    synchronize(device)

    with torch.no_grad():
        oracle_best, oracle_median = _time_cuda_us(lambda: oracle_full(*inputs), device, warmup, rep)
        compiled_best, compiled_median = _time_cuda_us(lambda: compiled(*inputs), device, warmup, rep)

    num_k_tiles = math.ceil(K_TOTAL / BLOCK_K)
    input_elements = BATCH * C_HALF * TIME
    partial_elements = 2 * num_k_tiles * C_HALF
    logical_bytes = (3 * input_elements + 2 * partial_elements + C_TOTAL) * 4
    print(
        f"oracle cooperative split-k demucs gate+sum: best={oracle_best:.3f} us "
        f"median={oracle_median:.3f} us warmup={warmup} rep={rep}"
    )
    print(
        f"torch.compile Repro.forward: best={compiled_best:.3f} us "
        f"median={compiled_median:.3f} us warmup={warmup} rep={rep}"
    )
    print(
        f"shape=[{BATCH}, {C_TOTAL}, {TIME}] + [{BATCH}, {C_HALF}, {TIME}] "
        f"partials={num_k_tiles}x2x{C_HALF} logical_traffic={logical_bytes / 1e6:.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare oracle against Repro.forward")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=3e-4)
    parser.add_argument("--atol", type=float, default=3e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")
    if args.warmup < 0 or args.rep <= 0:
        parser.error("--warmup must be non-negative and --rep must be positive")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
