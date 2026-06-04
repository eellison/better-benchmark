"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Demucs gated cat-and-sum `Repro.forward` result by fusing the add/slice/sigmoid/sub/mul producer, reducing both cat halves as sibling channel outputs over the large batch/time domain with cooperative Triton partials, and finalizing directly into the contiguous `[128]` return tensor, whereas Inductor currently lowers the slice/sigmoid/sub/mul/cat/sum graph as a generic reduction over the logical concatenated producer without one shared split-K multi-output schedule for the two output slices; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output reduction template that shares the add/sigmoid producer across disjoint cat slices while partitioning the large reduction axis; the fix is COOPERATIVE_SPLIT_K: add a split-K multi-output cat-sum template that emits shared producer partials for both branch reductions and writes the concatenated result without materializing the cat."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_b691b8dad90a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 64
C_HALF = 64
C_TOTAL = 128
TIME = 95696
OUT_SIZE = 128
BLOCK_T = 2048
PARTIAL_COUNT = BATCH



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
    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in inputs
    )


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


if triton is not None:

    @triton.jit
    def _demucs_gate_batch_partial_kernel(
        arg31_ptr,
        getitem27_ptr,
        arg14_ptr,
        partial_ptr,
        BATCH_: tl.constexpr,
        C_HALF_: tl.constexpr,
        C_TOTAL_: tl.constexpr,
        TIME_: tl.constexpr,
        BLOCK_T_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        batch = tl.program_id(1)
        offsets = tl.arange(0, BLOCK_T_)

        first_acc = tl.full((), 0.0, tl.float32)
        second_acc = tl.full((), 0.0, tl.float32)

        for start in range(0, TIME_, BLOCK_T_):
            s = start + offsets
            mask = s < TIME_

            base64 = batch * C_HALF_ * TIME_ + channel * TIME_ + s
            base128 = batch * C_TOTAL_ * TIME_ + channel * TIME_ + s

            add_value = (
                tl.load(arg31_ptr + base64, mask=mask, other=0.0).to(tl.float32)
                + tl.load(getitem27_ptr + base64, mask=mask, other=0.0).to(tl.float32)
            )
            slice0 = tl.load(arg14_ptr + base128, mask=mask, other=0.0).to(tl.float32)
            slice1 = tl.load(
                arg14_ptr + base128 + C_HALF_ * TIME_, mask=mask, other=0.0
            ).to(tl.float32)

            sig = tl.sigmoid(slice1)
            first_acc += tl.sum(sig * add_value, axis=0)
            second_acc += tl.sum(sig * (1.0 - sig) * slice0 * add_value, axis=0)

        partial_index = channel * BATCH_ + batch
        tl.store(partial_ptr + partial_index, first_acc)
        tl.store(partial_ptr + (C_HALF_ + channel) * BATCH_ + batch, second_acc)


    @triton.jit
    def _demucs_gate_finalize_kernel(
        partial_ptr,
        out_ptr,
        PARTIAL_COUNT_: tl.constexpr,
        C_HALF_: tl.constexpr,
        BLOCK_K_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        k = tl.arange(0, BLOCK_K_)
        mask = k < PARTIAL_COUNT_

        base = channel * PARTIAL_COUNT_ + k
        first = tl.load(partial_ptr + base, mask=mask, other=0.0)
        second = tl.load(
            partial_ptr + (C_HALF_ + channel) * PARTIAL_COUNT_ + k,
            mask=mask,
            other=0.0,
        )

        tl.store(out_ptr + channel, tl.sum(first, axis=0))
        tl.store(out_ptr + C_HALF_ + channel, tl.sum(second, axis=0))


def oracle_full(
    arg31_1: torch.Tensor,
    getitem_27: torch.Tensor,
    arg14_1: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if arg31_1.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA tensors")
    if arg31_1.shape != (BATCH, C_HALF, TIME):
        raise ValueError(f"expected arg31_1 shape {(BATCH, C_HALF, TIME)}, got {tuple(arg31_1.shape)}")
    if getitem_27.shape != (BATCH, C_HALF, TIME):
        raise ValueError(
            f"expected getitem_27 shape {(BATCH, C_HALF, TIME)}, got {tuple(getitem_27.shape)}"
        )
    if arg14_1.shape != (BATCH, C_TOTAL, TIME):
        raise ValueError(f"expected arg14_1 shape {(BATCH, C_TOTAL, TIME)}, got {tuple(arg14_1.shape)}")
    if arg31_1.dtype != torch.float32 or getitem_27.dtype != torch.float32 or arg14_1.dtype != torch.float32:
        raise ValueError("this oracle expects the captured float32 inputs")
    if not arg31_1.is_contiguous() or not getitem_27.is_contiguous() or not arg14_1.is_contiguous():
        raise ValueError("this oracle expects the captured contiguous input layouts")

    device = arg31_1.device
    partial = torch.empty((OUT_SIZE, PARTIAL_COUNT), device=device, dtype=torch.float32)
    out = torch.empty((OUT_SIZE,), device=device, dtype=torch.float32)

    _demucs_gate_batch_partial_kernel[(C_HALF, BATCH)](
        arg31_1,
        getitem_27,
        arg14_1,
        partial,
        BATCH_=BATCH,
        C_HALF_=C_HALF,
        C_TOTAL_=C_TOTAL,
        TIME_=TIME,
        BLOCK_T_=BLOCK_T,
        num_warps=16,
    )
    _demucs_gate_finalize_kernel[(C_HALF,)](
        partial,
        out,
        PARTIAL_COUNT_=PARTIAL_COUNT,
        C_HALF_=C_HALF,
        BLOCK_K_=triton.next_power_of_2(PARTIAL_COUNT),
        num_warps=8,
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
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        diff = (got.float() - ref.float()).abs()
        max_abs = diff.max().item() if diff.numel() else 0.0
        mean_abs = diff.mean().item() if diff.numel() else 0.0
        denom = ref.float().abs().clamp_min(1e-12)
        max_rel = (diff / denom).max().item() if diff.numel() else 0.0
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} stride={got.stride()} "
            f"expected_shape={list(ref.shape)} expected_dtype={ref.dtype} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this Triton oracle check")

    inputs = make_inputs(device)
    with torch.no_grad():
        ref = reference_outputs(inputs, device)
        got = _as_tuple(oracle_full(*inputs))
    synchronize(device)
    return _compare_outputs(got, ref, rtol=rtol, atol=atol)


def _time_cuda_us(
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    rep: int,
) -> tuple[float, float]:
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
    return times[0], times[len(times) // 2]


def run_bench(device: torch.device, warmup: int, rep: int) -> tuple[float, float]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this Triton oracle benchmark")

    inputs = make_inputs(device)
    with torch.no_grad():
        oracle_full(*inputs)
    synchronize(device)

    with torch.no_grad():
        best_us, median_us = _time_cuda_us(lambda: oracle_full(*inputs), device, warmup, rep)

    input_elements = BATCH * C_HALF * TIME
    partial_elements = OUT_SIZE * PARTIAL_COUNT
    logical_bytes = (4 * input_elements + 2 * partial_elements + OUT_SIZE) * 4
    print(
        f"oracle_multi_output_reduction full-scope: best={best_us:.3f} us "
        f"median={median_us:.3f} us warmup={warmup} rep={rep}"
    )
    print(
        f"shape=[{BATCH}, {C_HALF}, {TIME}] + [{BATCH}, {C_TOTAL}, {TIME}] "
        f"block_t={BLOCK_T} partials={PARTIAL_COUNT} logical_traffic={logical_bytes / 1e9:.3f} GB"
    )
    return best_us, median_us


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
