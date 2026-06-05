"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Demucs gated cat-and-sum `Repro.forward` result by fusing the add/slice/sigmoid/sub/mul producer, reducing both cat halves as sibling channel outputs over the large batch/time domain with cooperative Triton partials, and finalizing directly into the contiguous `[128]` return tensor, whereas Inductor currently lowers the slice/sigmoid/sub/mul/cat/sum graph as a generic reduction over the logical concatenated producer without one shared split-K multi-output schedule for the two output slices; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output reduction template that shares the add/sigmoid producer across disjoint cat slices while partitioning the large reduction axis; the fix is COOPERATIVE_SPLIT_K: add a split-K multi-output cat-sum template that emits shared producer partials for both branch reductions and writes the concatenated result without materializing the cat."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, Any

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_b691b8dad90a"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)
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


def get_inputs() -> tuple[object, ...]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return make_inputs(device)


def get_repro_instance() -> torch.nn.Module:
    module = _load_repro_module()
    return module.Repro()


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


def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    return oracle_full(*inputs)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
