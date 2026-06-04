"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle writes the returned zero-padded `slice_scatter` tensor and accumulates the sibling masked channel sum from the same source-tile pass, whereas Inductor currently treats the `full -> slice_scatter` side output and the `where(...).sum([0, 2])` reduction as separate generic producers and consumers over `getitem_27`; Inductor cannot do this today because its scheduler/codegen does not model zero-fill `slice_scatter` with a required materialized side output plus a source-space masked reduction epilogue as one structured scatter-reduce template; the fix is SCATTER_REDUCE: add a structured `slice_scatter` lowering that emits the padded side-output stores while accumulating sibling reductions directly from the scattered source tile."""
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
except ImportError:  # pragma: no cover - keeps CPU-only syntax checks working.
    triton = None
    tl = None


REPRO_ID = "sum_837a32994d31"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_demucs_train_003_8b355ac6"

N = 64
C = 1024
W = 364
PAD = 4
PADDED_W = 372
BLOCK_SIZE = 1024
N_TILES = math.ceil((N * PADDED_W) / BLOCK_SIZE)
FINAL_BLOCK = 32



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if triton is not None:

    @triton.jit
    def _slice_scatter_reduce_kernel(
        src_ptr,
        mask_ptr,
        full_ptr,
        padded_out_ptr,
        partial_ptr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        PAD_: tl.constexpr,
        PADDED_W_: tl.constexpr,
        BLOCK_SIZE_: tl.constexpr,
        N_TILES_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_SIZE_ + tl.arange(0, BLOCK_SIZE_)
        valid = offsets < (N_ * PADDED_W_)

        n_idx = offsets // PADDED_W_
        padded_w_idx = offsets - n_idx * PADDED_W_
        in_bounds = (padded_w_idx >= PAD_) & (padded_w_idx < (PAD_ + W_)) & valid
        src_w_idx = padded_w_idx - PAD_

        src_offsets = n_idx * (C_ * W_) + channel * W_ + src_w_idx
        safe_src_offsets = tl.where(in_bounds, src_offsets, 0)
        padded_offsets = n_idx * (C_ * PADDED_W_) + channel * PADDED_W_ + padded_w_idx

        src_vals = tl.load(src_ptr + safe_src_offsets, mask=in_bounds, other=0.0).to(tl.float32)
        mask_vals = tl.load(mask_ptr + safe_src_offsets, mask=in_bounds, other=0)
        full_val = tl.load(full_ptr).to(tl.float32)

        padded_vals = tl.where(in_bounds, src_vals, 0.0)
        tl.store(padded_out_ptr + padded_offsets, padded_vals, mask=valid)

        reduce_vals = tl.where(mask_vals, full_val, src_vals)
        reduce_vals = tl.where(in_bounds, reduce_vals, 0.0)
        partial = tl.sum(reduce_vals, axis=0)
        tl.store(partial_ptr + channel * N_TILES_ + tile, partial)


    @triton.jit
    def _finalize_sum_kernel(
        partial_ptr,
        out_ptr,
        N_TILES_: tl.constexpr,
        FINAL_BLOCK_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, FINAL_BLOCK_)
        mask = offsets < N_TILES_
        vals = tl.load(partial_ptr + channel * N_TILES_ + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + channel, tl.sum(vals, axis=0))


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
        module = _load_repro_module()
        inputs = module.make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def oracle_torch(
    getitem_27: torch.Tensor,
    arg34_1: torch.Tensor,
    full_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.empty((N, C, PADDED_W), device=getitem_27.device, dtype=getitem_27.dtype)
    padded[:, :, :PAD].zero_()
    padded[:, :, PAD : PAD + W] = getitem_27
    padded[:, :, PAD + W :].zero_()
    reduced = torch.where(arg34_1, full_1, getitem_27).sum(dim=(0, 2))
    return padded, reduced


def oracle_triton(
    getitem_27: torch.Tensor,
    arg34_1: torch.Tensor,
    full_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem_27.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    padded = torch.empty((N, C, PADDED_W), device=getitem_27.device, dtype=getitem_27.dtype)
    partial = torch.empty((C, N_TILES), device=getitem_27.device, dtype=torch.float32)
    reduced = torch.empty((C,), device=getitem_27.device, dtype=torch.float32)

    _slice_scatter_reduce_kernel[(C, N_TILES)](
        getitem_27,
        arg34_1,
        full_1,
        padded,
        partial,
        N_=N,
        C_=C,
        W_=W,
        PAD_=PAD,
        PADDED_W_=PADDED_W,
        BLOCK_SIZE_=BLOCK_SIZE,
        N_TILES_=N_TILES,
        num_warps=8,
    )
    _finalize_sum_kernel[(C,)](
        partial,
        reduced,
        N_TILES_=N_TILES,
        FINAL_BLOCK_=FINAL_BLOCK,
        num_warps=1,
    )
    return padded, reduced


def oracle_slice_scatter_reduce(
    getitem_27: torch.Tensor,
    arg34_1: torch.Tensor,
    full_1: torch.Tensor,
    *,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
    if impl == "auto":
        impl = "triton" if getitem_27.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(getitem_27, arg34_1, full_1)
    if impl == "torch":
        return oracle_torch(getitem_27, arg34_1, full_1)
    raise ValueError(f"unknown impl: {impl}")


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, torch.Tensor):
        return (value,)
    raise TypeError(f"expected tensor or tuple of tensors, got {type(value)!r}")


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        actual = _as_tuple(oracle_slice_scatter_reduce(*inputs, impl=impl))
        expected = _as_tuple(reference_outputs(inputs, device))
        synchronize(device)

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)}")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"allclose={value_ok} stride_match={stride_ok}"
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
    oracle_fn = lambda: oracle_slice_scatter_reduce(*inputs, impl=impl)
    with torch.no_grad():
        oracle_fn()
        synchronize(device)
        oracle_us = benchmark(oracle_fn, device, warmup, rep)
    print(
        f"oracle_us={oracle_us:.3f} impl={impl} shape={SHAPE_LABEL} "
        f"device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    default_impl = "triton" if triton is not None and torch.cuda.is_available() else "torch"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare the full repro.py return tuple against the oracle")
    parser.add_argument("--bench", action="store_true", help="benchmark the selected oracle implementation")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default=default_impl)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-2)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
