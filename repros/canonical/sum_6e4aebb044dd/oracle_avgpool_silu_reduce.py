"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full NFNet adaptive-average-pool backward plus SiLU-gradient channel-sum returned by repro.py directly from the `[128, 2304]` pooled gradient and `[128, 2304, 7, 7]` activation tensor, whereas Inductor currently lowers the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient reconstruction and the SiLU pointwise channel reduction as generic producer/consumer kernels; Inductor cannot do this today because its scheduler/codegen does not recognize zero-fill structured scatter/expand feeding only a pointwise channel reduction as a structured scatter-reduce without a materialized expanded tensor; the fix is SCATTER_REDUCE: add an average-pool-backward scatter/expand lowering that maps the pooled-gradient source directly into the channel reduction template and fuses the SiLU derivative epilogue."""
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
except ImportError:  # pragma: no cover - keeps syntax checks usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_6e4aebb044dd"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_nfnet_l0_train_001_369f714a"

N = 128
C = 2304
H = 7
W = 7
HW = H * W
INV_HW = 1.0 / HW

sys.path.insert(0, str(REPO_ROOT))


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
    return tuple(
        value.to(device=device) if isinstance(value, torch.Tensor) else value
        for value in module.make_inputs()
    )


def oracle_torch(
    mm: torch.Tensor,
    arg408_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    del _shape_param_0, _shape_param_1
    sigmoid = torch.reciprocal(torch.exp(torch.neg(arg408_1)) + 1.0)
    silu_grad = sigmoid * (arg408_1 * (1.0 - sigmoid) + 1.0)
    return (mm[:, :, None, None] * silu_grad).sum(dim=(0, 2, 3)) * INV_HW


if triton is not None:

    @triton.jit
    def _avgpool_silu_reduce_kernel(
        mm_ptr,
        x_ptr,
        out_ptr,
        stride_x_n: tl.constexpr,
        stride_x_c: tl.constexpr,
        stride_x_h: tl.constexpr,
        stride_x_w: tl.constexpr,
        N_: tl.constexpr,
        C_: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_n = tl.program_id(1)
        c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        rows = tl.arange(0, BLOCK_ROWS)
        n = pid_n * BLOCK_N + rows // 64
        spatial = rows % 64
        h = spatial // 7
        w = spatial - h * 7
        valid_spatial = spatial < 49
        mask = (n[:, None] < N_) & valid_spatial[:, None] & (c[None, :] < C_)

        x_offsets = (
            n[:, None] * stride_x_n
            + c[None, :] * stride_x_c
            + h[:, None] * stride_x_h
            + w[:, None] * stride_x_w
        )
        mm_offsets = n[:, None] * C_ + c[None, :]

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
        sigmoid = 1.0 / (tl.exp(-x) + 1.0)
        vals = mm * sigmoid * (x * (1.0 - sigmoid) + 1.0) * (1.0 / 49.0)
        tl.atomic_add(out_ptr + c, tl.sum(vals, axis=0), sem="relaxed", mask=c < C_)


def oracle_triton(
    mm: torch.Tensor,
    arg408_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    del _shape_param_0, _shape_param_1
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda" or arg408_1.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    out = torch.zeros((C,), device=mm.device, dtype=torch.float32)
    block_c = 16
    block_n = 8
    block_rows = block_n * 64
    grid = (triton.cdiv(C, block_c), triton.cdiv(N, block_n))
    _avgpool_silu_reduce_kernel[grid](
        mm,
        arg408_1,
        out,
        stride_x_n=arg408_1.stride(0),
        stride_x_c=arg408_1.stride(1),
        stride_x_h=arg408_1.stride(2),
        stride_x_w=arg408_1.stride(3),
        N_=N,
        C_=C,
        BLOCK_C=block_c,
        BLOCK_N=block_n,
        BLOCK_ROWS=block_rows,
        num_warps=8,
    )
    return out


def oracle_avgpool_silu_reduce(
    *inputs: object,
    impl: str = "auto",
) -> torch.Tensor:
    first = inputs[0]
    if not isinstance(first, torch.Tensor):
        raise TypeError("first input must be a tensor")
    if impl == "auto":
        impl = "triton" if first.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    out = model(*inputs)
    return out if isinstance(out, tuple) else (out,)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / expected.float().abs().clamp_min(1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        actual = oracle_avgpool_silu_reduce(*inputs, impl=impl)
        expected = reference_outputs(inputs, device)[0]
        synchronize(device)

    max_abs, max_rel = _max_diff(actual, expected)
    value_ok = torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)
    dtype_ok = actual.dtype == expected.dtype
    stride_ok = actual.stride() == expected.stride()
    ok = value_ok and dtype_ok and stride_ok
    print(
        f"output[0]: shape={list(actual.shape)} dtype={actual.dtype} "
        f"stride={actual.stride()} max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
        f"allclose={value_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
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
    oracle_fn = lambda: oracle_avgpool_silu_reduce(*inputs, impl=actual_impl)
    with torch.no_grad():
        oracle_fn()
        synchronize(device)
        oracle_us = benchmark(oracle_fn, device, warmup, rep)
    print(
        f"oracle_avgpool_silu_reduce: {oracle_us:.3f} us impl={actual_impl} "
        f"shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    default_impl = "triton" if triton is not None and torch.cuda.is_available() else "torch"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare full Repro.forward output against the oracle")
    parser.add_argument("--bench", action="store_true", help="benchmark the oracle")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default=default_impl)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
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
