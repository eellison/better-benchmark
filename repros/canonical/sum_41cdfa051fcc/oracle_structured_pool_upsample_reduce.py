"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full NFNet average-pool-backward, GELU-derivative, and channel reduction directly from the `[128, 3072]` pooled gradient and `[128, 3072, 6, 6]` activation, whereas Inductor currently materializes the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient tensor and schedules the GELU derivative plus `[0, 2, 3]` sum as generic tensor work; Inductor cannot do this today because its scheduler/codegen does not recognize view/as_strided scatter followed by broadcasted average-pool backward as a structured scatter-reduce feeding a channel reduction; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter/expand lowering that computes the pointwise GELU derivative and accumulates the returned channel sum from source-space tiles."""
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
except ModuleNotFoundError:  # pragma: no cover - keeps --help usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_41cdfa051fcc"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_dm_nfnet_f0_train_001_70c9d47c"

N = 128
C = 3072
H = 6
W = 6
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
GAMMA = 1.7015043497085571
RSQRT2 = 0.7071067811865476
RSQRT2PI = 0.3989422804014327
BLOCK_SIZE = 8192

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]



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
    mm: torch.Tensor,
    arg413_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    pool_grad = mm[:, :, None, None] * INV_HW * GAMMA
    cdf = 0.5 * (torch.erf(arg413_1 * RSQRT2) + 1.0)
    pdf_term = arg413_1 * torch.exp(-0.5 * arg413_1 * arg413_1) * RSQRT2PI
    gelu_grad = cdf + pdf_term
    return (pool_grad * gelu_grad).sum(dim=(0, 2, 3))


if triton is not None:

    @triton.jit
    def _avgpool_gelu_reduce_kernel(
        mm_ptr,
        x_ptr,
        out_ptr,
        mm_stride_n: tl.constexpr,
        mm_stride_c: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        GAMMA_: tl.constexpr,
        RSQRT2_: tl.constexpr,
        RSQRT2PI_: tl.constexpr,
        BLOCK_SIZE_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE_)
        active = offsets < N_HW_
        n_idx = offsets // HW_
        hw_idx = offsets - n_idx * HW_
        h_idx = hw_idx // W_
        w_idx = hw_idx - h_idx * W_

        mm_offsets = n_idx * mm_stride_n + channel * mm_stride_c
        x_offsets = (
            n_idx * x_stride_n
            + channel * x_stride_c
            + h_idx * x_stride_h
            + w_idx * x_stride_w
        )

        mm_vals = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)
        x_vals = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)

        pool_grad = mm_vals * INV_HW_ * GAMMA_
        cdf = 0.5 * (tl.erf(x_vals * RSQRT2_) + 1.0)
        pdf_term = x_vals * tl.exp(-0.5 * x_vals * x_vals) * RSQRT2PI_
        gelu_grad = cdf + pdf_term
        values = tl.where(active, pool_grad * gelu_grad, 0.0)
        tl.store(out_ptr + channel, tl.sum(values, axis=0))


def oracle_triton(
    mm: torch.Tensor,
    arg413_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    out = torch.empty((C,), device=mm.device, dtype=torch.float32)
    _avgpool_gelu_reduce_kernel[(C,)](
        mm,
        arg413_1,
        out,
        mm_stride_n=mm.stride(0),
        mm_stride_c=mm.stride(1),
        x_stride_n=arg413_1.stride(0),
        x_stride_c=arg413_1.stride(1),
        x_stride_h=arg413_1.stride(2),
        x_stride_w=arg413_1.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        GAMMA_=GAMMA,
        RSQRT2_=RSQRT2,
        RSQRT2PI_=RSQRT2PI,
        BLOCK_SIZE_=BLOCK_SIZE,
        num_warps=8,
    )
    return out


def oracle_structured_pool_upsample_reduce(
    *inputs: object,
    impl: str = "auto",
) -> torch.Tensor:
    mm = inputs[0]
    if not isinstance(mm, torch.Tensor):
        raise TypeError("first input must be a tensor")
    if impl == "auto":
        impl = "triton" if mm.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
    raise ValueError(f"unknown impl: {impl}")


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> torch.Tensor:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    return model(*inputs)


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
        actual = oracle_structured_pool_upsample_reduce(*inputs, impl=impl)
        synchronize(device)

    max_abs, max_rel = _max_diff(actual, expected)
    value_ok = torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)
    stride_ok = actual.stride() == expected.stride()
    print(
        f"output: shape={list(actual.shape)} stride={actual.stride()} "
        f"expected_stride={expected.stride()} max_abs={max_abs:.6e} "
        f"max_rel={max_rel:.6e} allclose={value_ok} stride_match={stride_ok}"
    )
    print(f"Correctness: {'PASS' if value_ok and stride_ok else 'FAIL'}")
    return value_ok and stride_ok


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


def _compile_with_config(
    model: torch.nn.Module,
    inputs: tuple[object, ...],
    config: dict[str, object],
    device: torch.device,
):
    import torch._dynamo
    import torch._inductor.config as inductor_config

    torch._dynamo.reset()
    with inductor_config.patch(config):
        compiled = torch.compile(model)
        with torch.no_grad():
            for _ in range(3):
                compiled(*inputs)
            synchronize(device)
    return compiled


def run_bench(device: torch.device, impl: str, warmup: int, rep: int, no_compile: bool) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = impl
    if actual_impl == "auto":
        actual_impl = "triton" if device.type == "cuda" and triton is not None else "torch"

    logical_read_bytes = (N * C * 4) + (N * C * H * W * 4)
    logical_write_bytes = C * 4
    print(
        f"oracle shape: mm=f32[{N}, {C}], activation=f32[{N}, {C}, {H}, {W}], "
        f"out=f32[{C}]"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_structured_pool_upsample_reduce(*inputs, impl=actual_impl)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_structured_pool_upsample_reduce(*inputs, impl=actual_impl),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_structured_pool_upsample_reduce: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device}"
    )

    if no_compile:
        return

    if device.type != "cuda":
        print("torch.compile config timings skipped: CUDA is required")
        return

    module = _load_repro_module()
    print("torch.compile full Repro.forward timings:")
    for label, config in COMPILE_CONFIGS:
        try:
            compiled = _compile_with_config(module.Repro().to(device), inputs, config, device)
            compiled_us = benchmark(lambda: compiled(*inputs), device, warmup, rep)
            print(f"torch.compile {label}: {compiled_us:.3f} us")
        except Exception as exc:
            print(f"torch.compile {label}: FAILED ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile coordinate-descent and combo config baselines",
    )
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    device = torch.device(args.device)
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(
            device=device,
            impl=args.impl,
            warmup=args.warmup,
            rep=args.rep,
            no_compile=args.no_compile,
        )


if __name__ == "__main__":
    main()
