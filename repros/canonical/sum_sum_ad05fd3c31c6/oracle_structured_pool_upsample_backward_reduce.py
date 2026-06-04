"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle revalidates the complete GhostNet adaptive-average-pool backward, ReLU-gated batch-norm-backward return tuple from the original `[512, 960]` pooled gradient and `[512, 960, 7, 7]` activation, emitting both the full input-gradient tensor and gamma-gradient vector without materializing the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient tensor; the measured hand oracle does not beat the historical compiled result because the required full side-output store plus channel reductions force either a strided per-channel fused schedule or a materialized generic producer, and Inductor already generates a competitive fused reduction/materialization schedule for this exact shape; the fix is SCATTER_REDUCE: only a structured average-pool-backward scatter-reduce lowering that maps each pooled-gradient source directly into the ReLU-gated BN reduction template while emitting the dependent full output tensor from the same schedule would establish a true lower floor."""
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


REPRO_ID = "sum_sum_ad05fd3c31c6"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_ghostnet_100_train_001_3f38dae0"

N = 512
C = 960
H = 7
W = 7
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
REDUCTION_SCALE = 1.0 / N_HW
FULL_BLOCK_SIZE = 32768



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config_on_device(config: dict, device: torch.device) -> dict:
    return {
        "inputs": [
            {**spec, "device": str(device)}
            if isinstance(spec, dict) and spec.get("kind") == "tensor"
            else spec
            for spec in config["inputs"]
        ]
    }


def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = _config_on_device(next(iter(configs.values())), device)
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


def _assert_inputs(
    getitem: torch.Tensor,
    arg472_1: torch.Tensor,
    arg473_1: torch.Tensor,
    arg474_1: torch.Tensor,
    arg192_1: torch.Tensor,
    arg193_1: torch.Tensor,
    full: torch.Tensor,
) -> None:
    expected = {
        "getitem": ((N, C, 1, 1), torch.float32),
        "arg472_1": ((N, C, H, W), torch.float32),
        "arg473_1": ((1, C, 1, 1), torch.float32),
        "arg474_1": ((1, C, 1, 1), torch.float32),
        "arg192_1": ((C,), torch.float32),
        "arg193_1": ((C,), torch.float32),
        "full": ((), torch.float32),
    }
    values = {
        "getitem": getitem,
        "arg472_1": arg472_1,
        "arg473_1": arg473_1,
        "arg474_1": arg474_1,
        "arg192_1": arg192_1,
        "arg193_1": arg193_1,
        "full": full,
    }
    for name, tensor in values.items():
        shape, dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} has dtype {tensor.dtype}, expected {dtype}")


def oracle_torch(
    getitem: torch.Tensor,
    arg472_1: torch.Tensor,
    arg473_1: torch.Tensor,
    arg474_1: torch.Tensor,
    arg192_1: torch.Tensor,
    arg193_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0
    _assert_inputs(getitem, arg472_1, arg473_1, arg474_1, arg192_1, arg193_1, full)

    mean = arg473_1.reshape(C)
    invstd = arg474_1.reshape(C)
    centered = arg472_1 - mean[None, :, None, None]
    affine = (
        centered
        * invstd[None, :, None, None]
        * arg192_1[None, :, None, None]
        + arg193_1[None, :, None, None]
    )
    pool_grad = getitem[:, :, 0, 0][:, :, None, None] * INV_HW
    grad_bn_out = torch.where(torch.relu(affine) <= 0.0, full, pool_grad)

    grad_sum = grad_bn_out.sum(dim=(0, 2, 3))
    centered_grad_sum = (grad_bn_out * centered).sum(dim=(0, 2, 3))
    mean_term = grad_sum * REDUCTION_SCALE
    var_term = centered_grad_sum * REDUCTION_SCALE * invstd * invstd
    input_scale = invstd * arg192_1

    out0_value = (
        grad_bn_out
        - centered * var_term[None, :, None, None]
        - mean_term[None, :, None, None]
    ) * input_scale[None, :, None, None]
    out0 = torch.empty_strided(
        tuple(arg472_1.shape),
        tuple(arg472_1.stride()),
        device=arg472_1.device,
        dtype=arg472_1.dtype,
    )
    out0.copy_(out0_value)
    out1 = centered_grad_sum * invstd
    return out0, out1


if triton is not None:

    @triton.jit
    def _onepass_relu_bn_channel_kernel(
        getitem_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        out0_ptr,
        out1_ptr,
        getitem_stride_n: tl.constexpr,
        getitem_stride_c: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_SIZE_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE_)
        active = offsets < N_HW_
        n_idx = offsets // HW_
        spatial = offsets - n_idx * HW_

        getitem_offsets = n_idx * getitem_stride_n + channel * getitem_stride_c
        x_offsets = (
            n_idx * x_stride_n
            + channel * x_stride_c
            + spatial
        )
        out_offsets = (
            n_idx * out_stride_n
            + channel * out_stride_c
            + spatial
        )

        mean = tl.load(mean_ptr + channel * mean_stride_c).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel * invstd_stride_c).to(tl.float32)
        weight = tl.load(weight_ptr + channel * weight_stride_c).to(tl.float32)
        bias = tl.load(bias_ptr + channel * bias_stride_c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        pool = tl.load(getitem_ptr + getitem_offsets, mask=active, other=0.0).to(tl.float32) * INV_HW_
        centered = x - mean
        affine = centered * invstd * weight + bias
        gate = ((affine > 0.0) | (affine != affine)) & active
        grad = tl.where(gate, pool, full_value)
        grad = tl.where(active, grad, 0.0)

        sum0 = tl.sum(grad, axis=0)
        sum1 = tl.sum(grad * tl.where(active, centered, 0.0), axis=0)
        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        out = (grad - centered * var_term - mean_term) * invstd * weight

        tl.store(out0_ptr + out_offsets, out, mask=active)
        tl.store(out1_ptr + channel, sum1 * invstd)


def oracle_triton(
    getitem: torch.Tensor,
    arg472_1: torch.Tensor,
    arg473_1: torch.Tensor,
    arg474_1: torch.Tensor,
    arg192_1: torch.Tensor,
    arg193_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    _assert_inputs(getitem, arg472_1, arg473_1, arg474_1, arg192_1, arg193_1, full)

    out0 = torch.empty_strided(
        tuple(arg472_1.shape),
        tuple(arg472_1.stride()),
        device=arg472_1.device,
        dtype=arg472_1.dtype,
    )
    out1 = torch.empty((C,), device=arg472_1.device, dtype=torch.float32)

    _onepass_relu_bn_channel_kernel[(C,)](
        getitem,
        arg472_1,
        arg473_1,
        arg474_1,
        arg192_1,
        arg193_1,
        full,
        out0,
        out1,
        getitem_stride_n=getitem.stride(0),
        getitem_stride_c=getitem.stride(1),
        x_stride_n=arg472_1.stride(0),
        x_stride_c=arg472_1.stride(1),
        mean_stride_c=arg473_1.stride(1),
        invstd_stride_c=arg474_1.stride(1),
        weight_stride_c=arg192_1.stride(0),
        bias_stride_c=arg193_1.stride(0),
        out_stride_n=out0.stride(0),
        out_stride_c=out0.stride(1),
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_SIZE_=FULL_BLOCK_SIZE,
        num_warps=8,
        num_stages=4,
    )
    return out0, out1


def oracle_structured_pool_upsample_backward_reduce(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
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


class OracleModule(torch.nn.Module):
    def __init__(self, impl: str = "torch") -> None:
        super().__init__()
        self.impl = impl

    def forward(self, *inputs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return oracle_structured_pool_upsample_backward_reduce(*inputs, impl=self.impl)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    if device.type != "cuda" or (device.index is not None and device.index != 0):
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    out = model(*inputs)
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError(f"expected a 2-tuple from Repro.forward, got {type(out)!r}")
    return out


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), diff.mean().item(), rel.max().item()


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_structured_pool_upsample_backward_reduce(*inputs, impl=impl)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, mean_abs, max_rel = _diff_stats(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and shape_ok and dtype_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={value_ok} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(max(1, warmup)):
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

    activation = inputs[1]
    if isinstance(activation, torch.Tensor):
        print(
            f"oracle shape: pooled=f32[{N}, {C}], activation=f32[{N}, {C}, {H}, {W}] "
            f"stride={activation.stride()} shape={SHAPE_LABEL}"
        )
    logical_read_bytes = (2 * N * C * HW + 4 * C + C) * 4
    logical_write_bytes = (N * C * HW + C) * 4
    print(f"one-pass logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_structured_pool_upsample_backward_reduce(*inputs, impl=actual_impl)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_structured_pool_upsample_backward_reduce(*inputs, impl=actual_impl),
            device,
            warmup,
            rep,
        )

    print(
        f"oracle_structured_pool_upsample_backward_reduce: {oracle_us:.3f} us "
        f"impl={actual_impl} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare full oracle outputs with Repro.forward")
    parser.add_argument("--bench", action="store_true", help="benchmark the full oracle")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    with torch.no_grad():
        if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
            sys.exit(1)
        if args.bench:
            run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
