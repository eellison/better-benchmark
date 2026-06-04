"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the Visformer global average-pool backward `as_strided_scatter -> expand -> div` as a structured scatter-reduce, deriving the two `[768]` channel summaries from the `[128,768]` pooled-gradient source while writing the required `[128,768,7,7]` BN-backward side output directly, whereas Inductor currently materializes the expanded `[128,768,7,7]` pool-gradient tensor and schedules its sibling reductions as generic consumers; Inductor cannot do this today because its scheduler/codegen does not model view/as_strided scatter followed by broadcasted average-pool backward as a structured scatter-reduce with a materialized side output and channel-reduction epilogue; the fix is SCATTER_REDUCE: add a structured global-average-pool-backward lowering that computes reductions from the source tile while emitting the required expanded side-output stores."""
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
except ModuleNotFoundError:
    triton = None
    tl = None


REPRO_ID = "sum_sum_2778c0896f68"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_timm_visformer_small_train_train_001_3aaad9f0"

N = 128
C = 768
H = 7
W = 7
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
REDUCTION_SCALE = 1.0 / (N * HW)
BLOCK_SIZE = 8192



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
    def _avgpool_scatter_reduce_kernel(
        mm_ptr,
        x_ptr,
        norm_ptr,
        scale_ptr,
        out0_ptr,
        out1_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_SIZE_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE_)
        mask = offsets < N_HW_

        n_idx = offsets // HW_
        hw_idx = offsets % HW_
        nchw_offsets = n_idx * (C_ * HW_) + channel * HW_ + hw_idx

        mm_vals = tl.load(mm_ptr + n_idx * C_ + channel, mask=mask, other=0.0).to(tl.float32)
        x_vals = tl.load(x_ptr + nchw_offsets, mask=mask, other=0.0).to(tl.float32)
        pool_vals = mm_vals * INV_HW_

        pool_sum = tl.sum(pool_vals, axis=0)
        weighted_sum = tl.sum(pool_vals * x_vals, axis=0)

        norm = tl.load(norm_ptr + channel).to(tl.float32)
        scale = tl.load(scale_ptr + channel).to(tl.float32)
        pool_mean = pool_sum * REDUCTION_SCALE_
        weighted_mean_norm2 = weighted_sum * REDUCTION_SCALE_ * norm * norm
        affine_scale = norm * scale

        out_vals = (pool_vals - x_vals * weighted_mean_norm2 - pool_mean) * affine_scale
        tl.store(out0_ptr + nchw_offsets, out_vals, mask=mask)
        tl.store(out1_ptr + channel, weighted_sum * norm)


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
    arg215_1: torch.Tensor,
    arg213_1: torch.Tensor,
    arg91_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-space PyTorch version of the average-pool-backward scatter-reduce."""
    n, _, h, w = arg215_1.shape
    hw = h * w
    reduction_scale = 1.0 / (n * hw)

    pool_grad = mm[:, :, None, None] / hw
    pool_sum = pool_grad.expand_as(arg215_1).sum(dim=(0, 2, 3))
    weighted_sum = (pool_grad * arg215_1).sum(dim=(0, 2, 3))

    pool_mean = pool_sum * reduction_scale
    weighted_mean_norm2 = weighted_sum * reduction_scale * arg213_1 * arg213_1
    affine_scale = arg213_1 * arg91_1

    out0 = (
        pool_grad
        - arg215_1 * weighted_mean_norm2[None, :, None, None]
        - pool_mean[None, :, None, None]
    ) * affine_scale[None, :, None, None]
    out1 = weighted_sum * arg213_1
    return out0, out1


def oracle_triton(
    mm: torch.Tensor,
    arg215_1: torch.Tensor,
    arg213_1: torch.Tensor,
    arg91_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    out0 = torch.empty_like(arg215_1)
    out1 = torch.empty_like(arg213_1)

    _avgpool_scatter_reduce_kernel[(C,)](
        mm,
        arg215_1,
        arg213_1,
        arg91_1,
        out0,
        out1,
        C_=C,
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_SIZE_=BLOCK_SIZE,
        num_warps=8,
    )
    return out0, out1


def oracle_scatter_reduce(*inputs: object, impl: str = "auto") -> tuple[torch.Tensor, torch.Tensor]:
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


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        actual = oracle_scatter_reduce(*inputs, impl=impl)
        synchronize(device)

    ok = True
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
    actual_impl = impl
    if actual_impl == "auto":
        actual_impl = "triton" if device.type == "cuda" and triton is not None else "torch"

    with torch.no_grad():
        oracle_scatter_reduce(*inputs, impl=actual_impl)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_scatter_reduce(*inputs, impl=actual_impl),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_scatter_reduce: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device}"
    )


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
    args = parser.parse_args()

    if not args.check and not args.bench:
        args.check = True
        args.bench = True

    device = torch.device(args.device)
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
