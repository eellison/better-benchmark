"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the MobileViT adaptive-average-pool backward, SiLU backward, and batch-norm-backward tuple directly from the `[128, 640]` pooled gradient and `[128, 640, 8, 8]` activation tensor, emitting both the returned input-gradient tensor and gamma-gradient vector without materializing the zero-fill `as_strided_scatter -> as_strided -> expand -> div` pool-gradient tensor, whereas Inductor currently lowers that structured scatter/expand as a generic producer and schedules the returned full tensor plus sibling channel reductions as ordinary consumers; Inductor cannot do this today because its scheduler/codegen does not model zero-fill view/as_strided scatter followed by broadcasted average-pool backward as a structured scatter-reduce with both materialized side-output stores and reduction epilogues; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter/expand lowering that shares the pooled-gradient source across the SiLU/BN backward pointwise, emits the required full tensor, and accumulates sibling channel reductions in the same template."""
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
except ModuleNotFoundError:  # pragma: no cover - allows CPU-only syntax checks.
    triton = None
    tl = None


REPRO_ID = "sum_sum_609200a71825"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_mobilevit_s_train_001_7c8f2393"

N = 128
C = 640
H = 8
W = 8
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
REDUCTION_SCALE = 1.0 / (N * HW)
BLOCK_SIZE = 1024
N_TILES = math.ceil(N_HW / BLOCK_SIZE)
FULL_BLOCK_SIZE = N_HW



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
    arg391_1: torch.Tensor,
    arg392_1: torch.Tensor,
    arg393_1: torch.Tensor,
    arg148_1: torch.Tensor,
    arg149_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-space average-pool-backward + SiLU + BN-backward tuple."""
    x = arg391_1
    mean = arg392_1
    invstd = arg393_1
    weight = arg148_1[None, :, None, None]
    bias = arg149_1[None, :, None, None]

    centered = x - mean
    affine = centered * invstd * weight + bias
    sigmoid = torch.reciprocal(torch.exp(torch.neg(affine)) + 1.0)
    silu_backward = sigmoid * (affine * (1.0 - sigmoid) + 1.0)
    pool_grad = mm[:, :, None, None] * INV_HW
    grad_bn_out = pool_grad * silu_backward

    grad_sum = grad_bn_out.sum(dim=(0, 2, 3))
    centered_grad_sum = (grad_bn_out * centered).sum(dim=(0, 2, 3))
    mean_term = grad_sum * REDUCTION_SCALE
    var_term = centered_grad_sum * REDUCTION_SCALE * arg393_1.reshape(C) * arg393_1.reshape(C)
    input_scale = arg393_1.reshape(C) * arg148_1

    out0 = (
        grad_bn_out
        - centered * var_term[None, :, None, None]
        - mean_term[None, :, None, None]
    ) * input_scale[None, :, None, None]
    out1 = centered_grad_sum * arg393_1.reshape(C)
    return out0, out1


if triton is not None:

    @triton.jit
    def _partial_reduce_kernel(
        mm_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        partial0_ptr,
        partial1_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        BLOCK_SIZE_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_SIZE_ + tl.arange(0, BLOCK_SIZE_)
        mask = offsets < N_HW_

        n_idx = offsets // HW_
        hw_idx = offsets - n_idx * HW_
        nchw_offsets = n_idx * (C_ * HW_) + channel * HW_ + hw_idx

        x_vals = tl.load(x_ptr + nchw_offsets, mask=mask, other=0.0).to(tl.float32)
        mm_vals = tl.load(mm_ptr + n_idx * C_ + channel, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)

        centered = x_vals - mean
        affine = centered * invstd * weight + bias
        sigmoid = 1.0 / (tl.exp(-affine) + 1.0)
        grad_bn_out = mm_vals * INV_HW_ * sigmoid * (affine * (1.0 - sigmoid) + 1.0)
        grad_bn_out = tl.where(mask, grad_bn_out, 0.0)
        centered_prod = tl.where(mask, grad_bn_out * centered, 0.0)

        partial_idx = channel * tl.num_programs(1) + tile
        tl.store(partial0_ptr + partial_idx, tl.sum(grad_bn_out, axis=0))
        tl.store(partial1_ptr + partial_idx, tl.sum(centered_prod, axis=0))

    @triton.jit
    def _finalize_reduce_kernel(
        partial0_ptr,
        partial1_ptr,
        invstd_ptr,
        sum0_ptr,
        sum1_ptr,
        out1_ptr,
        N_TILES_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_TILES_)
        mask = offsets < N_TILES_
        base = channel * N_TILES_
        sum0_vals = tl.load(partial0_ptr + base + offsets, mask=mask, other=0.0)
        sum1_vals = tl.load(partial1_ptr + base + offsets, mask=mask, other=0.0)
        sum0 = tl.sum(sum0_vals, axis=0)
        sum1 = tl.sum(sum1_vals, axis=0)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        tl.store(sum0_ptr + channel, sum0)
        tl.store(sum1_ptr + channel, sum1)
        tl.store(out1_ptr + channel, sum1 * invstd)

    @triton.jit
    def _write_output_kernel(
        mm_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_SIZE_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_SIZE_ + tl.arange(0, BLOCK_SIZE_)
        mask = offsets < N_HW_

        n_idx = offsets // HW_
        hw_idx = offsets - n_idx * HW_
        nchw_offsets = n_idx * (C_ * HW_) + channel * HW_ + hw_idx

        x_vals = tl.load(x_ptr + nchw_offsets, mask=mask, other=0.0).to(tl.float32)
        mm_vals = tl.load(mm_ptr + n_idx * C_ + channel, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        sum0 = tl.load(sum0_ptr + channel).to(tl.float32)
        sum1 = tl.load(sum1_ptr + channel).to(tl.float32)

        centered = x_vals - mean
        affine = centered * invstd * weight + bias
        sigmoid = 1.0 / (tl.exp(-affine) + 1.0)
        grad_bn_out = mm_vals * INV_HW_ * sigmoid * (affine * (1.0 - sigmoid) + 1.0)
        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        out_vals = (grad_bn_out - centered * var_term - mean_term) * invstd * weight
        tl.store(out0_ptr + nchw_offsets, out_vals, mask=mask)

    @triton.jit
    def _onepass_avgpool_silu_bn_kernel(
        mm_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
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
        hw_idx = offsets - n_idx * HW_
        nchw_offsets = n_idx * (C_ * HW_) + channel * HW_ + hw_idx

        x_vals = tl.load(x_ptr + nchw_offsets, mask=mask, other=0.0).to(tl.float32)
        mm_vals = tl.load(mm_ptr + n_idx * C_ + channel, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)

        centered = x_vals - mean
        affine = centered * invstd * weight + bias
        sigmoid = 1.0 / (tl.exp(-affine) + 1.0)
        grad_bn_out = mm_vals * INV_HW_ * sigmoid * (affine * (1.0 - sigmoid) + 1.0)
        grad_bn_out = tl.where(mask, grad_bn_out, 0.0)

        sum0 = tl.sum(grad_bn_out, axis=0)
        sum1 = tl.sum(tl.where(mask, grad_bn_out * centered, 0.0), axis=0)
        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        out_vals = (grad_bn_out - centered * var_term - mean_term) * invstd * weight

        tl.store(out0_ptr + nchw_offsets, out_vals, mask=mask)
        tl.store(out1_ptr + channel, sum1 * invstd)


def oracle_triton(
    mm: torch.Tensor,
    arg391_1: torch.Tensor,
    arg392_1: torch.Tensor,
    arg393_1: torch.Tensor,
    arg148_1: torch.Tensor,
    arg149_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    out0 = torch.empty_like(arg391_1)
    out1 = torch.empty_like(arg148_1)

    _onepass_avgpool_silu_bn_kernel[(C,)](
        mm,
        arg391_1,
        arg392_1,
        arg393_1,
        arg148_1,
        arg149_1,
        out0,
        out1,
        C_=C,
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_SIZE_=FULL_BLOCK_SIZE,
        num_warps=16,
    )
    return out0, out1


def oracle_avgpool_silu_bn_backward(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor]:
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
        actual = oracle_avgpool_silu_bn_backward(*inputs, impl=impl)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

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
        oracle_avgpool_silu_bn_backward(*inputs, impl=actual_impl)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_avgpool_silu_bn_backward(*inputs, impl=actual_impl),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_avgpool_silu_bn_backward: {oracle_us:.3f} us "
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
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
