"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full VoVNet max-pool-backward scatter_add, ReLU-gated mask overwrite, and batch-norm-backward return tuple directly from the `[32,512,14,14]` pool-gradient source and low-memory max-pool offsets, reusing the same source-space scatter reconstruction for the two channel reductions and the `[32,512,28,28]` side output, whereas Inductor currently materializes the dense `[16384,784]` scatter_add result and schedules the ReLU gate, sibling channel reductions, and dependent BN-backward pointwise output as separate generic kernels; Inductor cannot do this today because scheduler/codegen does not model low-memory max-pool-backward scatter_add as a structured scatter-reduce producer with both reduction epilogues and a required materialized side-output store; the fix is SCATTER_REDUCE: add a max-pool-backward scatter-reduce lowering that gathers contributing pool-gradient elements from offsets, accumulates the BN channel summaries, and emits the full BN input-gradient tensor from the same structured template."""
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
except ModuleNotFoundError:  # pragma: no cover - keeps syntax checks usable without Triton.
    triton = None
    tl = None


REPRO_ID = "sum_sum_adb09ad67b46"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_timm_vovnet_train_001_9aaf945e"

N = 32
C = 512
GETITEM54_C = 1472
OH = 14
OW = 14
IH = 28
IW = 28
KERNEL = 3
STRIDE = 2
OUT_HW = OH * OW
IN_HW = IH * IW
TOTAL_IN = N * IN_HW
REDUCTION_SCALE = 1.0 / (N * IH * IW)
BLOCK_SIZE = 1024

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
    inputs = module.make_inputs()
    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def _assert_common_inputs(
    getitem_54: torch.Tensor,
    getitem_69: torch.Tensor,
    arg140_1: torch.Tensor,
    arg136_1: torch.Tensor,
    arg137_1: torch.Tensor,
    arg138_1: torch.Tensor,
    arg33_1: torch.Tensor,
    arg34_1: torch.Tensor,
    full: torch.Tensor,
) -> None:
    expected = {
        "getitem_54": ((N, GETITEM54_C, OH, OW), torch.float32),
        "getitem_69": ((N, C, OH, OW), torch.float32),
        "arg140_1": ((N, C, OH, OW), torch.int8),
        "arg136_1": ((N, C, IH, IW), torch.float32),
        "arg137_1": ((1, C, 1, 1), torch.float32),
        "arg138_1": ((1, C, 1, 1), torch.float32),
        "arg33_1": ((C,), torch.float32),
        "arg34_1": ((C,), torch.float32),
        "full": ((), torch.float32),
    }
    values = {
        "getitem_54": getitem_54,
        "getitem_69": getitem_69,
        "arg140_1": arg140_1,
        "arg136_1": arg136_1,
        "arg137_1": arg137_1,
        "arg138_1": arg138_1,
        "arg33_1": arg33_1,
        "arg34_1": arg34_1,
        "full": full,
    }
    for name, tensor in values.items():
        shape, dtype = expected[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} has dtype {tensor.dtype}, expected {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for this fixed-shape oracle")


def _pool_backward_scatter_torch(
    getitem_54: torch.Tensor,
    getitem_69: torch.Tensor,
    arg140_1: torch.Tensor,
) -> torch.Tensor:
    pool_grad = getitem_54[:, :C, :, :] + getitem_69
    scatter = torch.zeros((N, C, IH, IW), device=pool_grad.device, dtype=pool_grad.dtype)
    for ky in range(KERNEL):
        h_len = min(OH, max(0, (IH - 1 - ky) // STRIDE + 1))
        if h_len == 0:
            continue
        for kx in range(KERNEL):
            w_len = min(OW, max(0, (IW - 1 - kx) // STRIDE + 1))
            if w_len == 0:
                continue
            offset = ky * KERNEL + kx
            scatter[:, :, ky : ky + STRIDE * h_len : STRIDE, kx : kx + STRIDE * w_len : STRIDE].add_(
                torch.where(
                    arg140_1[:, :, :h_len, :w_len] == offset,
                    pool_grad[:, :, :h_len, :w_len],
                    torch.zeros((), device=pool_grad.device, dtype=pool_grad.dtype),
                )
            )
    return scatter


def oracle_torch(
    getitem_54: torch.Tensor,
    getitem_69: torch.Tensor,
    arg140_1: torch.Tensor,
    arg136_1: torch.Tensor,
    arg137_1: torch.Tensor,
    arg138_1: torch.Tensor,
    arg33_1: torch.Tensor,
    arg34_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2
    _assert_common_inputs(
        getitem_54,
        getitem_69,
        arg140_1,
        arg136_1,
        arg137_1,
        arg138_1,
        arg33_1,
        arg34_1,
        full,
    )

    scatter = _pool_backward_scatter_torch(getitem_54, getitem_69, arg140_1)
    mean = arg137_1.reshape(C)
    invstd = arg138_1.reshape(C)
    centered = arg136_1 - mean[None, :, None, None]
    affine = centered * invstd[None, :, None, None] * arg33_1[None, :, None, None] + arg34_1[
        None, :, None, None
    ]
    grad_bn_out = torch.where(affine <= 0.0, full, scatter)

    grad_sum = grad_bn_out.sum(dim=(0, 2, 3))
    centered_grad_sum = (grad_bn_out * centered).sum(dim=(0, 2, 3))
    mean_term = grad_sum * REDUCTION_SCALE
    var_term = centered_grad_sum * REDUCTION_SCALE * invstd * invstd
    input_scale = invstd * arg33_1
    out0 = (grad_bn_out - centered * var_term[None, :, None, None] - mean_term[None, :, None, None]) * input_scale[
        None, :, None, None
    ]
    out1 = centered_grad_sum * invstd
    return out0, out1


if triton is not None:

    @triton.jit
    def _scatter_value_from_lowmem_offsets(
        getitem_54_ptr,
        getitem_69_ptr,
        offsets_ptr,
        n_idx,
        c,
        ih,
        iw,
        active,
        BLOCK_: tl.constexpr,
    ):
        accum = tl.full((BLOCK_,), 0.0, tl.float32)
        for ky in tl.static_range(0, 3):
            dy = ih - ky
            oh = dy // 2
            valid_y = (ih >= ky) & ((dy - oh * 2) == 0) & (oh < 14)
            for kx in tl.static_range(0, 3):
                dx = iw - kx
                ow = dx // 2
                valid_x = (iw >= kx) & ((dx - ow * 2) == 0) & (ow < 14)
                source_valid = active & valid_y & valid_x
                pool_idx = (n_idx * 512 + c) * 196 + oh * 14 + ow
                lowmem_offset = tl.load(offsets_ptr + pool_idx, mask=source_valid, other=-1).to(tl.int32)
                take = source_valid & (lowmem_offset == ky * 3 + kx)
                getitem54_idx = (n_idx * 1472 + c) * 196 + oh * 14 + ow
                grad = tl.load(getitem_54_ptr + getitem54_idx, mask=take, other=0.0).to(tl.float32)
                grad += tl.load(getitem_69_ptr + pool_idx, mask=take, other=0.0).to(tl.float32)
                accum += grad
        return accum

    @triton.jit
    def _partial_bn_sums_kernel(
        getitem_54_ptr,
        getitem_69_ptr,
        offsets_ptr,
        activation_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial0_ptr,
        partial1_ptr,
        num_tiles: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK_ + tl.arange(0, BLOCK_)
        active = k < 25088

        n_idx = k // 784
        hw = k - n_idx * 784
        ih = hw // 28
        iw = hw - ih * 28

        scatter = _scatter_value_from_lowmem_offsets(
            getitem_54_ptr,
            getitem_69_ptr,
            offsets_ptr,
            n_idx,
            c,
            ih,
            iw,
            active,
            BLOCK_=BLOCK_,
        )
        activation_idx = (n_idx * 512 + c) * 784 + hw
        x = tl.load(activation_ptr + activation_idx, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        centered = x - mean
        affine = centered * invstd * weight + bias
        grad_bn_out = tl.where(affine <= 0.0, full_value, scatter)
        grad_bn_out = tl.where(active, grad_bn_out, 0.0)
        centered = tl.where(active, centered, 0.0)

        tl.store(partial0_ptr + c * num_tiles + tile, tl.sum(grad_bn_out, axis=0))
        tl.store(partial1_ptr + c * num_tiles + tile, tl.sum(grad_bn_out * centered, axis=0))

    @triton.jit
    def _finalize_bn_sums_kernel(
        partial0_ptr,
        partial1_ptr,
        invstd_ptr,
        sum0_ptr,
        sum1_ptr,
        out1_ptr,
        num_tiles: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES_)
        active = tiles < num_tiles
        sum0_vals = tl.load(partial0_ptr + c * num_tiles + tiles, mask=active, other=0.0).to(tl.float32)
        sum1_vals = tl.load(partial1_ptr + c * num_tiles + tiles, mask=active, other=0.0).to(tl.float32)
        sum0 = tl.sum(sum0_vals, axis=0)
        sum1 = tl.sum(sum1_vals, axis=0)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        tl.store(sum0_ptr + c, sum0)
        tl.store(sum1_ptr + c, sum1)
        tl.store(out1_ptr + c, sum1 * invstd)

    @triton.jit
    def _bn_input_grad_kernel(
        getitem_54_ptr,
        getitem_69_ptr,
        offsets_ptr,
        activation_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        BLOCK_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK_ + tl.arange(0, BLOCK_)
        active = k < 25088

        n_idx = k // 784
        hw = k - n_idx * 784
        ih = hw // 28
        iw = hw - ih * 28

        scatter = _scatter_value_from_lowmem_offsets(
            getitem_54_ptr,
            getitem_69_ptr,
            offsets_ptr,
            n_idx,
            c,
            ih,
            iw,
            active,
            BLOCK_=BLOCK_,
        )
        activation_idx = (n_idx * 512 + c) * 784 + hw
        x = tl.load(activation_ptr + activation_idx, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        sum0 = tl.load(sum0_ptr + c).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c).to(tl.float32)

        centered = x - mean
        affine = centered * invstd * weight + bias
        grad_bn_out = tl.where(affine <= 0.0, full_value, scatter)
        mean_term = sum0 * 3.985969387755102e-05
        var_term = sum1 * 3.985969387755102e-05 * invstd * invstd
        input_scale = invstd * weight
        out = (grad_bn_out - centered * var_term - mean_term) * input_scale
        tl.store(out0_ptr + activation_idx, out, mask=active)


def oracle_triton(
    getitem_54: torch.Tensor,
    getitem_69: torch.Tensor,
    arg140_1: torch.Tensor,
    arg136_1: torch.Tensor,
    arg137_1: torch.Tensor,
    arg138_1: torch.Tensor,
    arg33_1: torch.Tensor,
    arg34_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem_54.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")
    _assert_common_inputs(
        getitem_54,
        getitem_69,
        arg140_1,
        arg136_1,
        arg137_1,
        arg138_1,
        arg33_1,
        arg34_1,
        full,
    )

    num_tiles = triton.cdiv(TOTAL_IN, BLOCK_SIZE)
    partial0 = torch.empty((C, num_tiles), device=arg136_1.device, dtype=torch.float32)
    partial1 = torch.empty((C, num_tiles), device=arg136_1.device, dtype=torch.float32)
    sum0 = torch.empty((C,), device=arg136_1.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=arg136_1.device, dtype=torch.float32)
    out0 = torch.empty_like(arg136_1)
    out1 = torch.empty((C,), device=arg136_1.device, dtype=torch.float32)

    grid = (C, num_tiles)
    _partial_bn_sums_kernel[grid](
        getitem_54,
        getitem_69,
        arg140_1,
        arg136_1,
        arg137_1,
        arg138_1,
        arg33_1,
        arg34_1,
        full,
        partial0,
        partial1,
        num_tiles=num_tiles,
        BLOCK_=BLOCK_SIZE,
        num_warps=8,
    )
    _finalize_bn_sums_kernel[(C,)](
        partial0,
        partial1,
        arg138_1,
        sum0,
        sum1,
        out1,
        num_tiles=num_tiles,
        BLOCK_TILES_=triton.next_power_of_2(num_tiles),
        num_warps=4,
    )
    _bn_input_grad_kernel[grid](
        getitem_54,
        getitem_69,
        arg140_1,
        arg136_1,
        arg137_1,
        arg138_1,
        arg33_1,
        arg34_1,
        full,
        sum0,
        sum1,
        out0,
        BLOCK_=BLOCK_SIZE,
        num_warps=8,
    )
    return out0, out1


def oracle_structured_pool_upsample_reduce(
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
        return oracle_structured_pool_upsample_reduce(*inputs, impl=self.impl)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    model = module.Repro().to(device)
    out = model(*inputs)
    if not isinstance(out, tuple) or len(out) != 2:
        raise TypeError(f"expected a 2-tuple from Repro.forward, got {type(out)!r}")
    return out


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

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        shape_ok = got.shape == ref.shape
        stride_ok = got.stride() == ref.stride()
        ok = ok and value_ok and shape_ok and stride_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} stride={got.stride()} "
            f"expected_stride={ref.stride()} max_abs={max_abs:.6e} "
            f"max_rel={max_rel:.6e} allclose={value_ok} "
            f"shape_match={shape_ok} stride_match={stride_ok}"
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
        f"impl={actual_impl} shape={SHAPE_LABEL} block={BLOCK_SIZE}"
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
    if args.check and not run_check(device=device, impl=args.impl, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
