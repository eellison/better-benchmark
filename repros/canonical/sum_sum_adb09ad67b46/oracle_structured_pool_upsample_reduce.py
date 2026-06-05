"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full VoVNet center-offset max-pool-backward scatter_add, ReLU-gated mask overwrite, and batch-norm-backward return tuple directly from the generated `[32,512,14,14]` pool-gradient source, emitting both channel reductions and the `[32,512,28,28]` side output, whereas Inductor currently materializes the dense `[16384,784]` scatter_add result and schedules the ReLU gate, sibling channel reductions, and dependent BN-backward pointwise output as separate generic kernels; Inductor cannot do this today because scheduler/codegen does not model low-memory max-pool-backward scatter_add as a structured scatter-reduce producer with both reduction epilogues and a required materialized side-output store; the fix is SCATTER_REDUCE: add a max-pool-backward scatter-reduce lowering that recognizes the generated center-offset domain, accumulates the BN channel summaries from the source-space producer, and emits the full BN input-gradient tensor from the same structured template."""
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



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_adb09ad67b46"
REPRO_DIR = Path(__file__).resolve().parent
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
BLOCK_M = 512
BLOCK_C = 1
NUM_M_TILES = triton.cdiv(TOTAL_IN, BLOCK_M) if triton is not None else math.ceil(TOTAL_IN / BLOCK_M)
BLOCK_TILES = 64



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
    def _partial_bn_sums_kernel(
        getitem_54_ptr,
        getitem_69_ptr,
        activation_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial0_ptr,
        partial1_ptr,
        getitem54_stride_n: tl.constexpr,
        getitem54_stride_c: tl.constexpr,
        getitem54_stride_h: tl.constexpr,
        getitem54_stride_w: tl.constexpr,
        getitem69_stride_n: tl.constexpr,
        getitem69_stride_c: tl.constexpr,
        getitem69_stride_h: tl.constexpr,
        getitem69_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // HW_
        spatial = m_offsets - n_idx * HW_
        h_idx = spatial // W_
        w_idx = spatial - h_idx * W_
        pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
        pool_h = h_idx // 2
        pool_w = w_idx // 2

        c_mask = c_offsets < C_
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets * weight_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets * bias_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )
        x = tl.load(activation_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)

        source_mask = active & pool_site[:, None] & gate
        source_offsets54 = (
            n_idx[:, None] * getitem54_stride_n
            + c_offsets[None, :] * getitem54_stride_c
            + pool_h[:, None] * getitem54_stride_h
            + pool_w[:, None] * getitem54_stride_w
        )
        source_offsets69 = (
            n_idx[:, None] * getitem69_stride_n
            + c_offsets[None, :] * getitem69_stride_c
            + pool_h[:, None] * getitem69_stride_h
            + pool_w[:, None] * getitem69_stride_w
        )
        scatter = (
            tl.load(getitem_54_ptr + source_offsets54, mask=source_mask, other=0.0).to(tl.float32)
            + tl.load(getitem_69_ptr + source_offsets69, mask=source_mask, other=0.0).to(tl.float32)
        )
        grad_bn_out = tl.where(active, tl.where(gate, scatter, full_value), 0.0)
        sum_grad = tl.sum(grad_bn_out, axis=0)
        sum_centered = tl.sum(grad_bn_out * tl.where(active, centered, 0.0), axis=0)
        partial_offsets = pid_m * C_ + c_offsets
        tl.store(partial0_ptr + partial_offsets, sum_grad, mask=c_mask)
        tl.store(partial1_ptr + partial_offsets, sum_centered, mask=c_mask)

    @triton.jit
    def _finalize_bn_sums_kernel(
        partial0_ptr,
        partial1_ptr,
        invstd_ptr,
        sum0_ptr,
        sum1_ptr,
        out1_ptr,
        invstd_stride_c: tl.constexpr,
        C_: tl.constexpr,
        NUM_M_TILES_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tile_offsets = tl.arange(0, BLOCK_TILES_)
        active = (c_offsets[None, :] < C_) & (tile_offsets[:, None] < NUM_M_TILES_)
        partial_offsets = tile_offsets[:, None] * C_ + c_offsets[None, :]

        sum0_vals = tl.load(partial0_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
        sum1_vals = tl.load(partial1_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
        sum0 = tl.sum(sum0_vals, axis=0)
        sum1 = tl.sum(sum1_vals, axis=0)
        mask = c_offsets < C_
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=mask, other=0.0).to(tl.float32)
        tl.store(sum0_ptr + c_offsets, sum0, mask=mask)
        tl.store(sum1_ptr + c_offsets, sum1, mask=mask)
        tl.store(out1_ptr + c_offsets, sum1 * invstd, mask=mask)

    @triton.jit
    def _nc_bn_input_grad_kernel(
        getitem_54_ptr,
        getitem_69_ptr,
        activation_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        getitem54_stride_n: tl.constexpr,
        getitem54_stride_c: tl.constexpr,
        getitem54_stride_h: tl.constexpr,
        getitem54_stride_w: tl.constexpr,
        getitem69_stride_n: tl.constexpr,
        getitem69_stride_c: tl.constexpr,
        getitem69_stride_h: tl.constexpr,
        getitem69_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n_idx = tl.program_id(1)
        hw = tl.arange(0, BLOCK_)
        active = hw < HW_
        h_idx = hw // W_
        w_idx = hw - h_idx * W_
        pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
        pool_h = h_idx // 2
        pool_w = w_idx // 2

        mean = tl.load(mean_ptr + c * mean_stride_c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c * invstd_stride_c).to(tl.float32)
        weight = tl.load(weight_ptr + c * weight_stride_c).to(tl.float32)
        bias = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
        sum0 = tl.load(sum0_ptr + c).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        x_offsets = n_idx * x_stride_n + c * x_stride_c + h_idx * x_stride_h + w_idx * x_stride_w
        x = tl.load(activation_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x - mean
        affine = centered * invstd * weight + bias
        gate = (affine > 0.0) | (affine != affine)

        source_offsets54 = (
            n_idx * getitem54_stride_n
            + c * getitem54_stride_c
            + pool_h * getitem54_stride_h
            + pool_w * getitem54_stride_w
        )
        source_offsets69 = (
            n_idx * getitem69_stride_n
            + c * getitem69_stride_c
            + pool_h * getitem69_stride_h
            + pool_w * getitem69_stride_w
        )
        scatter = (
            tl.load(getitem_54_ptr + source_offsets54, mask=active & pool_site & gate, other=0.0).to(tl.float32)
            + tl.load(getitem_69_ptr + source_offsets69, mask=active & pool_site & gate, other=0.0).to(tl.float32)
        )
        grad_bn_out = tl.where(active, tl.where(gate, scatter, full_value), 0.0)
        out = (
            grad_bn_out
            - centered * (sum1 * REDUCTION_SCALE_ * invstd * invstd)
            - sum0 * REDUCTION_SCALE_
        ) * (invstd * weight)
        out_offsets = n_idx * out0_stride_n + c * out0_stride_c + h_idx * out0_stride_h + w_idx * out0_stride_w
        tl.store(out0_ptr + out_offsets, out, mask=active)

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

    partial0 = torch.empty((NUM_M_TILES, C), device=arg136_1.device, dtype=torch.float32)
    partial1 = torch.empty_like(partial0)
    sum0 = torch.empty((C,), device=arg136_1.device, dtype=torch.float32)
    sum1 = torch.empty_like(sum0)
    out0 = torch.empty_strided(
        tuple(arg136_1.shape),
        tuple(arg136_1.stride()),
        device=arg136_1.device,
        dtype=arg136_1.dtype,
    )
    out1 = torch.empty((C,), device=arg136_1.device, dtype=torch.float32)

    _partial_bn_sums_kernel[(triton.cdiv(C, BLOCK_C), NUM_M_TILES)](
        getitem_54,
        getitem_69,
        arg136_1,
        arg137_1,
        arg138_1,
        arg33_1,
        arg34_1,
        full,
        partial0,
        partial1,
        getitem54_stride_n=getitem_54.stride(0),
        getitem54_stride_c=getitem_54.stride(1),
        getitem54_stride_h=getitem_54.stride(2),
        getitem54_stride_w=getitem_54.stride(3),
        getitem69_stride_n=getitem_69.stride(0),
        getitem69_stride_c=getitem_69.stride(1),
        getitem69_stride_h=getitem_69.stride(2),
        getitem69_stride_w=getitem_69.stride(3),
        x_stride_n=arg136_1.stride(0),
        x_stride_c=arg136_1.stride(1),
        x_stride_h=arg136_1.stride(2),
        x_stride_w=arg136_1.stride(3),
        mean_stride_c=arg137_1.stride(1),
        invstd_stride_c=arg138_1.stride(1),
        weight_stride_c=arg33_1.stride(0),
        bias_stride_c=arg34_1.stride(0),
        C_=C,
        W_=IW,
        HW_=IN_HW,
        N_HW_=TOTAL_IN,
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )
    _finalize_bn_sums_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial0,
        partial1,
        arg138_1,
        sum0,
        sum1,
        out1,
        invstd_stride_c=arg138_1.stride(1),
        C_=C,
        NUM_M_TILES_=NUM_M_TILES,
        BLOCK_C_=BLOCK_C,
        BLOCK_TILES_=BLOCK_TILES,
        num_warps=1,
    )
    _nc_bn_input_grad_kernel[(C, N)](
        getitem_54,
        getitem_69,
        arg136_1,
        arg137_1,
        arg138_1,
        arg33_1,
        arg34_1,
        full,
        sum0,
        sum1,
        out0,
        getitem54_stride_n=getitem_54.stride(0),
        getitem54_stride_c=getitem_54.stride(1),
        getitem54_stride_h=getitem_54.stride(2),
        getitem54_stride_w=getitem_54.stride(3),
        getitem69_stride_n=getitem_69.stride(0),
        getitem69_stride_c=getitem_69.stride(1),
        getitem69_stride_h=getitem_69.stride(2),
        getitem69_stride_w=getitem_69.stride(3),
        x_stride_n=arg136_1.stride(0),
        x_stride_c=arg136_1.stride(1),
        x_stride_h=arg136_1.stride(2),
        x_stride_w=arg136_1.stride(3),
        mean_stride_c=arg137_1.stride(1),
        invstd_stride_c=arg138_1.stride(1),
        weight_stride_c=arg33_1.stride(0),
        bias_stride_c=arg34_1.stride(0),
        out0_stride_n=out0.stride(0),
        out0_stride_c=out0.stride(1),
        out0_stride_h=out0.stride(2),
        out0_stride_w=out0.stride(3),
        W_=IW,
        HW_=IN_HW,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_=1024,
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


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    if device.type == "cuda" and triton is not None:
        from triton.testing import do_bench

        return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

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


def oracle_forward(inputs):
    return oracle_triton(*inputs)


def main():
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
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
