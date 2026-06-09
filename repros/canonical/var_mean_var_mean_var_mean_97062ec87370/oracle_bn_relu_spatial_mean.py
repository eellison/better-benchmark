"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Inception training-BatchNorm multi-branch scope by reducing each branch's channel statistics, updating all running mean/variance copy_ outputs, applying affine ReLU, preserving the fixed channel concatenation virtually, and writing the final spatial mean `[128, 2048]` directly without materializing the six normalized activations or intermediate cats, whereas Inductor currently schedules the independent BN reductions, mutable running-stat side effects, concat producers, and downstream spatial mean as generic regions with avoidable activation and concat traffic; Inductor cannot do this today because its scheduler does not keep fixed multi-branch concat producers virtual across training normalization reductions into a later reduction consumer with side-effect outputs; the fix is SCHEDULER_FUSION: teach the BN-training scheduler to expose running-stat epilogues while fusing static concat and spatial-mean consumers into the normalization plan."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


N = 128
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
OUT_CHANNELS = 2048
BRANCH_CHANNELS = (320, 384, 384, 384, 384, 192)
BRANCH_OFFSETS = (0, 320, 704, 1088, 1472, 1856)
EPS = 1.0e-3
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804
STAT_BLOCK = 8192
OUT_BLOCK_C = 16


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _load_branch_values(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
        x4_ptr,
        x5_ptr,
        channel,
        n_idx,
        hw_idx,
        mask,
        c0: tl.constexpr,
        c1: tl.constexpr,
        c2: tl.constexpr,
        c3: tl.constexpr,
        c4: tl.constexpr,
        c5: tl.constexpr,
        off0: tl.constexpr,
        off1: tl.constexpr,
        off2: tl.constexpr,
        off3: tl.constexpr,
        off4: tl.constexpr,
        off5: tl.constexpr,
        hw_size: tl.constexpr,
    ):
        b0 = (channel >= off0) & (channel < off0 + c0)
        b1 = (channel >= off1) & (channel < off1 + c1)
        b2 = (channel >= off2) & (channel < off2 + c2)
        b3 = (channel >= off3) & (channel < off3 + c3)
        b4 = (channel >= off4) & (channel < off4 + c4)
        b5 = (channel >= off5) & (channel < off5 + c5)

        lc0 = tl.where(b0, channel - off0, 0)
        lc1 = tl.where(b1, channel - off1, 0)
        lc2 = tl.where(b2, channel - off2, 0)
        lc3 = tl.where(b3, channel - off3, 0)
        lc4 = tl.where(b4, channel - off4, 0)
        lc5 = tl.where(b5, channel - off5, 0)

        vals = tl.load(x0_ptr + (n_idx * c0 + lc0) * hw_size + hw_idx, mask=mask & b0, other=0.0).to(tl.float32)
        vals += tl.load(x1_ptr + (n_idx * c1 + lc1) * hw_size + hw_idx, mask=mask & b1, other=0.0).to(tl.float32)
        vals += tl.load(x2_ptr + (n_idx * c2 + lc2) * hw_size + hw_idx, mask=mask & b2, other=0.0).to(tl.float32)
        vals += tl.load(x3_ptr + (n_idx * c3 + lc3) * hw_size + hw_idx, mask=mask & b3, other=0.0).to(tl.float32)
        vals += tl.load(x4_ptr + (n_idx * c4 + lc4) * hw_size + hw_idx, mask=mask & b4, other=0.0).to(tl.float32)
        vals += tl.load(x5_ptr + (n_idx * c5 + lc5) * hw_size + hw_idx, mask=mask & b5, other=0.0).to(tl.float32)
        return vals, b0, b1, b2, b3, b4, b5, lc0, lc1, lc2, lc3, lc4, lc5

    @triton.jit
    def _stats_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
        x4_ptr,
        x5_ptr,
        rm0_ptr,
        rv0_ptr,
        rm1_ptr,
        rv1_ptr,
        rm2_ptr,
        rv2_ptr,
        rm3_ptr,
        rv3_ptr,
        rm4_ptr,
        rv4_ptr,
        rm5_ptr,
        rv5_ptr,
        mean_ptr,
        invstd_ptr,
        elements_per_channel: tl.constexpr,
        hw_size: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size

        vals, b0, b1, b2, b3, b4, b5, lc0, lc1, lc2, lc3, lc4, lc5 = _load_branch_values(
            x0_ptr,
            x1_ptr,
            x2_ptr,
            x3_ptr,
            x4_ptr,
            x5_ptr,
            channel,
            n_idx,
            hw_idx,
            mask,
            c0=320,
            c1=384,
            c2=384,
            c3=384,
            c4=384,
            c5=192,
            off0=0,
            off1=320,
            off2=704,
            off3=1088,
            off4=1472,
            off5=1856,
            hw_size=64,
        )
        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

        old_rm0 = tl.load(rm0_ptr + lc0, mask=b0, other=0.0).to(tl.float32)
        old_rv0 = tl.load(rv0_ptr + lc0, mask=b0, other=0.0).to(tl.float32)
        old_rm1 = tl.load(rm1_ptr + lc1, mask=b1, other=0.0).to(tl.float32)
        old_rv1 = tl.load(rv1_ptr + lc1, mask=b1, other=0.0).to(tl.float32)
        old_rm2 = tl.load(rm2_ptr + lc2, mask=b2, other=0.0).to(tl.float32)
        old_rv2 = tl.load(rv2_ptr + lc2, mask=b2, other=0.0).to(tl.float32)
        old_rm3 = tl.load(rm3_ptr + lc3, mask=b3, other=0.0).to(tl.float32)
        old_rv3 = tl.load(rv3_ptr + lc3, mask=b3, other=0.0).to(tl.float32)
        old_rm4 = tl.load(rm4_ptr + lc4, mask=b4, other=0.0).to(tl.float32)
        old_rv4 = tl.load(rv4_ptr + lc4, mask=b4, other=0.0).to(tl.float32)
        old_rm5 = tl.load(rm5_ptr + lc5, mask=b5, other=0.0).to(tl.float32)
        old_rv5 = tl.load(rv5_ptr + lc5, mask=b5, other=0.0).to(tl.float32)

        new_rm0 = old_rm0 * (1.0 - momentum) + mean * momentum
        new_rv0 = old_rv0 * (1.0 - momentum) + var * running_var_correction * momentum
        new_rm1 = old_rm1 * (1.0 - momentum) + mean * momentum
        new_rv1 = old_rv1 * (1.0 - momentum) + var * running_var_correction * momentum
        new_rm2 = old_rm2 * (1.0 - momentum) + mean * momentum
        new_rv2 = old_rv2 * (1.0 - momentum) + var * running_var_correction * momentum
        new_rm3 = old_rm3 * (1.0 - momentum) + mean * momentum
        new_rv3 = old_rv3 * (1.0 - momentum) + var * running_var_correction * momentum
        new_rm4 = old_rm4 * (1.0 - momentum) + mean * momentum
        new_rv4 = old_rv4 * (1.0 - momentum) + var * running_var_correction * momentum
        new_rm5 = old_rm5 * (1.0 - momentum) + mean * momentum
        new_rv5 = old_rv5 * (1.0 - momentum) + var * running_var_correction * momentum

        tl.store(rm0_ptr + lc0, new_rm0, mask=b0)
        tl.store(rv0_ptr + lc0, new_rv0, mask=b0)
        tl.store(rm1_ptr + lc1, new_rm1, mask=b1)
        tl.store(rv1_ptr + lc1, new_rv1, mask=b1)
        tl.store(rm2_ptr + lc2, new_rm2, mask=b2)
        tl.store(rv2_ptr + lc2, new_rv2, mask=b2)
        tl.store(rm3_ptr + lc3, new_rm3, mask=b3)
        tl.store(rv3_ptr + lc3, new_rv3, mask=b3)
        tl.store(rm4_ptr + lc4, new_rm4, mask=b4)
        tl.store(rv4_ptr + lc4, new_rv4, mask=b4)
        tl.store(rm5_ptr + lc5, new_rm5, mask=b5)
        tl.store(rv5_ptr + lc5, new_rv5, mask=b5)

    @triton.jit
    def _spatial_mean_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
        x4_ptr,
        x5_ptr,
        w0_ptr,
        b0_ptr,
        w1_ptr,
        b1_ptr,
        w2_ptr,
        b2_ptr,
        w3_ptr,
        b3_ptr,
        w4_ptr,
        b4_ptr,
        w5_ptr,
        b5_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        out_channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        n_idx = tl.program_id(1)
        channels = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        hw = tl.arange(0, BLOCK_HW)
        channel_2d = channels[:, None]
        hw_2d = hw[None, :]
        mask = (channels[:, None] < out_channels) & (hw_2d < hw_size)

        vals, rb0, rb1, rb2, rb3, rb4, rb5, lc0, lc1, lc2, lc3, lc4, lc5 = _load_branch_values(
            x0_ptr,
            x1_ptr,
            x2_ptr,
            x3_ptr,
            x4_ptr,
            x5_ptr,
            channel_2d,
            n_idx,
            hw_2d,
            mask,
            c0=320,
            c1=384,
            c2=384,
            c3=384,
            c4=384,
            c5=192,
            off0=0,
            off1=320,
            off2=704,
            off3=1088,
            off4=1472,
            off5=1856,
            hw_size=64,
        )

        channel_mask = channels < out_channels
        mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)[:, None]
        invstd = tl.load(invstd_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)[:, None]
        rb0_1d = (channels >= 0) & (channels < 320)
        rb1_1d = (channels >= 320) & (channels < 704)
        rb2_1d = (channels >= 704) & (channels < 1088)
        rb3_1d = (channels >= 1088) & (channels < 1472)
        rb4_1d = (channels >= 1472) & (channels < 1856)
        rb5_1d = (channels >= 1856) & (channels < 2048)
        lc0_1d = tl.where(rb0_1d, channels, 0)
        lc1_1d = tl.where(rb1_1d, channels - 320, 0)
        lc2_1d = tl.where(rb2_1d, channels - 704, 0)
        lc3_1d = tl.where(rb3_1d, channels - 1088, 0)
        lc4_1d = tl.where(rb4_1d, channels - 1472, 0)
        lc5_1d = tl.where(rb5_1d, channels - 1856, 0)

        weight = tl.load(w0_ptr + lc0_1d, mask=channel_mask & rb0_1d, other=0.0).to(tl.float32)
        weight += tl.load(w1_ptr + lc1_1d, mask=channel_mask & rb1_1d, other=0.0).to(tl.float32)
        weight += tl.load(w2_ptr + lc2_1d, mask=channel_mask & rb2_1d, other=0.0).to(tl.float32)
        weight += tl.load(w3_ptr + lc3_1d, mask=channel_mask & rb3_1d, other=0.0).to(tl.float32)
        weight += tl.load(w4_ptr + lc4_1d, mask=channel_mask & rb4_1d, other=0.0).to(tl.float32)
        weight += tl.load(w5_ptr + lc5_1d, mask=channel_mask & rb5_1d, other=0.0).to(tl.float32)
        bias = tl.load(b0_ptr + lc0_1d, mask=channel_mask & rb0_1d, other=0.0).to(tl.float32)
        bias += tl.load(b1_ptr + lc1_1d, mask=channel_mask & rb1_1d, other=0.0).to(tl.float32)
        bias += tl.load(b2_ptr + lc2_1d, mask=channel_mask & rb2_1d, other=0.0).to(tl.float32)
        bias += tl.load(b3_ptr + lc3_1d, mask=channel_mask & rb3_1d, other=0.0).to(tl.float32)
        bias += tl.load(b4_ptr + lc4_1d, mask=channel_mask & rb4_1d, other=0.0).to(tl.float32)
        bias += tl.load(b5_ptr + lc5_1d, mask=channel_mask & rb5_1d, other=0.0).to(tl.float32)

        y = (vals - mean) * invstd
        y = y * weight[:, None] + bias[:, None]
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        reduced = tl.sum(y, axis=1) / hw_size
        tl.store(out_ptr + n_idx * out_channels + channels, reduced, mask=channel_mask)


def _expect_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    if len(inputs) != 31:
        raise ValueError(f"{REPRO_ID} expects 31 inputs, got {len(inputs)}")

    tensors: list[Any] = list(inputs)
    x_names = (
        "convolution_85",
        "convolution_87",
        "convolution_88",
        "convolution_91",
        "convolution_92",
        "convolution_93",
    )
    for branch, (index, channels, name) in enumerate(zip((0, 5, 10, 15, 20, 25), BRANCH_CHANNELS, x_names)):
        _expect_f32_tensor(name, tensors[index], (N, channels, HEIGHT, WIDTH), (channels * HW, HW, WIDTH, 1))
        for offset, suffix in enumerate(("running_mean", "running_var", "weight", "bias"), start=1):
            _expect_f32_tensor(
                f"branch{branch}_{suffix}",
                tensors[index + offset],
                (channels,),
                (1,),
            )

    if tuple(tensors[30]) != (N, OUT_CHANNELS):
        raise ValueError(f"unexpected output shape parameter: {tensors[30]!r}")

    device = tensors[0].device
    for item in tensors[:30]:
        if isinstance(item, torch.Tensor) and item.device != device:
            raise ValueError("all tensor inputs must be on the same device")
    return tuple(tensors)


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    (
        x0,
        rm0,
        rv0,
        w0,
        b0,
        x1,
        rm1,
        rv1,
        w1,
        b1,
        x2,
        rm2,
        rv2,
        w2,
        b2,
        x3,
        rm3,
        rv3,
        w3,
        b3,
        x4,
        rm4,
        rv4,
        w4,
        b4,
        x5,
        rm5,
        rv5,
        w5,
        b5,
        _shape,
    ) = inputs
    outputs = []
    updated = []
    for x, running_mean, running_var, weight, bias in (
        (x0, rm0, rv0, w0, b0),
        (x1, rm1, rv1, w1, b1),
        (x2, rm2, rv2, w2, b2),
        (x3, rm3, rv3, w3, b3),
        (x4, rm4, rv4, w4, b4),
        (x5, rm5, rv5, w5, b5),
    ):
        var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
        invstd = torch.rsqrt(var + EPS)
        y = torch.relu((x - mean) * invstd * weight[None, :, None, None] + bias[None, :, None, None])
        outputs.append(torch.mean(y, dim=(-1, -2), keepdim=False))
        mean_1d = mean.squeeze((0, 2, 3))
        var_1d = var.squeeze((0, 2, 3))
        running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
        running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
        updated.extend([running_mean, running_var])
    return (torch.cat(outputs, dim=1), *updated)


@oracle_impl(hardware="H100", shapes="(T([128, 320, 8, 8], f32), T([320], f32), T([320], f32), T([320], f32), T([320], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 8, 8], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 192, 8, 8], f32), T([192], f32), T([192], f32), T([192], f32), T([192], f32), S([128, 2048]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation."""
    tensors = _validate_inputs(inputs)
    if triton is None or not tensors[0].is_cuda:
        return _torch_reference(inputs)

    (
        x0,
        rm0,
        rv0,
        w0,
        b0,
        x1,
        rm1,
        rv1,
        w1,
        b1,
        x2,
        rm2,
        rv2,
        w2,
        b2,
        x3,
        rm3,
        rv3,
        w3,
        b3,
        x4,
        rm4,
        rv4,
        w4,
        b4,
        x5,
        rm5,
        rv5,
        w5,
        b5,
        _shape,
    ) = tensors

    mean = torch.empty_strided((OUT_CHANNELS,), (1,), device=x0.device, dtype=torch.float32)
    invstd = torch.empty_strided((OUT_CHANNELS,), (1,), device=x0.device, dtype=torch.float32)
    out = torch.empty_strided((N, OUT_CHANNELS), (OUT_CHANNELS, 1), device=x0.device, dtype=torch.float32)

    _stats_kernel[(OUT_CHANNELS,)](
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        rm0,
        rv0,
        rm1,
        rv1,
        rm2,
        rv2,
        rm3,
        rv3,
        rm4,
        rv4,
        rm5,
        rv5,
        mean,
        invstd,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        hw_size=HW,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_K=STAT_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _spatial_mean_kernel[(triton.cdiv(OUT_CHANNELS, OUT_BLOCK_C), N)](
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        w0,
        b0,
        w1,
        b1,
        w2,
        b2,
        w3,
        b3,
        w4,
        b4,
        w5,
        b5,
        mean,
        invstd,
        out,
        out_channels=OUT_CHANNELS,
        hw_size=HW,
        BLOCK_C=OUT_BLOCK_C,
        BLOCK_HW=HW,
        num_warps=8,
        num_stages=3,
    )
    return out, rm0, rv0, rm1, rv1, rm2, rv2, rm3, rv3, rm4, rv4, rm5, rv5


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        else:
            cloned.append(item)
    return tuple(cloned)


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    eager_inputs = _clone_inputs(inputs)
    oracle_inputs = _clone_inputs(inputs)
    with torch.no_grad():
        eager = instance(*eager_inputs)
        oracle_out = oracle_forward(oracle_inputs)
        if isinstance(oracle_inputs[0], torch.Tensor) and oracle_inputs[0].is_cuda:
            torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (eager_tensor, oracle_tensor) in enumerate(zip(eager_list, oracle_list)):
        shape_ok = eager_tensor.shape == oracle_tensor.shape
        dtype_ok = eager_tensor.dtype == oracle_tensor.dtype
        stride_ok = eager_tensor.stride() == oracle_tensor.stride()
        if not shape_ok or not dtype_ok or not stride_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"shape oracle={list(oracle_tensor.shape)} eager={list(eager_tensor.shape)} "
                f"dtype oracle={oracle_tensor.dtype} eager={eager_tensor.dtype} "
                f"stride oracle={oracle_tensor.stride()} eager={eager_tensor.stride()}"
            )
            all_pass = False
            continue

        if eager_tensor.is_floating_point():
            eager_f32 = eager_tensor.float()
            oracle_f32 = oracle_tensor.float()
            max_diff = (eager_f32 - oracle_f32).abs().max().item()
            values_ok = torch.allclose(eager_f32, oracle_f32, atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
                f"stride={eager_tensor.stride()} max_diff={max_diff:.2e})"
            )
        else:
            values_ok = torch.equal(eager_tensor, oracle_tensor)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(exact, shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
                f"stride={eager_tensor.stride()})"
            )
        all_pass = all_pass and bool(values_ok)

    alias_ok = (
        isinstance(oracle_out, tuple)
        and len(oracle_out) == 13
        and oracle_out[1] is oracle_inputs[1]
        and oracle_out[2] is oracle_inputs[2]
        and oracle_out[3] is oracle_inputs[6]
        and oracle_out[4] is oracle_inputs[7]
        and oracle_out[5] is oracle_inputs[11]
        and oracle_out[6] is oracle_inputs[12]
        and oracle_out[7] is oracle_inputs[16]
        and oracle_out[8] is oracle_inputs[17]
        and oracle_out[9] is oracle_inputs[21]
        and oracle_out[10] is oracle_inputs[22]
        and oracle_out[11] is oracle_inputs[26]
        and oracle_out[12] is oracle_inputs[27]
    )
    print(f"  running-stat aliases: {'PASS' if alias_ok else 'FAIL'}")
    return all_pass and alias_ok


# --- CLI entry point ---
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

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _run_check(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
