"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet channel-shuffle/cat/permute scope and both sibling batch-norm-backward-style branches by deriving the shuffled even/odd 58-channel slices directly, sharing the two branch reductions in one Triton reduction plan, and emitting all four returned outputs from the finalized channel summaries, whereas Inductor currently lowers the cat/view/permute/clone layout work and the two large BN-backward reduction/epilogue branches as generic scheduled regions with avoidable intermediate traffic; Inductor cannot do this today because its scheduler/codegen does not represent this fixed channel shuffle as a virtual producer feeding a full-scope multi-output reduction template; the fix is SCHEDULER_FUSION: add a guarded scheduler fusion that keeps ShuffleNet channel-shuffle cats virtual and sinks the paired BN-backward reductions plus tensor/vector epilogues into one full-scope plan."""
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
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


N = 512
C = 58
VIEW_C = 116
H = 28
W = 28
HW = H * W
NHW = N * HW
INV_NHW = 2.4912308673469386e-06
SHAPE_4D = (N, C, H, W)
STRIDE_4D = (C * HW, HW, W, 1)
SHAPE_VIEW = (N, VIEW_C, H, W)
STRIDE_VIEW = (VIEW_C * HW, HW, W, 1)
SHAPE_MEAN = (1, C, 1, 1)
STRIDE_MEAN = (C, 1, 1, 1)
SHAPE_VEC = (C,)
STRIDE_VEC = (1,)
SHAPE_SHUFFLE = (N, C, 2, H, W)
SHAPE_FINAL = (N, VIEW_C, H, W)
REDUCE_BLOCK = 1024
EPILOGUE_BLOCK = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _dual_branch_partial_kernel(
        view_ptr,
        getitem_ptr,
        grad_a_ptr,
        mean_a_ptr,
        invstd_a_ptr,
        weight_a_ptr,
        bias_a_ptr,
        grad_b_ptr,
        mean_b_ptr,
        invstd_b_ptr,
        weight_b_ptr,
        bias_b_ptr,
        full_ptr,
        partials_ptr,
        C_: tl.constexpr,
        VIEW_C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < NHW_
        n = k // HW_
        hw = k - n * HW_

        base_branch = n * C_ * HW_ + c * HW_ + hw
        mean_a = tl.load(mean_a_ptr + c).to(tl.float32)
        invstd_a = tl.load(invstd_a_ptr + c).to(tl.float32)
        weight_a = tl.load(weight_a_ptr + c).to(tl.float32)
        bias_a = tl.load(bias_a_ptr + c).to(tl.float32)
        mean_b = tl.load(mean_b_ptr + c).to(tl.float32)
        invstd_b = tl.load(invstd_b_ptr + c).to(tl.float32)
        weight_b = tl.load(weight_b_ptr + c).to(tl.float32)
        bias_b = tl.load(bias_b_ptr + c).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)

        odd_channel = c * 2 + 1
        odd_from_view = odd_channel < C_
        odd_getitem_channel = tl.where(odd_from_view, 0, odd_channel - C_)
        odd_view_offsets = n * VIEW_C_ * HW_ + odd_channel * HW_ + hw
        odd_getitem_offsets = n * C_ * HW_ + odd_getitem_channel * HW_ + hw
        odd_view = tl.load(view_ptr + odd_view_offsets, mask=active & odd_from_view, other=0.0).to(tl.float32)
        odd_getitem = tl.load(
            getitem_ptr + odd_getitem_offsets,
            mask=active & (odd_from_view == 0),
            other=0.0,
        ).to(tl.float32)
        shuffled_odd = tl.where(odd_from_view, odd_view, odd_getitem)

        even_channel = c * 2
        even_from_view = even_channel < C_
        even_getitem_channel = tl.where(even_from_view, 0, even_channel - C_)
        even_view_offsets = n * VIEW_C_ * HW_ + even_channel * HW_ + hw
        even_getitem_offsets = n * C_ * HW_ + even_getitem_channel * HW_ + hw
        even_view = tl.load(view_ptr + even_view_offsets, mask=active & even_from_view, other=0.0).to(tl.float32)
        even_getitem = tl.load(
            getitem_ptr + even_getitem_offsets,
            mask=active & (even_from_view == 0),
            other=0.0,
        ).to(tl.float32)
        shuffled_even = tl.where(even_from_view, even_view, even_getitem)

        grad_a = tl.load(grad_a_ptr + base_branch, mask=active, other=0.0).to(tl.float32)
        centered_a = grad_a - mean_a
        affine_a = centered_a * invstd_a
        affine_a = affine_a * weight_a + bias_a
        where_a = tl.where(affine_a <= 0.0, full_value, shuffled_odd)
        where_a = tl.where(active, where_a, 0.0)
        centered_a = tl.where(active, centered_a, 0.0)

        grad_b = tl.load(grad_b_ptr + base_branch, mask=active, other=0.0).to(tl.float32)
        centered_b = grad_b - mean_b
        affine_b = centered_b * invstd_b
        affine_b = affine_b * weight_b + bias_b
        where_b = tl.where(affine_b <= 0.0, full_value, shuffled_even)
        where_b = tl.where(active, where_b, 0.0)
        centered_b = tl.where(active, centered_b, 0.0)

        partial_offset = c * NUM_TILES_ + tile
        plane = C_ * NUM_TILES_
        tl.store(partials_ptr + partial_offset, tl.sum(where_a, axis=0))
        tl.store(partials_ptr + plane + partial_offset, tl.sum(where_a * centered_a, axis=0))
        tl.store(partials_ptr + 2 * plane + partial_offset, tl.sum(where_b, axis=0))
        tl.store(partials_ptr + 3 * plane + partial_offset, tl.sum(where_b * centered_b, axis=0))

    @triton.jit
    def _finalize_stats_kernel(
        partials_ptr,
        invstd_a_ptr,
        invstd_b_ptr,
        stats_ptr,
        out_vec_a_ptr,
        out_vec_b_ptr,
        C_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        INV_NHW_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles < NUM_TILES_
        offsets = c * NUM_TILES_ + tiles
        plane = C_ * NUM_TILES_

        sum_a = tl.sum(tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        dot_a = tl.sum(tl.load(partials_ptr + plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum_b = tl.sum(tl.load(partials_ptr + 2 * plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        dot_b = tl.sum(tl.load(partials_ptr + 3 * plane + offsets, mask=mask, other=0.0).to(tl.float32), axis=0)

        invstd_a = tl.load(invstd_a_ptr + c).to(tl.float32)
        invstd_b = tl.load(invstd_b_ptr + c).to(tl.float32)
        stats_plane = C_
        tl.store(stats_ptr + c, sum_a * INV_NHW_)
        tl.store(stats_ptr + stats_plane + c, (dot_a * INV_NHW_) * (invstd_a * invstd_a))
        tl.store(stats_ptr + 2 * stats_plane + c, sum_b * INV_NHW_)
        tl.store(stats_ptr + 3 * stats_plane + c, (dot_b * INV_NHW_) * (invstd_b * invstd_b))
        tl.store(out_vec_a_ptr + c, dot_a * invstd_a)
        tl.store(out_vec_b_ptr + c, dot_b * invstd_b)

    @triton.jit
    def _dual_branch_epilogue_kernel(
        view_ptr,
        getitem_ptr,
        grad_a_ptr,
        mean_a_ptr,
        invstd_a_ptr,
        weight_a_ptr,
        bias_a_ptr,
        grad_b_ptr,
        mean_b_ptr,
        invstd_b_ptr,
        weight_b_ptr,
        bias_b_ptr,
        full_ptr,
        stats_ptr,
        out_a_ptr,
        out_b_ptr,
        C_: tl.constexpr,
        VIEW_C_: tl.constexpr,
        HW_: tl.constexpr,
        NHW_: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < NHW_
        n = k // HW_
        hw = k - n * HW_

        base_branch = n * C_ * HW_ + c * HW_ + hw
        full_value = tl.load(full_ptr).to(tl.float32)

        odd_channel = c * 2 + 1
        odd_from_view = odd_channel < C_
        odd_getitem_channel = tl.where(odd_from_view, 0, odd_channel - C_)
        odd_view_offsets = n * VIEW_C_ * HW_ + odd_channel * HW_ + hw
        odd_getitem_offsets = n * C_ * HW_ + odd_getitem_channel * HW_ + hw
        odd_view = tl.load(view_ptr + odd_view_offsets, mask=active & odd_from_view, other=0.0).to(tl.float32)
        odd_getitem = tl.load(
            getitem_ptr + odd_getitem_offsets,
            mask=active & (odd_from_view == 0),
            other=0.0,
        ).to(tl.float32)
        shuffled_odd = tl.where(odd_from_view, odd_view, odd_getitem)

        even_channel = c * 2
        even_from_view = even_channel < C_
        even_getitem_channel = tl.where(even_from_view, 0, even_channel - C_)
        even_view_offsets = n * VIEW_C_ * HW_ + even_channel * HW_ + hw
        even_getitem_offsets = n * C_ * HW_ + even_getitem_channel * HW_ + hw
        even_view = tl.load(view_ptr + even_view_offsets, mask=active & even_from_view, other=0.0).to(tl.float32)
        even_getitem = tl.load(
            getitem_ptr + even_getitem_offsets,
            mask=active & (even_from_view == 0),
            other=0.0,
        ).to(tl.float32)
        shuffled_even = tl.where(even_from_view, even_view, even_getitem)

        mean_a = tl.load(mean_a_ptr + c).to(tl.float32)
        invstd_a = tl.load(invstd_a_ptr + c).to(tl.float32)
        weight_a = tl.load(weight_a_ptr + c).to(tl.float32)
        bias_a = tl.load(bias_a_ptr + c).to(tl.float32)
        grad_a = tl.load(grad_a_ptr + base_branch, mask=active, other=0.0).to(tl.float32)
        centered_a = grad_a - mean_a
        affine_a = centered_a * invstd_a
        affine_a = affine_a * weight_a + bias_a
        where_a = tl.where(affine_a <= 0.0, full_value, shuffled_odd)
        mean_term_a = tl.load(stats_ptr + c).to(tl.float32)
        coeff_a = tl.load(stats_ptr + C_ + c).to(tl.float32)
        scale_a = invstd_a * weight_a
        out_a = ((where_a - centered_a * coeff_a) - mean_term_a) * scale_a

        mean_b = tl.load(mean_b_ptr + c).to(tl.float32)
        invstd_b = tl.load(invstd_b_ptr + c).to(tl.float32)
        weight_b = tl.load(weight_b_ptr + c).to(tl.float32)
        bias_b = tl.load(bias_b_ptr + c).to(tl.float32)
        grad_b = tl.load(grad_b_ptr + base_branch, mask=active, other=0.0).to(tl.float32)
        centered_b = grad_b - mean_b
        affine_b = centered_b * invstd_b
        affine_b = affine_b * weight_b + bias_b
        where_b = tl.where(affine_b <= 0.0, full_value, shuffled_even)
        mean_term_b = tl.load(stats_ptr + 2 * C_ + c).to(tl.float32)
        coeff_b = tl.load(stats_ptr + 3 * C_ + c).to(tl.float32)
        scale_b = invstd_b * weight_b
        out_b = ((where_b - centered_b * coeff_b) - mean_term_b) * scale_b

        tl.store(out_a_ptr + base_branch, out_a, mask=active)
        tl.store(out_b_ptr + base_branch, out_b, mask=active)


def _oracle_forward_torch(inputs: list[Any] | tuple[Any, ...]):
    (
        view_30,
        getitem_147,
        arg152_1,
        arg153_1,
        arg154_1,
        arg14_1,
        arg15_1,
        full,
        arg143_1,
        arg144_1,
        arg145_1,
        arg7_1,
        arg8_1,
        shape_param_0,
        shape_param_1,
    ) = inputs

    shuffled = (
        torch.cat([view_30[:, :C, :, :], getitem_147], 1)
        .view(shape_param_0)
        .permute(0, 2, 1, 3, 4)
        .clone(memory_format=torch.contiguous_format)
        .view(shape_param_1)
    )
    slice_even = shuffled[:, :C, :, :]
    slice_odd = shuffled[:, C:VIEW_C, :, :]

    mean_a = arg153_1.squeeze((0, 2, 3))[None, :, None, None]
    invstd_a = arg154_1.squeeze((0, 2, 3))[None, :, None, None]
    weight_a = arg14_1[None, :, None, None]
    bias_a = arg15_1[None, :, None, None]
    centered_a = arg152_1 - mean_a
    affine_a = centered_a * invstd_a
    affine_a = affine_a * weight_a + bias_a
    where_a = torch.where(torch.relu(affine_a) <= 0, full, slice_odd)
    sum_a = where_a.sum([0, 2, 3])
    dot_a = (where_a * centered_a).sum([0, 2, 3])
    out_a = ((where_a - centered_a * ((dot_a * INV_NHW) * arg154_1.squeeze((0, 2, 3)).square())[None, :, None, None]) - (sum_a * INV_NHW)[None, :, None, None])
    out_a = out_a * (arg154_1.squeeze((0, 2, 3)) * arg14_1)[None, :, None, None]
    out_vec_a = dot_a * arg154_1.squeeze((0, 2, 3))

    mean_b = arg144_1.squeeze((0, 2, 3))[None, :, None, None]
    invstd_b = arg145_1.squeeze((0, 2, 3))[None, :, None, None]
    weight_b = arg7_1[None, :, None, None]
    bias_b = arg8_1[None, :, None, None]
    centered_b = arg143_1 - mean_b
    affine_b = centered_b * invstd_b
    affine_b = affine_b * weight_b + bias_b
    where_b = torch.where(torch.relu(affine_b) <= 0, full, slice_even)
    sum_b = where_b.sum([0, 2, 3])
    dot_b = (where_b * centered_b).sum([0, 2, 3])
    out_b = ((where_b - centered_b * ((dot_b * INV_NHW) * arg145_1.squeeze((0, 2, 3)).square())[None, :, None, None]) - (sum_b * INV_NHW)[None, :, None, None])
    out_b = out_b * (arg145_1.squeeze((0, 2, 3)) * arg7_1)[None, :, None, None]
    out_vec_b = dot_b * arg145_1.squeeze((0, 2, 3))
    return out_a, out_vec_a, out_b, out_vec_b


def _require_f32_tensor(name: str, value: Any, shape: tuple[int, ...], stride: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _require_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    try:
        shape = tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc
    if shape != expected:
        raise ValueError(f"{name} is {shape}, expected {expected}")


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 15:
        raise ValueError(f"{REPRO_ID} expects 15 inputs, got {len(inputs)}")

    (
        view_30,
        getitem_147,
        arg152_1,
        arg153_1,
        arg154_1,
        arg14_1,
        arg15_1,
        full,
        arg143_1,
        arg144_1,
        arg145_1,
        arg7_1,
        arg8_1,
        shape_param_0,
        shape_param_1,
    ) = inputs

    view_30 = _require_f32_tensor("view_30", view_30, SHAPE_VIEW, STRIDE_VIEW)
    getitem_147 = _require_f32_tensor("getitem_147", getitem_147, SHAPE_4D, STRIDE_4D)
    arg152_1 = _require_f32_tensor("arg152_1", arg152_1, SHAPE_4D, STRIDE_4D)
    arg153_1 = _require_f32_tensor("arg153_1", arg153_1, SHAPE_MEAN, STRIDE_MEAN)
    arg154_1 = _require_f32_tensor("arg154_1", arg154_1, SHAPE_MEAN, STRIDE_MEAN)
    arg14_1 = _require_f32_tensor("arg14_1", arg14_1, SHAPE_VEC, STRIDE_VEC)
    arg15_1 = _require_f32_tensor("arg15_1", arg15_1, SHAPE_VEC, STRIDE_VEC)
    full = _require_f32_tensor("full", full, (), ())
    arg143_1 = _require_f32_tensor("arg143_1", arg143_1, SHAPE_4D, STRIDE_4D)
    arg144_1 = _require_f32_tensor("arg144_1", arg144_1, SHAPE_MEAN, STRIDE_MEAN)
    arg145_1 = _require_f32_tensor("arg145_1", arg145_1, SHAPE_MEAN, STRIDE_MEAN)
    arg7_1 = _require_f32_tensor("arg7_1", arg7_1, SHAPE_VEC, STRIDE_VEC)
    arg8_1 = _require_f32_tensor("arg8_1", arg8_1, SHAPE_VEC, STRIDE_VEC)
    _require_shape("_shape_param_0", shape_param_0, SHAPE_SHUFFLE)
    _require_shape("_shape_param_1", shape_param_1, SHAPE_FINAL)

    device = view_30.device
    tensors = (
        getitem_147,
        arg152_1,
        arg153_1,
        arg154_1,
        arg14_1,
        arg15_1,
        full,
        arg143_1,
        arg144_1,
        arg145_1,
        arg7_1,
        arg8_1,
    )
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")

    return (
        view_30,
        getitem_147,
        arg152_1,
        arg153_1,
        arg154_1,
        arg14_1,
        arg15_1,
        full,
        arg143_1,
        arg144_1,
        arg145_1,
        arg7_1,
        arg8_1,
    )


@oracle_impl(hardware="H100", shapes="(T([512, 116, 28, 28], f32), T([512, 58, 28, 28], f32), T([512, 58, 28, 28], f32), T([1, 58, 1, 1], f32), T([1, 58, 1, 1], f32), T([58], f32), T([58], f32), T([], f32), T([512, 58, 28, 28], f32), T([1, 58, 1, 1], f32), T([1, 58, 1, 1], f32), T([58], f32), T([58], f32), S([512, 58, 2, 28, 28]), S([512, 116, 28, 28]))")
def oracle_forward(inputs):
    """Run the exact Repro.forward scope with virtual channel shuffle and paired BN-backward branches."""
    if not isinstance(inputs, tuple):
        inputs = tuple(inputs)
    if not isinstance(inputs[0], torch.Tensor) or not inputs[0].is_cuda:
        return _oracle_forward_torch(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    (
        view_30,
        getitem_147,
        arg152_1,
        arg153_1,
        arg154_1,
        arg14_1,
        arg15_1,
        full,
        arg143_1,
        arg144_1,
        arg145_1,
        arg7_1,
        arg8_1,
    ) = _validate_inputs(inputs)

    device = view_30.device
    num_tiles = triton.cdiv(NHW, REDUCE_BLOCK)
    block_tiles = triton.next_power_of_2(num_tiles)
    partials = torch.empty_strided((4, C, num_tiles), (C * num_tiles, num_tiles, 1), device=device, dtype=torch.float32)
    stats = torch.empty_strided((4, C), (C, 1), device=device, dtype=torch.float32)
    out_a = torch.empty_strided(SHAPE_4D, STRIDE_4D, device=device, dtype=torch.float32)
    out_vec_a = torch.empty_strided(SHAPE_VEC, STRIDE_VEC, device=device, dtype=torch.float32)
    out_b = torch.empty_strided(SHAPE_4D, STRIDE_4D, device=device, dtype=torch.float32)
    out_vec_b = torch.empty_strided(SHAPE_VEC, STRIDE_VEC, device=device, dtype=torch.float32)

    _dual_branch_partial_kernel[(C, num_tiles)](
        view_30,
        getitem_147,
        arg152_1,
        arg153_1,
        arg154_1,
        arg14_1,
        arg15_1,
        arg143_1,
        arg144_1,
        arg145_1,
        arg7_1,
        arg8_1,
        full,
        partials,
        C_=C,
        VIEW_C_=VIEW_C,
        HW_=HW,
        NHW_=NHW,
        NUM_TILES_=num_tiles,
        BLOCK=REDUCE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _finalize_stats_kernel[(C,)](
        partials,
        arg154_1,
        arg145_1,
        stats,
        out_vec_a,
        out_vec_b,
        C_=C,
        NUM_TILES_=num_tiles,
        BLOCK_TILES=block_tiles,
        INV_NHW_=INV_NHW,
        num_warps=8,
        num_stages=4,
    )
    _dual_branch_epilogue_kernel[(C, triton.cdiv(NHW, EPILOGUE_BLOCK))](
        view_30,
        getitem_147,
        arg152_1,
        arg153_1,
        arg154_1,
        arg14_1,
        arg15_1,
        arg143_1,
        arg144_1,
        arg145_1,
        arg7_1,
        arg8_1,
        full,
        stats,
        out_a,
        out_b,
        C_=C,
        VIEW_C_=VIEW_C,
        HW_=HW,
        NHW_=NHW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return out_a, out_vec_a, out_b, out_vec_b


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
