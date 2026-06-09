"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Repro.forward return by fusing `getitem_394 + getitem_397` with the low-memory max-pool-backward scatter-add from `add_493[:, 480:768]`/`getitem_65` into one channels-last source, then feeding the four sibling BN/ReLU-backward channel groups to emit all eight tensor and channel-reduction outputs with shared reduction summaries, whereas Inductor currently materializes the decomposed scatter-add/view/channels-last clone/add chain and schedules the four masked BN-backward reductions and tensor epilogues as separate generic kernels; Inductor cannot do this today because its scheduler/codegen cannot represent a structured maxpool scatter-add producer as a fused source for multiple sibling channel reductions plus their dependent per-element epilogues; the fix is SCATTER_REDUCE: add guarded scatter-add producer fusion that sinks low-memory max-pool backward scatter construction into downstream channel reductions and shares the produced channel summaries with the tensor epilogue kernels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    has_stochastic_ops,
)


N = 128
ADD_C = 768
CTOTAL = 288
LOW_C = 64
MID_C = 64
WIDE_C = 96
HIGH_C = 64
POOL_H = 17
POOL_W = 17
POOL_HW = POOL_H * POOL_W
H = 35
W = 35
HW = H * W
SPATIAL = N * HW
BN_SCALE = 6.3775510204081635e-06

DEFAULT_BLOCK_C = 64
DEFAULT_BLOCK_K = 256
DEFAULT_FINAL_BLOCK_C = 8
DEFAULT_EPILOGUE_BLOCK_K = 128
SOURCE_BLOCK = 1024
SCATTER_BLOCK = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> None:
    if len(inputs) != 28:
        raise ValueError(f"expected 28 repro inputs, got {len(inputs)}")
    tensors = inputs[:25]
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("first 25 repro inputs must be tensors")
    if any(x.device.type != "cuda" for x in tensors):
        raise RuntimeError("oracle expects CUDA tensors from make_inputs()")

    (
        add_493,
        getitem_65,
        getitem_394,
        getitem_397,
        convolution_25,
        getitem_55,
        rsqrt_25,
        primals_156,
        primals_157,
        full_default,
        convolution_24,
        getitem_53,
        rsqrt_24,
        primals_150,
        primals_151,
        convolution_21,
        getitem_47,
        rsqrt_21,
        primals_132,
        primals_133,
        convolution_19,
        getitem_43,
        rsqrt_19,
        primals_120,
        primals_121,
        shape0,
        shape1,
        shape2,
    ) = inputs

    expected = {
        "add_493": ((N, ADD_C, POOL_H, POOL_W), torch.float32, (221952, 1, 13056, ADD_C)),
        "getitem_65": ((N, CTOTAL, POOL_H, POOL_W), torch.int8, (83232, 1, 4896, CTOTAL)),
        "getitem_394": ((N, CTOTAL, H, W), torch.float32, (352800, 1, 10080, CTOTAL)),
        "getitem_397": ((N, CTOTAL, H, W), torch.float32, (352800, 1, 10080, CTOTAL)),
        "convolution_25": ((N, HIGH_C, H, W), torch.float32, (78400, 1, 2240, HIGH_C)),
        "getitem_55": ((1, HIGH_C, 1, 1), torch.float32, (HIGH_C, 1, 1, 1)),
        "rsqrt_25": ((1, HIGH_C, 1, 1), torch.float32, (HIGH_C, 1, 1, 1)),
        "primals_156": ((HIGH_C,), torch.float32, (1,)),
        "primals_157": ((HIGH_C,), torch.float32, (1,)),
        "full_default": ((), torch.float32, ()),
        "convolution_24": ((N, WIDE_C, H, W), torch.float32, (117600, 1, 3360, WIDE_C)),
        "getitem_53": ((1, WIDE_C, 1, 1), torch.float32, (WIDE_C, 1, 1, 1)),
        "rsqrt_24": ((1, WIDE_C, 1, 1), torch.float32, (WIDE_C, 1, 1, 1)),
        "primals_150": ((WIDE_C,), torch.float32, (1,)),
        "primals_151": ((WIDE_C,), torch.float32, (1,)),
        "convolution_21": ((N, MID_C, H, W), torch.float32, (78400, 1, 2240, MID_C)),
        "getitem_47": ((1, MID_C, 1, 1), torch.float32, (MID_C, 1, 1, 1)),
        "rsqrt_21": ((1, MID_C, 1, 1), torch.float32, (MID_C, 1, 1, 1)),
        "primals_132": ((MID_C,), torch.float32, (1,)),
        "primals_133": ((MID_C,), torch.float32, (1,)),
        "convolution_19": ((N, LOW_C, H, W), torch.float32, (78400, 1, 2240, LOW_C)),
        "getitem_43": ((1, LOW_C, 1, 1), torch.float32, (LOW_C, 1, 1, 1)),
        "rsqrt_19": ((1, LOW_C, 1, 1), torch.float32, (LOW_C, 1, 1, 1)),
        "primals_120": ((LOW_C,), torch.float32, (1,)),
        "primals_121": ((LOW_C,), torch.float32, (1,)),
    }
    values = {
        "add_493": add_493,
        "getitem_65": getitem_65,
        "getitem_394": getitem_394,
        "getitem_397": getitem_397,
        "convolution_25": convolution_25,
        "getitem_55": getitem_55,
        "rsqrt_25": rsqrt_25,
        "primals_156": primals_156,
        "primals_157": primals_157,
        "full_default": full_default,
        "convolution_24": convolution_24,
        "getitem_53": getitem_53,
        "rsqrt_24": rsqrt_24,
        "primals_150": primals_150,
        "primals_151": primals_151,
        "convolution_21": convolution_21,
        "getitem_47": getitem_47,
        "rsqrt_21": rsqrt_21,
        "primals_132": primals_132,
        "primals_133": primals_133,
        "convolution_19": convolution_19,
        "getitem_43": getitem_43,
        "rsqrt_19": rsqrt_19,
        "primals_120": primals_120,
        "primals_121": primals_121,
    }
    for name, (shape, dtype, stride) in expected.items():
        tensor = values[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} != {dtype}")
        if tuple(tensor.stride()) != stride:
            raise ValueError(f"{name} stride {tuple(tensor.stride())} != {stride}")

    if tuple(shape0) != (N * CTOTAL, POOL_HW):
        raise ValueError(f"unexpected shape parameter 0: {shape0}")
    if tuple(shape1) != (N * CTOTAL, POOL_HW):
        raise ValueError(f"unexpected shape parameter 1: {shape1}")
    if tuple(shape2) != (N, CTOTAL, H, W):
        raise ValueError(f"unexpected shape parameter 2: {shape2}")


if triton is not None:

    # Preserve the repro's separate aten mul/add rounding before the ReLU mask.
    @triton.jit
    def _f32_mul(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _f32_add(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _load_branch_activation(
        conv19_ptr,
        conv21_ptr,
        conv24_ptr,
        conv25_ptr,
        n,
        c,
        h,
        w,
        active,
    ):
        low = c
        mid = c - 64
        wide = c - 128
        high = c - 224
        low_safe = tl.where(c < 64, low, 0)
        mid_safe = tl.where((c >= 64) & (c < 128), mid, 0)
        wide_safe = tl.where((c >= 128) & (c < 224), wide, 0)
        high_safe = tl.where(c >= 224, high, 0)

        low_mask = active & (c < 64)
        mid_mask = active & (c >= 64) & (c < 128)
        wide_mask = active & (c >= 128) & (c < 224)
        high_mask = active & (c >= 224)

        return (
            tl.load(
                conv19_ptr + n * (64 * 1225) + h * (35 * 64) + w * 64 + low_safe,
                mask=low_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                conv21_ptr + n * (64 * 1225) + h * (35 * 64) + w * 64 + mid_safe,
                mask=mid_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                conv24_ptr + n * (96 * 1225) + h * (35 * 96) + w * 96 + wide_safe,
                mask=wide_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                conv25_ptr + n * (64 * 1225) + h * (35 * 64) + w * 64 + high_safe,
                mask=high_mask,
                other=0.0,
            ).to(tl.float32)
        )

    @triton.jit
    def _load_branch_vector(
        low_ptr,
        mid_ptr,
        wide_ptr,
        high_ptr,
        c,
        c_mask,
    ):
        low = c
        mid = c - 64
        wide = c - 128
        high = c - 224
        low_safe = tl.where(c < 64, low, 0)
        mid_safe = tl.where((c >= 64) & (c < 128), mid, 0)
        wide_safe = tl.where((c >= 128) & (c < 224), wide, 0)
        high_safe = tl.where(c >= 224, high, 0)

        low_mask = c_mask & (c < 64)
        mid_mask = c_mask & (c >= 64) & (c < 128)
        wide_mask = c_mask & (c >= 128) & (c < 224)
        high_mask = c_mask & (c >= 224)

        return (
            tl.load(low_ptr + low_safe, mask=low_mask, other=0.0).to(tl.float32)
            + tl.load(mid_ptr + mid_safe, mask=mid_mask, other=0.0).to(tl.float32)
            + tl.load(wide_ptr + wide_safe, mask=wide_mask, other=0.0).to(tl.float32)
            + tl.load(high_ptr + high_safe, mask=high_mask, other=0.0).to(tl.float32)
        )

    @triton.jit
    def _init_source_kernel(
        getitem394_ptr,
        getitem397_ptr,
        source_ptr,
        NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < NUMEL
        value = (
            tl.load(getitem394_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(getitem397_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        tl.store(source_ptr + offsets, value, mask=mask)

    @triton.jit
    def _scatter_add_source_kernel(
        add_ptr,
        offsets_ptr,
        source_ptr,
        NUMEL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = linear < NUMEL
        c = linear % 288
        ow = (linear // 288) % 17
        oh = (linear // (288 * 17)) % 17
        n = linear // (288 * 289)
        local = tl.load(offsets_ptr + linear, mask=mask, other=0).to(tl.int32)
        kh = local // 3
        kw = local - kh * 3
        h = oh * 2 + kh
        w = ow * 2 + kw
        add_offsets = n * (768 * 289) + oh * (17 * 768) + ow * 768 + 480 + c
        source_offsets = n * (288 * 1225) + h * (35 * 288) + w * 288 + c
        value = tl.load(add_ptr + add_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.atomic_add(source_ptr + source_offsets, value, sem="relaxed", mask=mask)

    @triton.jit
    def _partial_reduction_kernel(
        source_ptr,
        full_ptr,
        conv19_ptr,
        mean19_ptr,
        invstd19_ptr,
        gamma19_ptr,
        beta19_ptr,
        conv21_ptr,
        mean21_ptr,
        invstd21_ptr,
        gamma21_ptr,
        beta21_ptr,
        conv24_ptr,
        mean24_ptr,
        invstd24_ptr,
        gamma24_ptr,
        beta24_ptr,
        conv25_ptr,
        mean25_ptr,
        invstd25_ptr,
        gamma25_ptr,
        beta25_ptr,
        partial_sum0_ptr,
        partial_sum1_ptr,
        num_tiles: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        c_mask = c < 288
        k_mask = k < 156800
        n = k // 1225
        spatial = k - n * 1225
        h = spatial // 35
        w = spatial - h * 35
        active = k_mask[:, None] & c_mask[None, :]

        c_mat = c[None, :]
        n_mat = n[:, None]
        h_mat = h[:, None]
        w_mat = w[:, None]

        source_offsets = n_mat * (288 * 1225) + h_mat * (35 * 288) + w_mat * 288 + c_mat
        source = tl.load(source_ptr + source_offsets, mask=active, other=0.0).to(
            tl.float32
        )
        x = _load_branch_activation(
            conv19_ptr,
            conv21_ptr,
            conv24_ptr,
            conv25_ptr,
            n_mat,
            c_mat,
            h_mat,
            w_mat,
            active,
        )

        mean = _load_branch_vector(mean19_ptr, mean21_ptr, mean24_ptr, mean25_ptr, c, c_mask)
        invstd = _load_branch_vector(invstd19_ptr, invstd21_ptr, invstd24_ptr, invstd25_ptr, c, c_mask)
        gamma = _load_branch_vector(gamma19_ptr, gamma21_ptr, gamma24_ptr, gamma25_ptr, c, c_mask)
        beta = _load_branch_vector(beta19_ptr, beta21_ptr, beta24_ptr, beta25_ptr, c, c_mask)
        fill = tl.load(full_ptr).to(tl.float32)

        centered = x - mean[None, :]
        normalized = _f32_mul(centered, invstd[None, :])
        scaled = _f32_mul(normalized, gamma[None, :])
        relu_input = _f32_add(scaled, beta[None, :])
        where_value = tl.where(relu_input <= 0.0, fill, source)
        where_value = tl.where(active, where_value, 0.0)
        centered = tl.where(active, centered, 0.0)

        sum0 = tl.sum(where_value, axis=0)
        sum1 = tl.sum(where_value * centered, axis=0)
        partial_offsets = c * num_tiles + tl.program_id(1)
        tl.store(partial_sum0_ptr + partial_offsets, sum0, mask=c_mask)
        tl.store(partial_sum1_ptr + partial_offsets, sum1, mask=c_mask)

    @triton.jit
    def _finalize_reductions_kernel(
        partial_sum0_ptr,
        partial_sum1_ptr,
        invstd19_ptr,
        invstd21_ptr,
        invstd24_ptr,
        invstd25_ptr,
        sum0_ptr,
        sum1_ptr,
        out1_ptr,
        out3_ptr,
        out5_ptr,
        out7_ptr,
        num_tiles: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        tiles = tl.arange(0, BLOCK_TILES)
        c_mask = c < 288
        tile_mask = tiles < num_tiles
        offsets = tiles[:, None] + c[None, :] * num_tiles
        active = tile_mask[:, None] & c_mask[None, :]

        partial0 = tl.load(partial_sum0_ptr + offsets, mask=active, other=0.0).to(
            tl.float32
        )
        partial1 = tl.load(partial_sum1_ptr + offsets, mask=active, other=0.0).to(
            tl.float32
        )
        total0 = tl.sum(partial0, axis=0)
        total1 = tl.sum(partial1, axis=0)
        invstd = _load_branch_vector(invstd19_ptr, invstd21_ptr, invstd24_ptr, invstd25_ptr, c, c_mask)
        vec = total1 * invstd

        tl.store(sum0_ptr + c, total0, mask=c_mask)
        tl.store(sum1_ptr + c, total1, mask=c_mask)

        low = c
        mid = c - 64
        wide = c - 128
        high = c - 224
        tl.store(out7_ptr + low, vec, mask=c_mask & (c < 64))
        tl.store(out5_ptr + mid, vec, mask=c_mask & (c >= 64) & (c < 128))
        tl.store(out3_ptr + wide, vec, mask=c_mask & (c >= 128) & (c < 224))
        tl.store(out1_ptr + high, vec, mask=c_mask & (c >= 224))

    @triton.jit
    def _epilogue_kernel(
        source_ptr,
        full_ptr,
        conv19_ptr,
        mean19_ptr,
        invstd19_ptr,
        gamma19_ptr,
        beta19_ptr,
        conv21_ptr,
        mean21_ptr,
        invstd21_ptr,
        gamma21_ptr,
        beta21_ptr,
        conv24_ptr,
        mean24_ptr,
        invstd24_ptr,
        gamma24_ptr,
        beta24_ptr,
        conv25_ptr,
        mean25_ptr,
        invstd25_ptr,
        gamma25_ptr,
        beta25_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        out2_ptr,
        out4_ptr,
        out6_ptr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        c_mask = c < 288
        k_mask = k < 156800
        n = k // 1225
        spatial = k - n * 1225
        h = spatial // 35
        w = spatial - h * 35
        active = k_mask[:, None] & c_mask[None, :]

        c_mat = c[None, :]
        n_mat = n[:, None]
        h_mat = h[:, None]
        w_mat = w[:, None]

        source_offsets = n_mat * (288 * 1225) + h_mat * (35 * 288) + w_mat * 288 + c_mat
        source = tl.load(source_ptr + source_offsets, mask=active, other=0.0).to(
            tl.float32
        )
        x = _load_branch_activation(
            conv19_ptr,
            conv21_ptr,
            conv24_ptr,
            conv25_ptr,
            n_mat,
            c_mat,
            h_mat,
            w_mat,
            active,
        )

        mean = _load_branch_vector(mean19_ptr, mean21_ptr, mean24_ptr, mean25_ptr, c, c_mask)
        invstd = _load_branch_vector(invstd19_ptr, invstd21_ptr, invstd24_ptr, invstd25_ptr, c, c_mask)
        gamma = _load_branch_vector(gamma19_ptr, gamma21_ptr, gamma24_ptr, gamma25_ptr, c, c_mask)
        beta = _load_branch_vector(beta19_ptr, beta21_ptr, beta24_ptr, beta25_ptr, c, c_mask)
        total0 = tl.load(sum0_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        total1 = tl.load(sum1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        fill = tl.load(full_ptr).to(tl.float32)

        centered = x - mean[None, :]
        normalized = _f32_mul(centered, invstd[None, :])
        scaled = _f32_mul(normalized, gamma[None, :])
        relu_input = _f32_add(scaled, beta[None, :])
        where_value = tl.where(relu_input <= 0.0, fill, source)
        variance_term = (
            total1[None, :] * 6.3775510204081635e-06 * invstd[None, :] * invstd[None, :]
        )
        mean_term = total0[None, :] * 6.3775510204081635e-06
        out = (where_value - centered * variance_term - mean_term) * (
            invstd[None, :] * gamma[None, :]
        )

        low = c_mat
        mid = c_mat - 64
        wide = c_mat - 128
        high = c_mat - 224
        out_low_offsets = n_mat * (64 * 1225) + h_mat * (35 * 64) + w_mat * 64 + low
        out_mid_offsets = n_mat * (64 * 1225) + h_mat * (35 * 64) + w_mat * 64 + mid
        out_wide_offsets = n_mat * (96 * 1225) + h_mat * (35 * 96) + w_mat * 96 + wide
        out_high_offsets = n_mat * (64 * 1225) + h_mat * (35 * 64) + w_mat * 64 + high

        tl.store(out6_ptr + out_low_offsets, out, mask=active & (c_mat < 64))
        tl.store(
            out4_ptr + out_mid_offsets,
            out,
            mask=active & (c_mat >= 64) & (c_mat < 128),
        )
        tl.store(
            out2_ptr + out_wide_offsets,
            out,
            mask=active & (c_mat >= 128) & (c_mat < 224),
        )
        tl.store(out0_ptr + out_high_offsets, out, mask=active & (c_mat >= 224))


def _empty_channels_last(device: torch.device, channels: int) -> torch.Tensor:
    return torch.empty_strided(
        (N, channels, H, W),
        (channels * HW, 1, W * channels, channels),
        device=device,
        dtype=torch.float32,
    )


def _oracle_fused(
    add_493: torch.Tensor,
    getitem_65: torch.Tensor,
    getitem_394: torch.Tensor,
    getitem_397: torch.Tensor,
    convolution_25: torch.Tensor,
    getitem_55: torch.Tensor,
    rsqrt_25: torch.Tensor,
    primals_156: torch.Tensor,
    primals_157: torch.Tensor,
    full_default: torch.Tensor,
    convolution_24: torch.Tensor,
    getitem_53: torch.Tensor,
    rsqrt_24: torch.Tensor,
    primals_150: torch.Tensor,
    primals_151: torch.Tensor,
    convolution_21: torch.Tensor,
    getitem_47: torch.Tensor,
    rsqrt_21: torch.Tensor,
    primals_132: torch.Tensor,
    primals_133: torch.Tensor,
    convolution_19: torch.Tensor,
    getitem_43: torch.Tensor,
    rsqrt_19: torch.Tensor,
    primals_120: torch.Tensor,
    primals_121: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    *,
    block_c: int = DEFAULT_BLOCK_C,
    block_k: int = DEFAULT_BLOCK_K,
    final_block_c: int = DEFAULT_FINAL_BLOCK_C,
    epilogue_block_k: int = DEFAULT_EPILOGUE_BLOCK_K,
) -> tuple[torch.Tensor, ...]:
    del _shape_param_0, _shape_param_1, _shape_param_2
    _require_triton()

    device = add_493.device
    num_tiles = triton.cdiv(SPATIAL, block_k)
    source = torch.empty_strided(
        (N, CTOTAL, H, W),
        (CTOTAL * HW, 1, W * CTOTAL, CTOTAL),
        device=device,
        dtype=torch.float32,
    )
    partial_sum0 = torch.empty((CTOTAL, num_tiles), device=device, dtype=torch.float32)
    partial_sum1 = torch.empty((CTOTAL, num_tiles), device=device, dtype=torch.float32)
    sum0 = torch.empty((CTOTAL,), device=device, dtype=torch.float32)
    sum1 = torch.empty((CTOTAL,), device=device, dtype=torch.float32)

    out0 = _empty_channels_last(device, HIGH_C)
    out1 = torch.empty((HIGH_C,), device=device, dtype=torch.float32)
    out2 = _empty_channels_last(device, WIDE_C)
    out3 = torch.empty((WIDE_C,), device=device, dtype=torch.float32)
    out4 = _empty_channels_last(device, MID_C)
    out5 = torch.empty((MID_C,), device=device, dtype=torch.float32)
    out6 = _empty_channels_last(device, LOW_C)
    out7 = torch.empty((LOW_C,), device=device, dtype=torch.float32)

    _init_source_kernel[(triton.cdiv(CTOTAL * SPATIAL, SOURCE_BLOCK),)](
        getitem_394,
        getitem_397,
        source,
        NUMEL=CTOTAL * SPATIAL,
        BLOCK=SOURCE_BLOCK,
        num_warps=8,
    )
    _scatter_add_source_kernel[(triton.cdiv(N * CTOTAL * POOL_HW, SCATTER_BLOCK),)](
        add_493,
        getitem_65,
        source,
        NUMEL=N * CTOTAL * POOL_HW,
        BLOCK=SCATTER_BLOCK,
        num_warps=8,
    )
    _partial_reduction_kernel[(triton.cdiv(CTOTAL, block_c), num_tiles)](
        source,
        full_default,
        convolution_19,
        getitem_43,
        rsqrt_19,
        primals_120,
        primals_121,
        convolution_21,
        getitem_47,
        rsqrt_21,
        primals_132,
        primals_133,
        convolution_24,
        getitem_53,
        rsqrt_24,
        primals_150,
        primals_151,
        convolution_25,
        getitem_55,
        rsqrt_25,
        primals_156,
        primals_157,
        partial_sum0,
        partial_sum1,
        num_tiles=num_tiles,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=8,
    )
    _finalize_reductions_kernel[(triton.cdiv(CTOTAL, final_block_c),)](
        partial_sum0,
        partial_sum1,
        rsqrt_19,
        rsqrt_21,
        rsqrt_24,
        rsqrt_25,
        sum0,
        sum1,
        out1,
        out3,
        out5,
        out7,
        num_tiles=num_tiles,
        BLOCK_C=final_block_c,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(CTOTAL, block_c), triton.cdiv(SPATIAL, epilogue_block_k))](
        source,
        full_default,
        convolution_19,
        getitem_43,
        rsqrt_19,
        primals_120,
        primals_121,
        convolution_21,
        getitem_47,
        rsqrt_21,
        primals_132,
        primals_133,
        convolution_24,
        getitem_53,
        rsqrt_24,
        primals_150,
        primals_151,
        convolution_25,
        getitem_55,
        rsqrt_25,
        primals_156,
        primals_157,
        sum0,
        sum1,
        out0,
        out2,
        out4,
        out6,
        BLOCK_C=block_c,
        BLOCK_K=epilogue_block_k,
        num_warps=8,
    )

    return out0, out1, out2, out3, out4, out5, out6, out7


@oracle_impl(hardware="H100", shapes="(T([128, 768, 17, 17], f32, stride=(221952, 1, 13056, 768)), T([128, 288, 17, 17], i8, stride=(83232, 1, 4896, 288), gen=Index(9)), T([128, 288, 35, 35], f32, stride=(352800, 1, 10080, 288)), T([128, 288, 35, 35], f32, stride=(352800, 1, 10080, 288)), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([], f32), T([128, 96, 35, 35], f32, stride=(117600, 1, 3360, 96)), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([128, 64, 35, 35], f32, stride=(78400, 1, 2240, 64)), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), S([36864, 289]), S([36864, 289]), S([128, 288, 35, 35]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    _validate_inputs(inputs)
    return _oracle_fused(*inputs)


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
