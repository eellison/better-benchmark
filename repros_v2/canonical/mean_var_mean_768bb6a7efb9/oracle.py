"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 residual add, bf16 9x9 spatial mean, 640-channel population var_mean LayerNorm, affine epilogue, final bf16 cast, and `[128,640]` output view with a tuned spatial reduction kernel feeding a row LayerNorm kernel, whereas Inductor lowers the decomposed add/mean/as_strided/permute/var_mean/rsqrt/affine/cast/permute/view graph through generic normalization scheduling around a pooled intermediate; Inductor cannot do this today because its norm lowering scheduler does not select the direct pooled-residual LayerNorm schedule that preserves bf16 add/mean boundaries and the final output scope; the fix is SCHEDULER_FUSION: add a guarded pooled-residual LayerNorm lowering that preserves casts, strides, and output scope while reducing intermediate scheduling overhead."""

import torch
import triton
import triton.language as tl
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint

from oracle_harness import oracle_impl


EPS = 1.0e-6
triton_helpers.set_driver_to_gpu()
_DEVICE = DeviceProperties.create(torch.device("cuda"))


@triton.jit
def _round_to_bf16_f32(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton_heuristics.reduction(
    size_hints={"x": 131072, "r0_": 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={
        "signature": {
            "x0_ptr": "*bf16",
            "x1_ptr": "*bf16",
            "pooled_ptr": "*bf16",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
            "R0_BLOCK": "constexpr",
        },
        "device": _DEVICE,
        "constants": {},
        "configs": [
            {
                (0,): [["tt.divisibility", 16]],
                (1,): [["tt.divisibility", 16]],
                (2,): [["tt.divisibility", 16]],
                (3,): [["tt.divisibility", 16]],
            }
        ],
    },
    inductor_meta={
        "grid_type": "Grid1D",
        "kernel_name": "oracle_spatial_mean_768bb6a7efb9",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": False,
        "num_load": 2,
        "num_reduction": 1,
        "are_deterministic_algorithms_enabled": False,
        "assert_indirect_indexing": True,
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
        "dynamic_scale_rblock": True,
        "max_autotune": False,
        "max_autotune_pointwise": False,
        "min_split_scan_rblock": 256,
        "spill_threshold": 16,
        "store_cubin": False,
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
)
@triton.jit
def _spatial_mean_kernel(
    x0_ptr,
    x1_ptr,
    pooled_ptr,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    xnumel = 81920
    r0_numel = 81
    x_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r_offsets = tl.arange(0, R0_BLOCK)[None, :]
    r_mask = r_offsets < r0_numel

    channels = x_offsets % 640
    batches = x_offsets // 640
    acc = tl.full((XBLOCK, R0_BLOCK), 0.0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r_index = r0_offset + r_offsets
        r_mask = r_index < r0_numel
        offsets = batches * 51840 + channels + r_index * 640
        x0 = tl.load(x0_ptr + offsets, mask=r_mask, eviction_policy="evict_first", other=0.0)
        x1 = tl.load(x1_ptr + offsets, mask=r_mask, eviction_policy="evict_first", other=0.0)
        acc += tl.where(r_mask, (x0 + x1).to(tl.float32), 0.0)

    summed = tl.sum(acc, axis=1)[:, None]
    tl.store(pooled_ptr + x_offsets, summed * (1.0 / 81.0))


@triton.jit
def _layernorm_kernel(
    pooled_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_C: tl.constexpr,
    EPS_: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < 640
    pooled = tl.load(pooled_ptr + row * 640 + cols, mask=mask, other=0.0).to(tl.float32)
    pooled = tl.where(mask, pooled, 0.0)
    mean = tl.sum(pooled, axis=0) * (1.0 / 640.0)
    centered = pooled - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) * (1.0 / 640.0)
    invstd = tl.rsqrt(variance + EPS_)

    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (centered * invstd) * weight + bias
    tl.store(out_ptr + row * 640 + cols, out, mask=mask)


# 10a79381: two T([128,640,9,9], bf16, stride=(51840,1,5760,640)) inputs.
@oracle_impl(
    hardware="B200",
    point="10a79381",
    LN_BLOCK_C=1024,
    ln_num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    LN_BLOCK_C: int,
    ln_num_warps: int,
):
    x0, x1, weight, bias, shape0, stride0, shape1 = inputs
    pooled = torch.empty_strided(
        (int(shape1[0]), int(shape1[1])),
        (int(shape1[1]), 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    raw_stream = get_raw_stream(x0.device.index or 0)
    _spatial_mean_kernel.run(
        x0,
        x1,
        pooled,
        128 * 640,
        81,
        stream=raw_stream,
    )
    _layernorm_kernel[(128,)](
        pooled,
        weight,
        bias,
        pooled,
        BLOCK_C=LN_BLOCK_C,
        EPS_=EPS,
        num_warps=ln_num_warps,
        num_stages=3,
    )
    return pooled
